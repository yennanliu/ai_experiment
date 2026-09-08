"""Exercise 1 — digit cer renderer collision.

    **(Easy)** Train the TinyCRNN on 5-digit random numeric strings for 500 steps. Report CER on a held-out set.

Reading of the exercise: the CER it asks for is 1.0000, and reporting that number
is only half the answer -- the interesting half is that no amount of training
could have moved it. `synthetic_line` picks its ink from `shade = 0.0 if
c.isalnum() else 0.5`, which is a function of the *character class*, not the
character: all 36 alnum symbols in the lesson's own `VOCAB` render to one
identical bitmap, so `build_batch(["12345"])` and `build_batch(["67890"])` are
bit-identical tensors. Every 5-digit string is the same image. The target then
carries 5*log(10) = 11.5129 nats that the input does not contain, which fixes an
exact floor of log(10) = 2.3026 nats/char on the lesson's own `ctc_loss`; the run
below plateaus at 2.6252, so the optimiser is working and the data is empty. CTC
resolves the ambiguity by collapsing to blank, and `greedy_ctc_decode` returns
the empty string on 50/50 held-out lines -- worse than the best constant guess,
which scores 0.8560. The control arm swaps in a renderer that gives each digit
its own shade and changes nothing else: same `TinyCRNN`, same `ctc_loss`, same
`greedy_ctc_decode`, 500 steps, CER 0.0680. Nothing is downloaded; both arms
train from scratch on `torch.manual_seed(0)`.

Structure: `edit` is the stdlib Levenshtein used for every CER here;
`render_batch` builds the (N, 1, 32, 80) batch through the lesson's own
`synthetic_line`, optionally over-painting each character cell with a
digit-specific shade, and returns the flat CTC targets alongside; `train_arm`
runs the exercise's own 500 steps of Adam at lr=1e-3 with batch 8, then decodes
50 held-out strings and reports CER, the loss plateau and the decode statistics;
`collision_probe` measures how many distinct bitmaps the renderer can emit and
what `build_batch` does with a character outside `VOCAB`; the module-level
`baseline` lambda scores the best constant guess. At 143 code lines this sits
above D14's 120-line target and 7 clear of the ceiling: six checks over two
independent 500-step training arms.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "19-ocr-document-understanding"

DIGITS, LENGTH, STEPS, BATCH, HELD = "0123456789", 5, 500, 8, 50
HEIGHT, WIDTH, CELL, TIMESTEPS, FLOOR = 32, 16 * 5, 16, 20, math.log(10)
baseline = lambda lines: min(sum(edit(digit * LENGTH, line) for line in lines)   # noqa: E731
                             for digit in DIGITS) / (HELD * LENGTH)


def edit(hyp, ref) -> int:
    prev = list(range(len(ref) + 1))
    for row, left in enumerate(hyp, 1):
        cur = [row]
        for col, right in enumerate(ref, 1):
            cur.append(min(prev[col] + 1, cur[-1] + 1, prev[col - 1] + (left != right)))
        prev = cur
    return prev[-1]


def render_batch(np, torch, ref, strings, sighted):
    imgs = np.ones((len(strings), 1, HEIGHT, WIDTH), dtype=np.float32)
    for row, text in enumerate(strings):
        line = ref.synthetic_line(text)
        for cell, char in enumerate(text) if sighted else ():
            line[6:HEIGHT - 6, cell * CELL + 2:cell * CELL + CELL - 2] = DIGITS.index(char) / 10.0
        imgs[row, 0, :, :line.shape[1]] = line
    return (torch.from_numpy(imgs),
            torch.tensor([ref.VOCAB.index(char) for text in strings for char in text]),
            torch.tensor([len(text) for text in strings], dtype=torch.long))


def train_arm(np, torch, ref, sighted) -> dict:
    torch.manual_seed(0)
    model = ref.TinyCRNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng, losses = np.random.default_rng(19), []
    lengths = torch.full((BATCH,), TIMESTEPS, dtype=torch.long)
    for _ in range(STEPS):
        strings = ["".join(rng.choice(list(DIGITS), LENGTH)) for _ in range(BATCH)]
        imgs, targets, target_lens = render_batch(np, torch, ref, strings, sighted)
        loss = ref.ctc_loss(model(imgs), targets, lengths, target_lens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    held = ["".join(rng.choice(list(DIGITS), LENGTH)) for _ in range(HELD)]
    with torch.no_grad():
        decoded = ref.greedy_ctc_decode(model(render_batch(np, torch, ref, held, sighted)[0]))
    preds = [ref.decode_to_str(ids) for ids in decoded]
    return {"cer": sum(map(edit, preds, held)) / (HELD * LENGTH), "distinct": len(set(preds)),
            "plateau": statistics.mean(losses[-50:]), "start": losses[0], "sample": preds[:3],
            "empty": sum(pred == "" for pred in preds), "held": held[:3],
            "exact": sum(pred == text for pred, text in zip(preds, held)),
            "constant": baseline(held)}


def collision_probe(np, torch, ref) -> dict:
    try:
        ref.build_batch(["a b"], max_len=3)
        outside = "no error"
    except ValueError as exc:
        outside = f"{type(exc).__name__}: {exc}"
    pair = [[render_batch(np, torch, ref, [text], seen)[0] for text in ("12345", "67890")]
            for seen in (False, True)]
    return {"renders": len({ref.synthetic_line(char).tobytes() for char in ref.VOCAB[1:]}),
            "shades": sorted(np.unique(ref.synthetic_line("12345")).tolist()), "outside": outside,
            "gap": sorted(np.unique(ref.synthetic_line(" ")).tolist()), "symbols": len(ref.VOCAB) - 1,
            "blind_equal": torch.equal(*pair[0]), "sighted_equal": torch.equal(*pair[1]),
            "pixels": int(pair[0][0].numel())}


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    return {"blind": train_arm(np, torch, ref, False), "vocab": len(ref.VOCAB),
            "sighted": train_arm(np, torch, ref, True),
            "collision": collision_probe(np, torch, ref)}


def verify(result):
    blind, sighted, coll = result["blind"], result["sighted"], result["collision"]
    return [
        practice.Check(
            "ANSWER: 500 steps on 5-digit random numeric strings gives held-out CER 1.0000",
            blind["cer"] == 1.0 and blind["empty"] == HELD,
            f"the lesson's own `TinyCRNN` + `ctc_loss` + `greedy_ctc_decode`, Adam at lr=1e-3, batch {BATCH}, "
            f"{STEPS} steps, {HELD} held-out strings: CER {blind['cer']:.4f}, {blind['exact']}/{HELD} exact, "
            f"and all {blind['empty']} decodes are the empty string -- {blind['distinct']} distinct output"),
        practice.Check(
            "FINDING: the renderer is character-blind, so no training run could have done better",
            coll["renders"] == 1 and coll["blind_equal"] and not coll["sighted_equal"],
            f"`synthetic_line` sets `shade = 0.0 if c.isalnum() else 0.5`, a function of the character *class*: "
            f"all {coll['symbols']} alnum symbols of the {result['vocab']}-symbol `VOCAB` give {coll['renders']} "
            f"distinct bitmap, shades {coll['shades']}, so the {coll['pixels']}-pixel tensors for '12345' and "
            f"'67890' are bit-identical. One shade per digit breaks the tie"),
        practice.Check(
            "MECHANISM: the loss plateau sits 14% above the exact information floor log(10)",
            FLOOR < blind["plateau"] < 1.2 * FLOOR,
            f"a constant image cannot carry the {LENGTH}*log(10) = {LENGTH * FLOOR:.4f} nats a uniform 5-digit "
            f"target needs, and `reduction='mean'` divides by target length, so nothing can beat log(10) = "
            f"{FLOOR:.4f} nats/char over {TIMESTEPS} time steps. Measured: {blind['start']:.3f} at step 0 "
            f"falling to {blind['plateau']:.4f} over the last 50 -- Adam found the floor, not a way past it"),
        practice.Check(
            "FINDING: CER 1.0 is worse than guessing — blank collapse loses to a constant digit",
            blind["constant"] < blind["cer"],
            f"CTC breaks the tie by putting its mass on blank, and a decode of \"\" costs one deletion per "
            f"reference character, i.e. CER exactly 1.0. Emitting the single best repeated digit instead scores "
            f"{blind['constant']:.4f} on the same held-out set, "
            f"{(1 - blind['constant']) / blind['constant']:.1%} better than the trained model"),
        practice.Check(
            "CONTROL: the same loop with a sighted renderer reaches CER 0.07 — the pipeline is fine",
            sighted["cer"] < 0.15 and sighted["exact"] > HELD // 2,
            f"over-painting each character cell with `DIGITS.index(c)/10` and changing nothing else -- same "
            f"`TinyCRNN`, `ctc_loss`, `greedy_ctc_decode`, {STEPS} steps and seed -- gives CER "
            f"{sighted['cer']:.4f}, {sighted['exact']}/{HELD} exact, {sighted['distinct']} distinct outputs, "
            f"plateau {sighted['plateau']:.4f}. Sample {sighted['sample']} vs targets {sighted['held']}"),
        practice.Check(
            "CONTROL: the 0.5 shade branch is dead code — `build_batch` cannot reach it",
            coll["outside"].startswith("ValueError") and coll["gap"] == [0.5, 1.0],
            f"`synthetic_line(' ')` does paint {coll['gap']}, but `build_batch` calls `VOCAB.index(c)` on every "
            f"character and `VOCAB` holds only blank plus the {coll['symbols']} alnum symbols, so "
            f"`build_batch(['a b'])` raises {coll['outside']}. Through the lesson's own batcher exactly two "
            f"shades are reachable: {coll['shades']}, ink and padding"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
