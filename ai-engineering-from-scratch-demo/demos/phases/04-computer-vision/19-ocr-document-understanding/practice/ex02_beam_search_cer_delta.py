"""Exercise 2 — beam search cer delta.

    **(Medium)** Replace greedy decoding with beam search (beam_width=5). Report CER delta. On which inputs does beam search win?

Reading of the exercise: the delta is real but tiny -- -0.0123, CER 0.6790 down to
0.6667 -- and the second question has an unusually sharp answer: beam search wins
on exactly one of the lesson's own 20 training lines, `xy910`, where greedy
spells `x910` and beam spells the reference. That line is the only one of the 20
whose length is 5, and `synthetic_line` inks every alnum symbol identically, so
it is also the only line whose rendered image is not bit-identical to the other
19. Beam search cannot win where the input carries no information; here it wins
on the single line that does. The mechanism is exact and checkable against the
lesson's own `ctc_loss`, whose forward algorithm marginalises the alignments that
`greedy_ctc_decode` only samples one of: the best path on that line has
probability 0.037392 and spells `x910`, while P(`xy910`) = 0.052094 beats
P(`x910`) = 0.048992. Prefix beam search maximises over labelings and so orders
them correctly, even though a 5-wide beam recovers only 78.9% of the winner's
mass. beam_width=5 is more than the problem needs: width 1 reproduces
`greedy_ctc_decode` on 20/20 lines and the CER saturates at width 3. Nothing is
downloaded; the model is the lesson's own `main()` run reproduced at its seeds.

Structure: `edit` is the stdlib Levenshtein, and the module-level `cer` and
`sweep` lambdas turn it into a rate and into a rate per beam width; `beam_decode`
is CTC prefix beam search over (blank-ending, symbol-ending) log-probability
pairs, returning the best labeling, its beam score and the number of prefix
expansions it cost; `exact_logp` reads a labeling's true log-probability out of
the lesson's own `ctc_loss` by undoing its `reduction="mean"` division, and so
serves as the exhaustive oracle both for the trained model and for the
three-step hand case that `hand_mat` builds; `train_lesson` reproduces
`main()`'s 200 Adam steps over its own `abc0..abc9` / `xy01..xy910` strings and
returns the log-prob tensor. At 146 code lines this sits above D14's 120-line
target and 4 clear of the ceiling: five checks over a training run, six beam
widths and one hand-built distribution.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "19-ocr-document-understanding"

STEPS, BATCH, MAXLEN, WIDTH, TIMESTEPS = 200, 8, 5, 5, 20
WIDTHS = (1, 2, 3, 5, 10, 25)
HAND = (0.45, 0.35, 0.20)         # blank, '0', '1' at each of three time steps
NEG = (-math.inf, -math.inf)
cer = lambda hyps, refs: sum(map(edit, hyps, refs)) / sum(map(len, refs))       # noqa: E731
sweep = lambda beams, refs: {w: cer(rows, refs) for w, rows in beams.items()}   # noqa: E731
hand_mat = lambda: [[math.log(p) for p in HAND] for _ in range(3)]              # noqa: E731


def edit(hyp, ref) -> int:
    prev = list(range(len(ref) + 1))
    for row, left in enumerate(hyp, 1):
        cur = [row]
        for col, right in enumerate(ref, 1):
            cur.append(min(prev[col] + 1, cur[-1] + 1, prev[col - 1] + (left != right)))
        prev = cur
    return prev[-1]


def beam_decode(np, mat, width):
    beams, expansions = {(): (0.0, -math.inf)}, 0
    for step in mat:
        nxt = {}
        for prefix, (blank_end, symbol_end) in beams.items():
            expansions += len(step)
            total, kept = np.logaddexp(blank_end, symbol_end), nxt.get(prefix, NEG)
            nxt[prefix] = (np.logaddexp(kept[0], total + step[0]), kept[1])
            for sym in range(1, len(step)):
                repeat = bool(prefix) and prefix[-1] == sym
                ext, grown = prefix + (sym,), nxt.get(prefix + (sym,), NEG)
                nxt[ext] = (grown[0], np.logaddexp(grown[1], (blank_end if repeat else total) + step[sym]))
                if repeat:
                    nxt[prefix] = (nxt[prefix][0], np.logaddexp(nxt[prefix][1], symbol_end + step[sym]))
        beams = dict(sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:width])
    best = max(beams.items(), key=lambda kv: np.logaddexp(*kv[1]))
    return list(best[0]), float(np.logaddexp(*best[1])), expansions


def exact_logp(torch, ref, log_probs, text) -> float:
    per_char = ref.ctc_loss(log_probs, torch.tensor([ref.VOCAB.index(ch) for ch in text]),
                            torch.tensor([log_probs.size(0)]), torch.tensor([len(text)]))
    return -per_char.item() * len(text)


def train_lesson(np, torch, ref):
    torch.manual_seed(0)
    np.random.seed(0)
    model = ref.TinyCRNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    strings = [f"abc{d}" for d in range(10)] + [f"xy{d}{d + 1}" for d in range(10)]
    lengths = torch.full((BATCH,), TIMESTEPS, dtype=torch.long)
    for _ in range(STEPS):
        imgs, targets, target_lens = ref.build_batch(
            [strings[i] for i in np.random.choice(len(strings), BATCH)], max_len=MAXLEN)
        loss = ref.ctc_loss(model(imgs), targets, lengths, target_lens)
        opt.zero_grad(); loss.backward(); opt.step()      # noqa: E702 - main()'s own update line
    model.eval()
    with torch.no_grad():
        return strings, model(ref.build_batch(strings, max_len=MAXLEN)[0])


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    strings, log_probs = train_lesson(np, torch, ref)
    mats = log_probs.transpose(0, 1).tolist()
    greedy = [ref.decode_to_str(ids) for ids in ref.greedy_ctc_decode(log_probs)]
    beams = {w: [ref.decode_to_str(beam_decode(np, mat, w)[0]) for mat in mats] for w in WIDTHS}
    won = [row for row in zip(strings, greedy, beams[WIDTH]) if row[1] != row[2]]
    line = strings.index(won[0][0])
    return {"greedy": cer(greedy, strings), "won": won, "cers": sweep(beams, strings), "line": line,
            "lengths": sorted({len(text) for text in strings}), "match": beams[1] == greedy,
            "outputs": len(set(greedy)), "strings": len(strings), "symbols": len(mats[0][0]),
            "cost": (beam_decode(np, mats[0], WIDTH)[2], len(mats[0]) * len(mats[0][0])),
            "exact": {text: exact_logp(torch, ref, log_probs[:, line:line + 1, :], text)  # noqa: E501
                      for text in won[0][:2]},
            "path": log_probs[:, line, :].max(dim=-1).values.sum().item(),
            "beam_score": beam_decode(np, mats[line], WIDTH)[1],
            "hand": {"greedy": 3 * math.log(HAND[0]), "paths": len(HAND) ** 3,
                     "beam": beam_decode(np, hand_mat(), WIDTH)[1],
                     "oracle": exact_logp(torch, ref, torch.tensor(hand_mat()).unsqueeze(1), "0")}}


def verify(result):
    won, hand, cers = result["won"], result["hand"], result["cers"]
    top, second = result["exact"][won[0][0]], result["exact"][won[0][1]]
    delta = cers[WIDTH] - result["greedy"]
    return [
        practice.Check(
            "ANSWER: beam_width=5 moves CER from 0.6790 to 0.6667 — a delta of -0.0123",
            cers[WIDTH] < result["greedy"],
            f"the lesson's own `main()` at its seeds ({STEPS} Adam steps over its {result['strings']} "
            f"`abc0..abc9` / `xy01..xy910` strings), decoded both ways: greedy {result['greedy']:.4f}, "
            f"beam-{WIDTH} {cers[WIDTH]:.4f}, delta {delta:+.4f} ({-delta / result['greedy']:.1%} rel.)"),
        practice.Check(
            "ANSWER: it wins on exactly one line — the only one whose image is not a collision",
            len(won) == 1 and won[0] == ("xy910", "x910", "xy910"),
            f"{len(won)} of {result['strings']} lines differ: reference {won[0][0]!r}, greedy {won[0][1]!r}, "
            f"beam {won[0][2]!r}. Lengths in the set are {result['lengths']} and `synthetic_line` inks every "
            f"alnum symbol alike, so that length-5 line is the only image unlike the other 19 -- greedy "
            f"collapses all 20 to {result['outputs']} outputs"),
        practice.Check(
            "MECHANISM: greedy maximises over paths, beam over labelings — the exact numbers cross",
            top > second > result["path"],
            f"from the lesson's `ctc_loss`, `reduction='mean'` undone: P({won[0][0]!r}) = {math.exp(top):.6f} "
            f"beats P({won[0][1]!r}) = {math.exp(second):.6f} by {math.exp(top - second):.4f}x, yet the best "
            f"of the {result['symbols']}**{TIMESTEPS} paths scores {math.exp(result['path']):.6f} and "
            f"spells {won[0][1]!r}"),
        practice.Check(
            "CONTROL: on a 3-step hand case the beam matches the lesson's CTC oracle to 1.0e-07",
            abs(hand["beam"] - hand["oracle"]) < 1e-06 and hand["oracle"] > hand["greedy"],
            f"a hand-built 3-step distribution of {HAND} over (blank, '0', '1'): greedy takes blank three "
            f"times for 0.45**3 = {math.exp(hand['greedy']):.6f}, while the {hand['paths']} paths "
            f"marginalised by `ctc_loss` put {math.exp(hand['oracle']):.6f} on '0', "
            f"{math.exp(hand['oracle'] - hand['greedy']):.4f}x more -- beam-{WIDTH} returns that to "
            f"{abs(hand['beam'] - hand['oracle']):.1e} nats, `F.ctc_loss`'s float32 residue"),
        practice.Check(
            "CONTROL: width 5 is over-provisioned and lossy, yet still orders a 1.06x gap correctly",
            result["match"] and cers[3] == cers[25] and result["beam_score"] < top,
            "CER by width: " + ", ".join(f"{w}={cers[w]:.4f}" for w in WIDTHS)
            + f". Width 1 reproduces `greedy_ctc_decode` on {result['strings']}/{result['strings']} lines "
            f"and nothing past 3 changes an output, while pruning costs mass: beam-{WIDTH} scores "
            f"{won[0][0]!r} at {result['beam_score']:.4f} against the exact {top:.4f}, "
            f"{math.exp(result['beam_score'] - top):.1%} of it, for {result['cost'][0]:,} expansions "
            f"against {result['cost'][1]:,} argmax comparisons"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
