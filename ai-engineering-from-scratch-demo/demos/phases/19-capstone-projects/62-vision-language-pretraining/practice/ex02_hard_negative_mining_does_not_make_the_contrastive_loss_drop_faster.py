"""Exercise 2 -- hard-negative mining does not make the contrastive loss drop faster.

    Add a hard-negative mining step: every other batch, select the hardest off-diagonal pair from the previous batch and append it. Train and inspect whether contrastive loss drops faster.

Reading of the exercise: an off-diagonal entry (i, j) is image i scored
against caption j. It is not a pair to learn as a match, so "append it" is
read as appending corpus items i and j (each with its own caption) to the
next batch, so the confusable image and caption meet again as negatives. The
hardest entry is the largest off-diagonal similarity in the previous batch,
and it is appended on every odd step (skipping an item already drawn). The
rest is the lesson's 50-step run. Seeds 0-4 are run with and without mining.
"Drops faster" is judged on the contrastive loss of 12 fixed 16-pair blocks
of the corpus at the end of the run, and on the training loss the lesson
prints.

**ANSWER: no.** After 50 steps the fixed-block contrastive loss is lower
with mining on 1 seed in 5, and the mean over seeds moves by +0.010
(worse). The training loss over the last 10 steps is higher with mining on
all 5 seeds.

**FINDING: the training loss the lesson prints penalises mining for batch
size alone.** A mined batch holds 17 or 18 pairs, not 16, and at chance the
InfoNCE loss is ln N: ln(18/16) = 0.118 higher before anything is learned.
On seed 0 the mined odd steps average 2.849 against 2.733 on its even steps;
without mining the odd and even steps average 2.764 and 2.734.

**FINDING: the doc's start and end values are not what the run prints.**
The doc says contrastive loss starts near ln(16) = 2.77 and drops toward 2.4.
Step 0 prints 3.046, above chance, because a scale of 14.3 on random
embeddings spreads the logits. Step 49 prints 2.362, but that is one batch:
the mean of the last 10 steps is 2.541 and step 45 printed 2.668.

Structure: `train()` replays the lesson's loop with a spy on `info_nce_loss`
(restored afterwards) to read each batch's similarity matrix; `fixed()` is
the end-of-run measure.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "62-vision-language-pretraining"
SEEDS = range(5)


def fixed(ref, model, corpus):
    """Mean InfoNCE over 12 fixed blocks of 16 corpus pairs."""
    with torch.no_grad():
        imgs, ids = ref.sample_batch(corpus, list(range(192)))
        img, txt = model.encode_image(imgs)[1], model.text_encoder(ids)
        losses = [ref.info_nce_loss(img[b:b + 16], txt[b:b + 16], model.log_tau)[0].item()
                  for b in range(0, 192, 16)]
    return sum(losses) / len(losses)


def train(ref, seed, mine):
    cfg = ref.PretrainConfig(seed=seed)
    torch.manual_seed(cfg.seed)
    model = ref.MultimodalModel(cfg).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    corpus = ref.make_mock_corpus(cfg.seed + 1, cfg.n_pairs, cfg.text_vocab, cfg.max_text_len)
    rng = np.random.default_rng(cfg.seed + 2)
    saved, seen, hard, hist, sizes = ref.info_nce_loss, {}, [], [], []

    def spy(image_emb, text_emb, log_tau):
        loss, sim = saved(image_emb, text_emb, log_tau)
        seen["sim"] = sim.detach().clone().fill_diagonal_(-math.inf)
        return loss, sim

    ref.info_nce_loss = spy
    try:
        for step in range(cfg.steps):
            idx = rng.choice(len(corpus), size=cfg.batch_size, replace=False).tolist()
            if mine and step % 2 == 1:
                idx += [k for k in hard if k not in idx]
            contrast, lm, _ = model(*ref.sample_batch(corpus, idx))
            opt.zero_grad(set_to_none=True)
            (contrast + cfg.lm_weight * lm).backward()
            opt.step()
            i, j = divmod(int(seen["sim"].argmax()), len(idx))
            hard = sorted({idx[i], idx[j]})
            hist.append(contrast.item())
            sizes.append(len(idx))
    finally:
        ref.info_nce_loss = saved
    return {"fixed": fixed(ref, model, corpus), "hist": hist, "sizes": sorted(set(sizes))}


def per_seed(runs, read):
    """[[plain, mined], ...] over the seeds."""
    return [[read(runs[(s, m)]) for m in (False, True)] for s in SEEDS]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {(s, m): train(ref, s, m) for s in SEEDS for m in (False, True)}
    base, mined = runs[(0, False)]["hist"], runs[(0, True)]["hist"]
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "fixed": per_seed(runs, lambda r: round(r["fixed"], 4)),
        "tail": per_seed(runs, lambda r: round(sum(r["hist"][-10:]) / 10, 4)),
        "sizes": runs[(0, True)]["sizes"],
        "odd_even": [round(sum(h[k::2]) / 25, 3) for h in (base, mined) for k in (1, 0)],
        "printed": [round(base[i], 3) for i in (0, 45, 49)],
        "doc": "`ln(16) = 2.77` toward 2.4" in doc,
    }


def verify(result):
    r = result
    wins = sum(m < b for b, m in r["fixed"])
    shift = round(sum(m - b for b, m in r["fixed"]) / len(r["fixed"]), 3)
    worse_tail = sum(m > b for b, m in r["tail"])
    return [
        practice.Check(
            "ANSWER: no",
            (wins, shift, worse_tail) == (1, 0.01, 5),
            f"fixed-block loss [plain, mined] per seed {r['fixed']}: mining lower on {wins}/5, "
            f"mean shift {shift:+}; last-10 training loss higher with mining on {worse_tail}/5",
        ),
        practice.Check(
            "FINDING: the training loss the lesson prints penalises mining for batch size alone",
            r["sizes"] == [16, 17, 18] and round(math.log(18 / 16), 3) == 0.118
            and r["odd_even"] == [2.764, 2.734, 2.849, 2.733],
            f"batch sizes {r['sizes']}; odd/even step means plain {r['odd_even'][:2]}, "
            f"mined {r['odd_even'][2:]}; ln(18/16) = {math.log(18 / 16):.3f}",
        ),
        practice.Check(
            "FINDING: the doc's start and end values are not what the run prints",
            r["doc"] and r["printed"] == [3.046, 2.668, 2.362] and round(r["tail"][0][0], 3) == 2.541,
            f"steps 0/45/49 print {r['printed']} vs doc 2.77 -> 2.4; last-10 mean {r['tail'][0][0]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
