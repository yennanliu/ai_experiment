"""Exercise 5 -- the LM loss does not regress ranking, and it never reaches the projection the doc says it trains.

    Train the same model with `lm_weight = 0` and again with `lm_weight = 1`. Compare contrastive loss; the LM loss should not regress the ranking objective.

Reading of the exercise: `PretrainConfig(lm_weight=0.0)` and
`PretrainConfig(lm_weight=1.0)` are each trained with the lesson's loop
(the same loop exercise 1 checks against `ref.train`) on seeds 0-4. Because one
batch's loss is noisy, the contrastive loss is read on 12 fixed 16-pair
blocks of the corpus, at the lesson's 50 steps and at 300. "Regress" is
judged against the spread between seeds.

**ANSWER: it does not regress.** Mean fixed-block contrastive loss over the
5 seeds is 2.521 without the LM loss and 2.517 with it at 50 steps, and 0.596
against 0.585 at 300. With the LM loss the value is lower on 4 of 5 seeds at
50 steps and 3 of 5 at 300. The differences (at most 0.016 at 50 steps) are
small against the seed spread at 300 steps (0.351 to 0.706).

**FINDING: the LM gradient reaches the encoder but not the projection.** On
one batch at init, the LM loss gives the ViT encoder a gradient of total size
190.6 and the projector and text encoder exactly 0. The decoder reads the
encoder's patch tokens, not the projected embedding. The doc's table says LM
affects "Encoder + projection + decoder", and the prose says "only the
decoder receives LM-loss gradient". Both are wrong. The contrastive loss
gives the decoder exactly 0.

**FINDING: with lm_weight = 0 the caption loss stays above uniform.** It
ends at 6.39-6.42 on the 5 seeds, above ln 512 = 6.238, because nothing
trains the decoder. With lm_weight = 1 it falls to 4.78-4.82 at 50 steps.

Structure: `train()` replays the lesson's loop and reads the fixed blocks at
the marks; `grads()` backpropagates each loss separately.
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
SEEDS, PARTS = range(5), ("encoder", "projector", "text_encoder", "decoder")


def blocks(ref, model, corpus):
    with torch.no_grad():
        imgs, ids = ref.sample_batch(corpus, list(range(192)))
        img, txt = model.encode_image(imgs)[1], model.text_encoder(ids)
        losses = [ref.info_nce_loss(img[b:b + 16], txt[b:b + 16], model.log_tau)[0].item()
                  for b in range(0, 192, 16)]
        lm = model(imgs, ids)[1].item()
    return sum(losses) / len(losses), lm


def train(ref, seed, weight):
    cfg = ref.PretrainConfig(seed=seed, lm_weight=weight)
    torch.manual_seed(cfg.seed)
    model = ref.MultimodalModel(cfg).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    corpus = ref.make_mock_corpus(cfg.seed + 1, cfg.n_pairs, cfg.text_vocab, cfg.max_text_len)
    rng = np.random.default_rng(cfg.seed + 2)
    out = {}
    for step in range(1, 301):
        idx = rng.choice(len(corpus), size=cfg.batch_size, replace=False).tolist()
        contrast, lm, _ = model(*ref.sample_batch(corpus, idx))
        opt.zero_grad(set_to_none=True)
        (contrast + cfg.lm_weight * lm).backward()
        opt.step()
        if step in (50, 300):
            out[step] = blocks(ref, model, corpus)
    return out


def grads(ref):
    """Total |grad| per module from each loss alone, on one batch at init."""
    cfg = ref.PretrainConfig()
    torch.manual_seed(cfg.seed)
    model = ref.MultimodalModel(cfg)
    corpus = ref.make_mock_corpus(1, 16, cfg.text_vocab, cfg.max_text_len)
    contrast, lm, _ = model(*ref.sample_batch(corpus, list(range(16))))
    out = {}
    for name, loss in (("lm", lm), ("contrast", contrast)):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        out[name] = [round(sum(p.grad.abs().sum().item() for p in getattr(model, part).parameters()
                               if p.grad is not None), 1) for part in PARTS]
    return out


def tables(runs):
    nce = {m: [[runs[(s, w)][m][0] for w in (0.0, 1.0)] for s in SEEDS] for m in (50, 300)}
    lm = {w: [round(runs[(s, w)][50 if w else 300][1], 2) for s in SEEDS] for w in (0.0, 1.0)}
    return nce, lm


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {(s, w): train(ref, s, w) for s in SEEDS for w in (0.0, 1.0)}
    doc, (nce, lm) = parity.doc_text(PHASE, LESSON), tables(runs)
    return {
        "nce": nce, "lm": lm, "grads": grads(ref),
        "doc": ["| Encoder + projection + decoder |" in doc,
                "only the decoder receives LM-loss gradient" in doc],
    }


def stats(pairs):
    """Mean, seeds where lm 1 is lower, largest gap, and range over [lm 0, lm 1] pairs."""
    plain, joint = zip(*pairs)
    return {"mean": [round(sum(plain) / 5, 3), round(sum(joint) / 5, 3)],
            "lower": sum(map(float.__gt__, plain, joint)),
            "gap": round(max(abs(a - b) for a, b in pairs), 3),
            "spread": [round(min(plain + joint), 3), round(max(plain + joint), 3)]}


def verify(result):
    r, g, lm0, lm1 = result, result["grads"], result["lm"][0.0], result["lm"][1.0]
    st = {m: stats(r["nce"][m]) for m in (50, 300)}
    mean, lower = {m: st[m]["mean"] for m in st}, {m: st[m]["lower"] for m in st}
    return [
        practice.Check(
            "ANSWER: it does not regress",
            (mean, lower, st[50]["gap"], st[300]["spread"])
            == ({50: [2.521, 2.517], 300: [0.596, 0.585]}, {50: 4, 300: 3}, 0.016, [0.351, 0.706]),
            f"mean fixed-block loss [lm 0, lm 1] {mean}; lm 1 lower on {lower} seeds; max gap "
            f"at 50 steps {st[50]['gap']}; seed spread at 300 {st[300]['spread']}",
        ),
        practice.Check(
            "FINDING: the LM gradient reaches the encoder but not the projection",
            (r["doc"], g["lm"][:3], g["contrast"][3]) == ([True, True], [190.6, 0.0, 0.0], 0.0),
            f"|grad| {dict(zip(PARTS, g['lm']))} from LM, {dict(zip(PARTS, g['contrast']))} "
            f"from contrast; doc claims {r['doc']}",
        ),
        practice.Check(
            "FINDING: with lm_weight = 0 the caption loss stays above uniform",
            min(lm0) > math.log(512) and [min(lm0), max(lm0), min(lm1), max(lm1)] == [6.39, 6.42, 4.78, 4.82],
            f"LM loss lm 0 at 300 steps {lm0}, lm 1 at 50 steps {lm1}; ln 512 = {math.log(512):.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
