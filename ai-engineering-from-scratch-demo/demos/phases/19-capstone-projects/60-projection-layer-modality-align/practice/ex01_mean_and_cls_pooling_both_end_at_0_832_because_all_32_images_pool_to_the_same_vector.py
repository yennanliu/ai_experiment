"""Exercise 1 -- mean pooling and CLS pooling both end at 0.832, because all 32 pooled images are the same vector.

    Replace CLS pooling with mean pooling over the 196 patch tokens and compare final loss after 200 steps. Mean pooling usually trains faster on synthetic data; CLS is more sample-efficient on natural images.

Reading of the exercise: the lesson's own `train()` is run twice at its
shipped config (seed 0, 32 pairs, 200 steps, Adam 3e-4), once as shipped and
once with the frozen encoder's pooled vector swapped from `tokens[:, 0]` to
`tokens[:, 1:].mean(1)` (the 196 patch tokens). "Final loss" is read three
ways, because `train()` reports the loss of one pair: the last step's loss,
the mean over the last 32 steps (one pass over the pairs), and the loss of the
trained projector over all 32 pairs at once. "Trains faster" is read as the
mean loss of the first 32-step pass.

**ANSWER: no measurable difference.** Final step loss 0.7984 (CLS) vs 0.7990
(mean), last-pass mean 0.8320 vs 0.8322, all-pairs loss 0.8133 vs 0.8133.
The first 32-step pass averages 1.001 with mean pooling and 1.005 with CLS,
so mean pooling does not train measurably faster here; the gap at every
reading is under 0.005.

**FINDING: pooling cannot matter, because the frozen random encoder gives
every image the same vector.** The 32 pooled vectors have pairwise cosine
>= 0.99996 under either pooling. A projector that ignores the image entirely
and emits the one best constant direction (the normalised sum of the 32 unit
caption vectors) scores 0.8130; both trained projectors sit 0.0003 above it.

**FINDING: nothing is aligned -- shuffling the captions changes nothing.**
Pair every image with another pair's caption and the last-pass mean is 0.8321,
against 0.8320 for the true pairing. The lesson's "about 0.80" is one pair's
loss at step 199, not the model's loss.

Structure: `run()` swaps module globals `train()` looks up, runs it silently
and records the encoder it built; `pooled()` re-encodes the 32 pairs with that
encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "60-projection-layer-modality-align"


def run(ref, **patch):
    """ref.train() at the shipped config with module globals swapped; restores them."""
    seen, make_enc = {}, patch.pop("VisionEncoder", ref.VisionEncoder)

    def encoder(cfg):
        seen["enc"] = make_enc(cfg)
        return seen["enc"]

    patch["VisionEncoder"] = encoder
    saved = {k: getattr(ref, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(ref, k, v)
        with parity.quiet():
            proj, stats = ref.train(ref.AlignConfig())
    finally:
        for k, v in saved.items():
            setattr(ref, k, v)
    return proj, stats, seen["enc"]


def mean_pool_encoder(ref):
    make = ref.VisionEncoder

    def build(cfg):
        enc = make(cfg)
        forward = enc.forward
        enc.forward = lambda x, store_attn=False: (lambda t: (t, t[:, 1:].mean(1)))(forward(x)[0])
        return enc

    return build


def pooled(ref, enc):
    """(pooled image vectors, caption vectors) for the 32 shipped pairs."""
    cfg = ref.AlignConfig()
    text = ref.MockTextEmbedding(cfg.vocab_size, cfg.text_hidden, seed=cfg.seed + 1)
    pairs = [ref.make_pair(cfg.seed + 1000 + i, cfg.vocab_size, cfg.max_caption_len)
             for i in range(cfg.pairs)]
    with torch.no_grad():
        return torch.cat([enc(img)[1] for img, _ in pairs]), torch.cat([text(ids) for _, ids in pairs])


def summary(ref, proj, stats, enc):
    x, t = pooled(ref, enc)
    n = F.normalize(x, dim=-1)
    sims = (n @ n.T)[~torch.eye(len(x), dtype=torch.bool)]
    with torch.no_grad():
        full = 1 - F.cosine_similarity(proj(x), t).mean().item()
    unit = F.normalize(t, dim=-1)
    return {"last": stats.final_loss, "pass": sum(stats.losses[-32:]) / 32,
            "first": sum(stats.losses[:32]) / 32, "full": full, "min_sim": sims.min().item(),
            "floor": 1 - unit.sum(0).norm().item() / len(t)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped_pair = ref.make_pair

    def shuffled(seed, vocab_size, max_len):
        other = 1000 + (seed - 1000 + 5) % 32
        return shipped_pair(seed, vocab_size, max_len)[0], shipped_pair(other, vocab_size, max_len)[1]

    return {"cls": summary(ref, *run(ref)),
            "mean": summary(ref, *run(ref, VisionEncoder=mean_pool_encoder(ref))),
            "shuffled": run(ref, make_pair=shuffled)[1].losses[-32:]}


def near(a, b, tol=2e-3):
    return abs(a - b) <= tol


def verify(result):
    c, m, sh = result["cls"], result["mean"], sum(result["shuffled"]) / 32
    gaps = [abs(c[k] - m[k]) for k in ("last", "pass", "full", "first")]
    return [
        practice.Check(
            "ANSWER: no measurable difference",
            all([near(c["last"], 0.7984), near(m["last"], 0.7990), near(c["pass"], 0.8320),
                 near(m["pass"], 0.8322), near(c["full"], 0.8133), max(gaps[:3]) < 1e-3,
                 near(c["first"], 1.005), near(m["first"], 1.001), gaps[3] < 5e-3]),
            f"CLS last/pass/full {c['last']:.4f}/{c['pass']:.4f}/{c['full']:.4f}, mean "
            f"{m['last']:.4f}/{m['pass']:.4f}/{m['full']:.4f}; first pass {c['first']:.3f} vs {m['first']:.3f}",
        ),
        practice.Check(
            "FINDING: pooling cannot matter, every image pools to the same vector",
            all([min(c["min_sim"], m["min_sim"]) >= 0.99996, near(c["floor"], 0.8130),
                 0 <= c["full"] - c["floor"] < 1e-3, 0 <= m["full"] - m["floor"] < 1e-3]),
            f"min pairwise cos CLS {c['min_sim']:.5f}, mean {m['min_sim']:.5f}; constant-output "
            f"floor {c['floor']:.4f}",
        ),
        practice.Check(
            "FINDING: nothing is aligned, shuffled captions give the same loss",
            all([near(sh, 0.8321), abs(sh - c["pass"]) < 1e-3, c["last"] < c["pass"] - 0.03]),
            f"shuffled last-pass {sh:.4f} vs true pairs {c['pass']:.4f}; step-199 loss {c['last']:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
