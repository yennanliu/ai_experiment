"""Exercise 3 -- a single linear layer matches the two-layer MLP within 0.001, and both emit one constant direction.

    Swap the two-layer MLP for a single linear layer and quantify the loss gap. The non-linearity matters more on natural image features and less on synthetic ones.

Reading of the exercise: the lesson's own `train()` (seed 0, 32 pairs, 200
steps, Adam 3e-4) is run as shipped and again with `MLPProjector` replaced by
`nn.Linear(768, 512)`. The gap is quantified on three readings of "loss": the
step-199 loss `train()` reports, the mean over the last 32 steps (one pass
over the pairs), and the trained projector's loss over all 32 pairs at once.

**ANSWER: the gap is within noise.** Step-199 loss 0.7984 (MLP) vs 0.7978
(linear), last-pass mean 0.8320 vs 0.8318, all-pairs 0.8133 vs 0.8133: the
linear layer is 0.0006 better at step 199 and within 0.0002 elsewhere, with
393,728 parameters against the MLP's 1,312,256 (the lesson's "1.3M" is
right).

**FINDING: neither projector uses the image.** The 32 projected image
embeddings have pairwise cosine >= 0.9999 for both heads: each learned one
output direction. The best constant direction (the normalised sum of the 32
unit caption vectors) scores 0.8130, and both heads end within 0.0005 of it.
The inputs leave nothing else to learn: the frozen random encoder's 32 CLS
vectors have pairwise cosine >= 0.99996. So "the non-linearity matters less
on synthetic features" holds here only because no projector can do better
than a constant.

Structure: `run()` swaps `train()`'s projector class and records the encoder
it built; `measure()` re-encodes the 32 pairs and scores the trained head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "60-projection-layer-modality-align"


def run(ref, projector=None):
    """ref.train() at the shipped config, optionally with another projector class."""
    seen, make_enc = {}, ref.VisionEncoder

    def encoder(cfg):
        seen["enc"] = make_enc(cfg)
        return seen["enc"]

    saved = (ref.VisionEncoder, ref.MLPProjector)
    ref.VisionEncoder, ref.MLPProjector = encoder, projector or ref.MLPProjector
    try:
        with parity.quiet():
            proj, stats = ref.train(ref.AlignConfig())
    finally:
        ref.VisionEncoder, ref.MLPProjector = saved
    return proj, stats, seen["enc"]


def min_offdiag_cos(x):
    n = F.normalize(x, dim=-1)
    return (n @ n.T)[~torch.eye(len(x), dtype=torch.bool)].min().item()


def measure(ref, proj, stats, enc):
    cfg = ref.AlignConfig()
    text = ref.MockTextEmbedding(cfg.vocab_size, cfg.text_hidden, seed=cfg.seed + 1)
    pairs = [ref.make_pair(cfg.seed + 1000 + i, cfg.vocab_size, cfg.max_caption_len)
             for i in range(cfg.pairs)]
    with torch.no_grad():
        x = torch.cat([enc(img)[1] for img, _ in pairs])
        t = torch.cat([text(ids) for _, ids in pairs])
        out = proj(x)
    return {"last": stats.final_loss, "pass": sum(stats.losses[-32:]) / 32,
            "full": 1 - F.cosine_similarity(out, t).mean().item(),
            "params": sum(p.numel() for p in proj.parameters()),
            "out_cos": min_offdiag_cos(out), "in_cos": min_offdiag_cos(x),
            "floor": 1 - F.normalize(t, dim=-1).sum(0).norm().item() / len(t)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"mlp": measure(ref, *run(ref)),
            "linear": measure(ref, *run(ref, lambda i, h, o: nn.Linear(i, o)))}


def near(a, b, tol=2e-3):
    return abs(a - b) <= tol


def verify(result):
    m, lin = result["mlp"], result["linear"]
    return [
        practice.Check(
            "ANSWER: the gap is within noise",
            all([
                near(m["last"], 0.7984),
                near(lin["last"], 0.7978),
                near(m["pass"], 0.8320),
                near(lin["pass"], 0.8318),
                near(m["full"], 0.8133),
                near(lin["full"], 0.8133),
                all(abs(m[k] - lin[k]) < 1e-3 for k in ("last", "pass", "full")),
                (m["params"], lin["params"]) == (1_312_256, 393_728),
            ]),
            f"MLP last/pass/full {m['last']:.4f}/{m['pass']:.4f}/{m['full']:.4f}, linear "
            f"{lin['last']:.4f}/{lin['pass']:.4f}/{lin['full']:.4f}; params {m['params']:,} vs "
            f"{lin['params']:,}",
        ),
        practice.Check(
            "FINDING: neither projector uses the image",
            all([
                min(m["out_cos"], lin["out_cos"]) >= 0.9999,
                m["in_cos"] >= 0.99996,
                near(m["floor"], 0.8130),
                all(0 <= r["full"] - r["floor"] < 5e-4 for r in (m, lin)),
            ]),
            f"min pairwise cos of outputs: MLP {m['out_cos']:.5f}, linear {lin['out_cos']:.5f}; "
            f"of CLS inputs {m['in_cos']:.5f}; constant-output floor {m['floor']:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
