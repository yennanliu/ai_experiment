"""Exercise 4 -- an L2 penalty cuts the output norm 3x, leaves cosine at 0.186, and shrinks used and unused directions alike.

    Add a small L2 penalty on the projector weights and watch how it interacts with cosine alignment (cosine is scale-invariant, so the penalty mostly shrinks unused directions).

Reading of the exercise: the penalty `lam * (||fc1.W||^2 + ||fc2.W||^2)`
(weights, not biases) is added to the lesson's per-pair loss inside its own
`train()` (seed 0, 32 pairs, 200 steps, Adam 3e-4), for lam = 0, 1e-4, 1e-3
and 1e-2. Alignment is the trained projector's mean cosine over all 32 pairs.
"Used directions" are the 32-dimensional span of the frozen encoder's 32 CLS
vectors -- the only inputs fc1 ever sees; "unused" is the other 736 input
dimensions. fc1's norm in each subspace is compared with its value at init.

**ANSWER: the penalty shrinks the weights and leaves the alignment alone.**
Mean cosine is 0.1867 at lam = 0, 1e-4 and 1e-3 and 0.1862 at 1e-2, while the
mean projected-embedding norm falls 51.2 -> 43.9 -> 15.5 (lam 0, 1e-3, 1e-2)
and fc2's norm 13.2 -> 10.4 -> 4.3. Cosine ignores scale, so the penalty is
nearly free until it starts to fight the one direction the head learned.

**FINDING: it does not "mostly" shrink unused directions -- it shrinks all of
them at a similar rate.** At lam = 1e-3 fc1 keeps x0.70 of its norm in the
used span and x0.61 in the unused one (x0.26 and x0.19 at 1e-2). The unused
part only loses more in absolute terms because at init it holds 96% of
fc1's squared norm. Without a penalty the unused part is almost untouched
(x1.002, while the used part grows x1.09): its gradient is exactly zero, and it moves only because Adam's
per-element step leaks outside the gradient's span.

Structure: `run()` wraps `train()`'s projector class to record init weights
and adds the penalty to its loss; `measure()` splits fc1 with a QR basis of
the 32 CLS vectors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "60-projection-layer-modality-align"
LAMS = (0.0, 1e-4, 1e-3, 1e-2)


def run(ref, lam):
    """ref.train() with an L2 weight penalty; returns (projector, fc1 init, encoder)."""
    seen = {}
    make_proj, make_enc, base_loss = ref.MLPProjector, ref.VisionEncoder, ref.cosine_alignment_loss

    def projector(i, h, o):
        seen["proj"] = make_proj(i, h, o)
        seen["w1"] = seen["proj"].fc1.weight.detach().clone()
        return seen["proj"]

    def encoder(cfg):
        seen["enc"] = make_enc(cfg)
        return seen["enc"]

    def loss(image_emb, text_emb):
        p = seen["proj"]
        return base_loss(image_emb, text_emb) + lam * (p.fc1.weight.square().sum() + p.fc2.weight.square().sum())

    saved = (make_proj, make_enc, base_loss)
    ref.MLPProjector, ref.VisionEncoder, ref.cosine_alignment_loss = projector, encoder, loss
    try:
        with parity.quiet():
            proj, _ = ref.train(ref.AlignConfig())
    finally:
        ref.MLPProjector, ref.VisionEncoder, ref.cosine_alignment_loss = saved
    return proj, seen["w1"], seen["enc"]


def measure(ref, proj, w1_init, enc):
    cfg = ref.AlignConfig()
    text = ref.MockTextEmbedding(cfg.vocab_size, cfg.text_hidden, seed=cfg.seed + 1)
    pairs = [ref.make_pair(cfg.seed + 1000 + i, cfg.vocab_size, cfg.max_caption_len)
             for i in range(cfg.pairs)]
    with torch.no_grad():
        x = torch.cat([enc(img)[1] for img, _ in pairs])
        out, t = proj(x), torch.cat([text(ids) for _, ids in pairs])
        basis = torch.linalg.qr(x.T).Q

        def split(w):
            used = w @ basis @ basis.T
            return used.norm().item(), (w - used).norm().item()

        (u0, n0), (u1, n1) = split(w1_init), split(proj.fc1.weight)
    return {"cos": F.cosine_similarity(out, t).mean().item(), "out_norm": out.norm(dim=-1).mean().item(),
            "fc2": proj.fc2.weight.norm().item(), "used": u1 / u0, "unused": n1 / n0,
            "unused_share": n0**2 / (u0**2 + n0**2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {lam: measure(ref, *run(ref, lam)) for lam in LAMS}


def near(a, b, tol=2e-3):
    return abs(a - b) <= tol


def verify(result):
    r0, r3, r2 = result[0.0], result[1e-3], result[1e-2]
    cos = [result[lam]["cos"] for lam in LAMS]
    return [
        practice.Check(
            "ANSWER: the penalty shrinks the weights and leaves the alignment alone",
            all([
                all(near(c, 0.1867) for c in cos[:3]),
                near(cos[3], 0.1862),
                near(r0["out_norm"], 51.2, 0.1),
                near(r3["out_norm"], 43.9, 0.1),
                near(r2["out_norm"], 15.5, 0.1),
                near(r2["fc2"], 4.33, 0.01),
            ]),
            f"cos {', '.join(f'{c:.4f}' for c in cos)}; embedding norm {r0['out_norm']:.1f} / "
            f"{r3['out_norm']:.1f} / {r2['out_norm']:.1f}; fc2 norm {r0['fc2']:.2f} / "
            f"{r3['fc2']:.2f} / {r2['fc2']:.2f} (lam 0, 1e-3, 1e-2)",
        ),
        practice.Check(
            "FINDING: it shrinks used and unused directions at a similar rate",
            all([
                near(r3["used"], 0.70, 0.01),
                near(r3["unused"], 0.61, 0.01),
                near(r2["used"], 0.26, 0.01),
                near(r2["unused"], 0.19, 0.01),
                near(r0["unused_share"], 0.96, 0.005),
                near(r0["unused"], 1.002),
                near(r0["used"], 1.09, 0.01),
            ]),
            f"fc1 norm kept, used/unused: lam 1e-3 x{r3['used']:.2f}/x{r3['unused']:.2f}, "
            f"lam 1e-2 x{r2['used']:.2f}/x{r2['unused']:.2f}, lam 0 x{r0['used']:.2f}/x{r0['unused']:.3f}; "
            f"unused share of init norm^2 {r0['unused_share']:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
