"""Exercise 2 -- under Adam a fixed tau from 0.01 to 100 trains the same projector, and a learned tau only falls.

    Add a learned scalar temperature to the cosine loss (`cos / tau`) and observe what happens when `tau` is too small (gradient noise) or too large (loss plateaus high).

Reading of the exercise: the per-pair loss becomes `1 - cos / tau`, run through
the lesson's own `train()` (seed 0, 32 pairs, Adam 3e-4). "Too small" and "too
large" are probed with fixed tau = 0.01, 1 and 100 for the shipped 200 steps;
"learned" is a `log_tau` parameter (tau starts at 1) attached to the projector,
so `train()`'s own Adam updates it, run for 600 steps. The alignment actually
reached is read as the cosine `(1 - loss) * tau`, averaged over the last 32
steps.

**ANSWER: tau changes the printed loss, not the training.** With fixed tau the
last-pass cosine is 0.1680 at tau = 0.01, 1 and 100 (spread under 1e-4) while
the step-199 loss reads -19.16, 0.798 and 0.998. So tau = 100 does look like
a high plateau and tau = 0.01 shows no gradient noise at all: the gradient is
scaled by exactly 1/tau and Adam divides that scale out. A learned tau does
not settle: it runs 1.000 -> 0.949 after 200 steps and 0.833 after 600, and
it keeps falling as long as the mean cosine is positive.

**FINDING: `1 - cos / tau` has no minimum in tau.** For any positive cosine
the loss goes to minus infinity as tau goes to 0, so a learned tau can lower
the loss without any change in alignment. From step 200 to step 600 the
last-pass loss drops 0.824 -> 0.791 while the cosine only moves 0.168 ->
0.175; at tau = 1 that cosine would score 0.825, so the whole drop is tau
shrinking. The temperature only has a stable optimum inside a softmax over
negatives (the InfoNCE of lesson 62), not in this per-pair loss.

Structure: `run()` swaps `train()`'s loss (and, for the learned case, its
projector class) and returns the step losses and the tau trajectory.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "60-projection-layer-modality-align"
FIXED = (0.01, 1.0, 100.0)


def run(ref, tau=None, steps=200):
    """Step losses, cosines and taus from ref.train() with the loss `1 - cos / tau`."""
    make, box, taus = ref.MLPProjector, [], []

    def projector(i, h, o):
        m = make(i, h, o)
        m.log_tau = nn.Parameter(torch.zeros(()))
        box.append(m)
        return m

    def loss(image_emb, text_emb):
        t = torch.tensor(tau) if tau is not None else box[-1].log_tau.exp()
        taus.append(t.item())
        return 1 - F.cosine_similarity(image_emb, text_emb).mean() / t

    saved = (ref.MLPProjector, ref.cosine_alignment_loss)
    ref.MLPProjector, ref.cosine_alignment_loss = projector, loss
    try:
        with parity.quiet():
            _, stats = ref.train(dataclasses.replace(ref.AlignConfig(), steps=steps))
    finally:
        ref.MLPProjector, ref.cosine_alignment_loss = saved
    cos = [(1 - lo) * t for lo, t in zip(stats.losses, taus)]
    return {"loss": stats.losses, "cos": cos, "tau": taus}


def last_pass(values, end):
    return sum(values[end - 32 : end]) / 32


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fixed = {tau: run(ref, tau) for tau in FIXED}
    learned = run(ref, steps=600)
    return {
        "fixed_cos": {t: last_pass(r["cos"], 200) for t, r in fixed.items()},
        "fixed_loss": {t: r["loss"][-1] for t, r in fixed.items()},
        "tau": [learned["tau"][0], max(learned["tau"]), learned["tau"][199], learned["tau"][-1]],
        "learned_cos": [last_pass(learned["cos"], 200), last_pass(learned["cos"], 600)],
        "learned_loss": [last_pass(learned["loss"], 200), last_pass(learned["loss"], 600)],
        "late_monotone": all(b <= a for a, b in zip(learned["tau"][50:], learned["tau"][51:])),
    }


def near(a, b, tol=2e-3):
    return abs(a - b) <= tol


def verify(result):
    fc, fl, tau = result["fixed_cos"], result["fixed_loss"], result["tau"]
    lc, ll = result["learned_cos"], result["learned_loss"]
    return [
        practice.Check(
            "ANSWER: tau changes the printed loss, not the training",
            all([
                max(fc.values()) - min(fc.values()) < 1e-4,
                near(fc[1.0], 0.1680),
                near(fl[0.01], -19.16, 0.02),
                near(fl[1.0], 0.798),
                near(fl[100.0], 0.998),
                tau[0] == 1.0,
                near(tau[2], 0.949),
                near(tau[3], 0.833),
            ]),
            f"last-pass cos {', '.join(f'tau={t}: {c:.4f}' for t, c in fc.items())}; step-199 loss "
            f"{', '.join(f'{v:.3f}' for v in fl.values())}; learned tau {tau[0]} -> {tau[2]:.3f} "
            f"(200) -> {tau[3]:.3f} (600)",
        ),
        practice.Check(
            "FINDING: 1 - cos/tau has no minimum in tau",
            all([
                result["late_monotone"],
                tau[1] < 1.01,
                ll[1] < ll[0] - 0.01,
                abs(lc[1] - lc[0]) < 0.02,
                near(ll[0], 0.8242),
                near(ll[1], 0.7914),
                ll[1] < (1 - lc[1]) - 0.03,
            ]),
            f"tau never rises after step 50 (peak {tau[1]:.4f}); last-pass loss {ll[0]:.4f} -> "
            f"{ll[1]:.4f} while cos {lc[0]:.4f} -> {lc[1]:.4f} (tau = 1 would score {1 - lc[1]:.4f})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
