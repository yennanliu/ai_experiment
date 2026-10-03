"""Exercise 1 — 64 Gaussians overflow on step zero, and the cause is a stale base loss.

    **Easy.** Run `code/main.py` with 4, 16, 64 Gaussians. Report final MSE vs target.

Reading of the exercise: `main()` loops over `n in [2, 4, 8]`; the exercise
swaps that list for `[4, 16, 64]` and keeps everything else -- the seed
`7 + n`, 30 calls of `finite_diff_step(gaussians, target, lr=0.5, eps=0.2)`.
The lesson's own `init_gaussians`, `finite_diff_step`, `render` and `mse` do
all the work; "final MSE" is measured on the parameters after the last step.

**ANSWER: 4 -> 0.0575, 16 -> 0.0280, 64 -> no number at all.** With 64
Gaussians `finite_diff_step` raises `OverflowError` inside `gaussian_value` on
the very first step: `sigma ** 2` overflows after sigma reaches ~1e238.

**FINDING: the crash is a bug in the optimiser, not a learning rate that is
too big.** `finite_diff_step` updates each coordinate in place as it goes, but
divides every probe by the same `base_loss` measured *before* any update. Each
later "gradient" therefore also contains every earlier update's loss change.
On step 0 at n=64 the probe loss passes 100x the base by probe 29 of 257, and
every coordinate after that gets a huge gradient. A corrected step -- every
difference taken against the same parameters, then all updates applied
together, same lr, eps, init -- takes 64 Gaussians from 15.36 to 0.0601
in 5 steps without overflow, and at n=16 it ends at 0.0099 against the lesson's 0.0280.

**FINDING: the target is exactly two Gaussians**, so every MSE above is
optimiser residual, not missing capacity. Through the lesson's own `render`,
sigma sqrt(3) colour 1 at (3, 3) plus sigma 2 colour 0.5 at (8, 8) leaves
1e-33. And more Gaussians is not monotone: n=24 ends at 0.117, 4x worse than 16.

**CONTROL: 64 is not unlucky seeding.** Seeds 0-4 at n=64 all overflow within
three steps.

Structure: `fit` runs the lesson's step; `probe_losses` records every loss the
step evaluates; `fixed_step` is the corrected update.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "12-3d-generation"
STEPS, LR, EPS, COUNTS = 30, 0.5, 0.2, (4, 16, 24, 64)
EXACT = [{"pos": [3.0, 3.0], "sigma": math.sqrt(3), "color": 1.0},
         {"pos": [8.0, 8.0], "sigma": 2.0, "color": 0.5}]


def fit(ref, target, n, seed, steps=STEPS, step=None):
    """Final MSE after `steps` updates, or the step index that overflowed."""
    gaussians = ref.init_gaussians(n, random.Random(seed))
    step = step or ref.finite_diff_step
    for index in range(steps):
        try:
            step(gaussians, target, LR, EPS)
        except OverflowError:
            return f"overflow at step {index}"
    return ref.mse(ref.render(gaussians), target)


def probe_losses(ref, target, n):
    """Every loss finite_diff_step evaluates on step 0, in order."""
    seen, original = [], ref.mse
    ref.mse = lambda a, b: seen.append(original(a, b)) or seen[-1]
    try:
        ref.finite_diff_step(ref.init_gaussians(n, random.Random(7 + n)), target, LR, EPS)
    except OverflowError:
        pass
    finally:
        ref.mse = original
    return seen


def coordinates(gaussians):
    """(container, key) for every scalar parameter, in finite_diff_step's order."""
    for g in gaussians:
        yield from ((g["pos"], i) for i in range(len(g["pos"])))
        yield from ((g, "sigma"), (g, "color"))


def fixed_step(ref):
    """finite_diff_step with every difference taken before any update."""

    def step(gaussians, target, lr, eps):
        base, grads = ref.mse(ref.render(gaussians), target), []
        for box, key in coordinates(gaussians):
            box[key] += eps
            grads.append((ref.mse(ref.render(gaussians), target) - base) / eps)
            box[key] -= eps
        for (box, key), grad in zip(list(coordinates(gaussians)), grads):
            box[key] -= lr * grad

    return step


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    target = ref.make_target(ref.SIZE)
    probes = probe_losses(ref, target, 64)
    return {
        "lesson": {n: fit(ref, target, n, 7 + n) for n in COUNTS},
        "first_blowup": next(i for i, v in enumerate(probes) if v > 100 * probes[0]),
        "probes": 4 * 64 + 1,
        "fixed64": fit(ref, target, 64, 71, steps=5, step=fixed_step(ref)),
        "fixed16": fit(ref, target, 16, 23, step=fixed_step(ref)),
        "exact": ref.mse(ref.render(EXACT), target),
        "seeds": [fit(ref, target, 64, seed, steps=3) for seed in range(5)],
    }


def verify(result):
    lesson = result["lesson"]
    return [
        practice.Check(
            "ANSWER: 4 and 16 report an MSE, 64 overflows on step zero",
            lesson[4] > lesson[16] and lesson[64] == "overflow at step 0",
            f"final MSE at the lesson's own seed 7+n and 30 steps -- n=4: {lesson[4]:.4f}, "
            f"n=16: {lesson[16]:.4f}, n=64: {lesson[64]} (OverflowError in gaussian_value)",
        ),
        practice.Check(
            "FINDING: the overflow is finite_diff_step's stale base_loss",
            result["first_blowup"] < 64 and isinstance(result["fixed64"], float)
            and result["fixed16"] < lesson[16] / 2,
            f"on step 0 at n=64 the probe loss passes 100x base_loss at probe "
            f"{result['first_blowup']} of {result['probes']}, because every probe is divided "
            f"against a loss measured before the earlier in-place updates. The same step with all "
            f"differences taken first (same lr, eps, init) gives n=64: {result['fixed64']:.4f} "
            f"after 5 steps and n=16: {result['fixed16']:.4f} vs the lesson's {lesson[16]:.4f}",
        ),
        practice.Check(
            "FINDING: the target is exactly two Gaussians, and more is not monotone",
            result["exact"] < 1e-25 and lesson[24] > 2 * lesson[16],
            f"two splats (sigma sqrt(3) colour 1 at (3,3); sigma 2 colour 0.5 at (8,8)) render "
            f"to MSE {result['exact']:.1e} through the lesson's render, so every number above is "
            f"optimiser residual; n=24 ends at {lesson[24]:.4f}, worse than n=16",
        ),
        practice.Check(
            "CONTROL: n=64 overflows for every seed tried, not just 71",
            all(str(s).startswith("overflow") for s in result["seeds"]),
            "seeds 0-4 at n=64, three steps each: " + "; ".join(map(str, result["seeds"])),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
