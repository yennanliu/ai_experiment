"""Exercise 5 — what fraction of float16 gradients vanish, with and without scaling.

    **Loss scaling experiment.** Simulate training with float16: create random
    gradients in the range [1e-9, 1e-3], convert to float16, and measure what
    fraction become zero. Then apply loss scaling (multiply by 1024), convert to
    float16, scale back, and measure the zero fraction again.

Reading of the exercise: "in the range [1e-9, 1e-3]" is ambiguous and the choice
decides the whole result. Uniform sampling puts almost every draw near 1e-3,
where nothing underflows and the experiment shows nothing. Gradients are
log-distributed in practice, so the magnitudes are sampled **log-uniformly**,
which is stated rather than hidden. Both boundaries are then probed
directly rather than inferred from the sample, because at 1024x nothing in
[1e-9, 1e-3] underflows any more — the interesting numbers are outside the range
the exercise specifies.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "13-numerical-stability"
N, SEED, SCALE = 20_000, 20260904, 1024
LOW, HIGH = 1e-9, 1e-3
FLOAT16_SUBNORMAL = 2.0 ** -24              # 5.96e-08
FLOAT16_MIN_NORMAL = 2.0 ** -14             # 6.10e-05


def log_uniform(rng, n):
    low, high = math.log10(LOW), math.log10(HIGH)
    return [10 ** rng.uniform(low, high) for _ in range(n)]


def zero_fraction(ref, values, scale=1):
    lost = sum(1 for v in values if ref.simulate_float16(v * scale) == 0.0)
    return lost / len(values)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "numerical")
    rng = random.Random(SEED)
    gradients = log_uniform(rng, N)
    plain = zero_fraction(ref, gradients)
    scaled = zero_fraction(ref, gradients, SCALE)
    uniform = [rng.uniform(LOW, HIGH) for _ in range(N)]
    # what the same experiment shows under the other reading of "range"
    uniform_plain = zero_fraction(ref, uniform)
    # probe the underflow boundary directly, rather than inferring it from the
    # sample: nothing in [1e-9, 1e-3] falls below it once scaled
    half = FLOAT16_SUBNORMAL / 2
    # one gradient chosen to sit just above the *scaled* threshold: it must
    # vanish unscaled and survive scaled. That is the whole mechanism.
    rescued = half / SCALE * 1.01
    boundary = {"just below": ref.simulate_float16(half * 0.99),
                "just above": ref.simulate_float16(half * 1.01),
                "rescued, unscaled": ref.simulate_float16(rescued),
                "rescued, scaled": ref.simulate_float16(rescued * SCALE),
                "below even scaled": ref.simulate_float16(half / SCALE * 0.99 * SCALE)}
    # and the overflow boundary: the scale at which the largest gradient blows up
    overflow_scale = next(2 ** k for k in range(1, 40)
                          if not math.isfinite(ref.simulate_float16(HIGH * 2 ** k)))
    return {"plain": plain, "scaled": scaled, "uniform_plain": uniform_plain, "n": N,
            "boundary": boundary, "rescued": rescued, "threshold_plain": half,
            "threshold_scaled": half / SCALE, "overflow_scale": overflow_scale,
            "predicted_overflow": 65504 / HIGH}


def verify(result):
    return [
        practice.Check(f"{result['n']:,} log-uniform gradients over [1e-9, 1e-3]",
                       0.1 < result["plain"] < 0.9,
                       f"{result['plain']:.1%} vanish in plain float16"),
        practice.Check(f"loss scaling by {SCALE} rescues most of them",
                       result["scaled"] < result["plain"] / 5,
                       f"{result['plain']:.1%} -> {result['scaled']:.1%} zeros, a "
                       f"{result['plain'] / max(result['scaled'], 1e-9):.0f}x reduction"),
        practice.Check("the vanishing threshold moves by exactly the scale factor",
                       result["boundary"]["just below"] == 0.0
                       and result["boundary"]["just above"] != 0.0
                       and result["boundary"]["rescued, unscaled"] == 0.0
                       and result["boundary"]["rescued, scaled"] != 0.0
                       and result["boundary"]["below even scaled"] == 0.0,
                       f"unscaled the boundary is 2⁻²⁵ = {result['threshold_plain']:.3e} "
                       f"(just below -> 0.0, just above -> "
                       f"{result['boundary']['just above']:.3e}). A gradient of "
                       f"{result['rescued']:.3e} vanishes unscaled and survives at "
                       f"{SCALE}x, while one 1% smaller still vanishes — so the threshold "
                       f"moves to exactly {result['threshold_scaled']:.3e}, buying "
                       f"{math.log2(SCALE):.0f} binary orders of magnitude and no more"),
        practice.Check("uniform sampling would have shown nothing",
                       result["uniform_plain"] < 0.01,
                       f"the same experiment with uniform draws loses "
                       f"{result['uniform_plain']:.2%} — almost every uniform draw from "
                       f"[1e-9, 1e-3] lands within a factor of 10 of the upper bound, "
                       f"far above float16's floor. The distribution is the experiment"),
        practice.Check("…and scaling has a ceiling on the other side",
                       result["overflow_scale"] / result["predicted_overflow"] < 2,
                       f"the largest gradient ({HIGH:g}) overflows to inf at scale "
                       f"{result['overflow_scale']:.3g}, against the predicted "
                       f"65504/{HIGH:g} = {result['predicted_overflow']:.3g}. Loss scaling "
                       f"is a window, not a fix: under 2⁻²⁵/scale you underflow, over "
                       f"65504/scale you overflow, and float16 gives you about "
                       f"{math.log2(65504 / (FLOAT16_SUBNORMAL / 2)):.0f} binary decades "
                       f"to fit the whole gradient distribution into"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
