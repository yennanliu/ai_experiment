"""Exercise 2 — rejection sampling Beta(2,5) from Uniform(0,1); acceptance rate.

    Use rejection sampling to generate samples from a Beta(2, 5) distribution
    using a Uniform(0, 1) proposal. Plot the accepted samples against the true
    Beta PDF. What is the theoretical acceptance rate?

Reading of the exercise: the closing question has an exact answer worth deriving
rather than measuring. With a Uniform(0,1) proposal the envelope constant M must
be the PDF's maximum, which for Beta(2,5) is at the mode x = 1/5 and equals
2.4576. Acceptance rate is 1/M = **40.69%**, because both the target and the
proposal integrate to 1 over the same support. Check 4 compares the measured rate
to that derivation; check 5 shows what happens to the rate when the proposal is
a worse fit, which is the reason rejection sampling scales badly.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "16-sampling-methods"
SEED, N = 20260904, 20_000
ALPHA, BETA = 2, 5
MODE = (ALPHA - 1) / (ALPHA + BETA - 2)


def beta_pdf(x, a=ALPHA, b=BETA):
    if not 0 < x < 1:
        return 0.0
    norm = math.gamma(a + b) / (math.gamma(a) * math.gamma(b))
    return norm * x ** (a - 1) * (1 - x) ** (b - 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "sampling")
    rng = random.Random(SEED)
    envelope = beta_pdf(MODE)
    accepted, attempts = [], 0
    while len(accepted) < N:
        x = rng.random()
        attempts += 1
        if rng.random() * envelope <= beta_pdf(x):
            accepted.append(x)
    ordered = sorted(accepted)
    # exact CDF of Beta(2,5): 1 − (1−x)^5(1+5x)  [integrate 30x(1−x)^4]
    cdf = lambda x: 1 - (1 - x) ** 5 * (1 + 5 * x)
    ks = max(max(abs((i + 1) / N - cdf(x)), abs(i / N - cdf(x)))
             for i, x in enumerate(ordered))
    # a deliberately poor envelope: 4x too tall
    poor_accepted, poor_attempts = 0, 0
    for _ in range(20_000):
        x = rng.random()
        poor_attempts += 1
        if rng.random() * envelope * 4 <= beta_pdf(x):
            poor_accepted += 1
    return {"envelope": envelope, "rate": N / attempts, "theoretical": 1 / envelope,
            "ks": ks, "mean": sum(accepted) / N, "attempts": attempts,
            "poor_rate": poor_accepted / poor_attempts,
            "lesson_batch": len(ref.rejection_sample_batch(
                beta_pdf, rng.random, lambda x: 1.0, envelope, 500))}


def verify(result):
    critical = 1.36 / math.sqrt(N)
    return [
        practice.Check("the envelope constant is the PDF maximum, 2.4576 at x=0.2",
                       abs(result["envelope"] - 2.4576) < 1e-3,
                       f"max Beta(2,5) PDF = {result['envelope']:.6f} at the mode "
                       f"x = {MODE:.1f}; a smaller M would not envelope the target"),
        practice.Check(f"{N:,} accepted samples match the exact Beta(2,5) CDF",
                       result["ks"] < critical,
                       f"KS D = {result['ks']:.5f} < {critical:.5f}"),
        practice.Check("the sample mean matches α/(α+β) = 2/7",
                       abs(result["mean"] - ALPHA / (ALPHA + BETA)) < 0.01,
                       f"{result['mean']:.5f} vs {ALPHA / (ALPHA + BETA):.5f}"),
        practice.Check("ANSWER: the theoretical acceptance rate is 1/M = 40.69%",
                       abs(result["rate"] - result["theoretical"]) < 0.01
                       and abs(result["theoretical"] - 0.4069) < 1e-3,
                       f"measured {result['rate']:.2%} over {result['attempts']:,} "
                       f"proposals, against 1/{result['envelope']:.4f} = "
                       f"{result['theoretical']:.2%}. Both densities integrate to 1 over "
                       f"[0,1], so the accepted fraction is exactly the area ratio"),
        practice.Check("…and a 4x-too-tall envelope quarters it, which is why this scales badly",
                       abs(result["poor_rate"] - result["theoretical"] / 4) < 0.02,
                       f"envelope × 4 gives {result['poor_rate']:.2%}, a quarter of "
                       f"{result['theoretical']:.2%}. In d dimensions the envelope gap "
                       f"compounds per axis, so acceptance decays exponentially and "
                       f"rejection sampling stops working"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
