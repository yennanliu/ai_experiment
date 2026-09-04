"""Exercise 1 — inverse transform sampling for the exponential distribution.

    Implement inverse transform sampling for the exponential distribution.
    Verify by sampling 10,000 values and comparing the histogram to the true
    PDF.

Reading of the exercise: "comparing the histogram to the true PDF" is the whole
task and it needs a statistic, or the comparison is an eyeball. Two are used: a
per-bin **z-score** against the analytic bin mass — a fixed percentage would be
wrong, since bin noise scales as 1/√expected — and a Kolmogorov-Smirnov distance
against the exact CDF, which needs no binning at all and so cannot be tuned by
choosing bin widths. The sample is seeded — 10,000 draws that differ per run cannot be
asserted on.
"""

from __future__ import annotations

import math
import random

from harness import practice

SEED, N, LAMBDA, BINS = 20260904, 10_000, 1.5, 20
UPPER = 4.0


def sample_exponential(rate, n, rng):
    """F(x) = 1 − e^(−λx), so F⁻¹(u) = −ln(1 − u)/λ."""
    return [-math.log(1.0 - rng.random()) / rate for _ in range(n)]


def cdf(x, rate=LAMBDA):
    return 1.0 - math.exp(-rate * x)


def solve():
    rng = random.Random(SEED)
    draws = sample_exponential(LAMBDA, N, rng)
    width = UPPER / BINS
    counts = [0] * BINS
    for value in draws:
        if value < UPPER:
            counts[int(value / width)] += 1
    rows = []
    for i, count in enumerate(counts):
        expected = N * (cdf((i + 1) * width) - cdf(i * width))
        # a bin's count is Binomial(N, p): sd ≈ sqrt(expected), so the honest
        # yardstick is a z-score, not a fixed percentage
        rows.append({"lo": i * width, "observed": count, "expected": expected,
                     "z": abs(count - expected) / math.sqrt(expected)})
    ordered = sorted(draws)
    ks = max(max(abs((i + 1) / N - cdf(x)), abs(i / N - cdf(x)))
             for i, x in enumerate(ordered))
    return {"rows": rows, "ks": ks, "mean": sum(draws) / N,
            "var": sum((x - sum(draws) / N) ** 2 for x in draws) / N,
            "n_beyond": sum(1 for x in draws if x >= UPPER)}


def verify(result):
    well_populated = [r for r in result["rows"] if r["expected"] >= 30]
    worst = max(r["z"] for r in well_populated)
    critical = 1.36 / math.sqrt(N)          # KS 95% critical value
    return [
        practice.Check(f"{N:,} values sampled by inverse transform",
                       result["n_beyond"] > 0,
                       f"{result['n_beyond']} fell beyond the {UPPER} histogram range, "
                       f"as an unbounded support requires"),
        practice.Check("every well-populated bin is within 3σ of the analytic PDF mass",
                       worst < 3.0,
                       f"worst z-score {worst:.2f} over {len(well_populated)} bins with "
                       f"expected count ≥ 30 — a fixed percentage would be the wrong "
                       f"yardstick here, since bin noise scales as 1/√expected"),
        practice.Check(f"KS distance to the exact CDF is below the 95% critical value",
                       result["ks"] < critical,
                       f"D = {result['ks']:.5f} < {critical:.5f} — a binning-free test, "
                       f"so it cannot be gamed by choosing bin widths"),
        practice.Check("sample mean matches 1/λ",
                       abs(result["mean"] - 1 / LAMBDA) < 0.02,
                       f"mean {result['mean']:.5f} vs 1/λ = {1 / LAMBDA:.5f}"),
        practice.Check("sample variance matches 1/λ² — the shape, not just the centre",
                       abs(result["var"] - 1 / LAMBDA ** 2) < 0.03,
                       f"var {result['var']:.5f} vs 1/λ² = {1 / LAMBDA ** 2:.5f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
