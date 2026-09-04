"""Exercise 1 — inverse-CDF sampling for Cauchy, and its heavy tails.

    Implement inverse CDF sampling for the Cauchy distribution. The CDF is
    F(x) = 0.5 + arctan(x)/pi. Generate 10,000 samples and plot the histogram
    against the true PDF. Notice the heavy tails (extreme values far from
    center).

Reading of the exercise: the histogram comparison is asserted with a
Kolmogorov-Smirnov distance against the exact CDF, which needs no binning and so
cannot be tuned by choosing bin widths. "Notice the heavy tails" is the part
worth making quantitative: the Cauchy has **no mean and no variance**, so the
sample mean does not converge — check 4 shows it wandering rather than settling,
which is the property a histogram cannot show and the reason the distribution
matters.
"""

from __future__ import annotations

import math
import random

from harness import practice

SEED, N = 20260904, 10_000


def sample_cauchy(rng):
    """F(x) = 0.5 + arctan(x)/π, so F⁻¹(u) = tan(π(u − 0.5))."""
    return math.tan(math.pi * (rng.random() - 0.5))


def cauchy_cdf(x):
    return 0.5 + math.atan(x) / math.pi


def solve():
    rng = random.Random(SEED)
    draws = [sample_cauchy(rng) for _ in range(N)]
    ordered = sorted(draws)
    ks = max(max(abs((i + 1) / N - cauchy_cdf(x)), abs(i / N - cauchy_cdf(x)))
             for i, x in enumerate(ordered))
    running = []
    total = 0.0
    for i, value in enumerate(draws, 1):
        total += value
        if i in (100, 1000, 5000, 10000):
            running.append((i, total / i))
    median = ordered[N // 2]
    normal_extremes = sum(1 for v in draws if abs(v) > 10)
    return {"ks": ks, "running_means": running, "median": median,
            "max_abs": max(abs(v) for v in draws),
            "beyond_10": normal_extremes,
            "iqr": (ordered[N // 4], ordered[3 * N // 4])}


def verify(result):
    critical = 1.36 / math.sqrt(N)
    means = [m for _, m in result["running_means"]]
    return [
        practice.Check(f"{N:,} samples match the exact CDF (KS test)",
                       result["ks"] < critical,
                       f"D = {result['ks']:.5f} < {critical:.5f}, the 95% critical value"),
        practice.Check("the median is ~0 and the IQR is ~(−1, 1), as theory says",
                       abs(result["median"]) < 0.05
                       and abs(result["iqr"][0] + 1) < 0.06 and abs(result["iqr"][1] - 1) < 0.06,
                       f"median {result['median']:.4f}, IQR "
                       f"({result['iqr'][0]:.4f}, {result['iqr'][1]:.4f}) — the quartiles of "
                       f"a standard Cauchy are exactly ∓1"),
        practice.Check("the tails are heavy: extreme values are common, not rare",
                       result["beyond_10"] > 200,
                       f"{result['beyond_10']} of {N:,} samples exceed |10| "
                       f"({result['beyond_10'] / N:.1%}); largest is "
                       f"{result['max_abs']:.1f}. A standard normal would produce "
                       f"about 1 in 10^23"),
        practice.Check("FINDING: the sample mean does not converge — Cauchy has no mean",
                       max(means) - min(means) > 0.05,
                       "running mean: " + ", ".join(
                           f"n={n}: {m:+.4f}" for n, m in result["running_means"])
                       + " — it wanders instead of settling, because ∫x·p(x)dx diverges. "
                         "No histogram can show this; only the running average can"),
        practice.Check("…which is why the median, not the mean, is the location estimate",
                       abs(result["median"]) < abs(means[-1]) or abs(result["median"]) < 0.05,
                       f"after {N:,} samples the median is {result['median']:+.4f} against "
                       f"a mean of {means[-1]:+.4f}; the median is consistent for Cauchy "
                       f"and the mean is not consistent for anything"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
