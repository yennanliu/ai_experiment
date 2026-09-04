"""Exercise 1 — 1000 walks of 10,000 steps: is the endpoint Gaussian?

    **Simulate 1000 random walks of 10000 steps.** Plot the distribution of final
    positions. Verify it is approximately Gaussian with mean 0 and standard
    deviation sqrt(10000) = 100.

Reading of the exercise: mean and standard deviation are the easy part, and they
would also match a distribution that is not Gaussian at all. So normality is
tested on its own terms — skewness and excess kurtosis near 0, plus a
Kolmogorov-Smirnov distance against the exact normal CDF (check 3). One detail
the exercise omits: a ±1 walk of 10,000 steps only reaches **even** endpoints, so
the distribution is supported on half the integers and no continuous test can be
exact — check 5 measures that gap.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "22-stochastic-processes"
N_WALKS, N_STEPS, SEED = 1000, 10_000, 20260904
EXPECTED_SD = math.sqrt(N_STEPS)


def normal_cdf(x, sd):
    return 0.5 * (1 + math.erf(x / (sd * math.sqrt(2))))


def _moments(finals):
    mean = sum(finals) / len(finals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in finals) / len(finals))
    centred = [(v - mean) / sd for v in finals]
    return {"mean": mean, "sd": sd,
            "skew": sum(v ** 3 for v in centred) / len(finals),
            "kurtosis": sum(v ** 4 for v in centred) / len(finals) - 3}


def _ks(finals):
    """Kolmogorov-Smirnov distance to the exact N(0, √N_STEPS) CDF."""
    ordered = sorted(finals)
    return max(max(abs((i + 1) / N_WALKS - normal_cdf(v, EXPECTED_SD)),
                   abs(i / N_WALKS - normal_cdf(v, EXPECTED_SD)))
               for i, v in enumerate(ordered))


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "stochastic")
    finals = [float(ref.random_walk_1d(N_STEPS, seed=SEED + i)[-1])
              for i in range(N_WALKS)]
    moments = _moments(finals)
    mean, sd = moments["mean"], moments["sd"]
    return {**moments, "ks": _ks(finals),
            "even": sum(1 for v in finals if int(v) % 2 == 0),
            "range": (min(finals), max(finals)),
            "within_1sd": sum(1 for v in finals if abs(v - mean) < sd) / N_WALKS}


def verify(result):
    critical = 1.36 / math.sqrt(N_WALKS)
    return [
        practice.Check(f"mean ≈ 0 over {N_WALKS:,} walks",
                       abs(result["mean"]) < 3 * EXPECTED_SD / math.sqrt(N_WALKS),
                       f"mean {result['mean']:+.3f}; the standard error of the mean is "
                       f"{EXPECTED_SD / math.sqrt(N_WALKS):.2f}, so this is well inside "
                       f"noise"),
        practice.Check(f"standard deviation ≈ √{N_STEPS} = {EXPECTED_SD:.0f}",
                       abs(result["sd"] - EXPECTED_SD) < 0.1 * EXPECTED_SD,
                       f"measured {result['sd']:.2f}, range "
                       f"({result['range'][0]:.0f}, {result['range'][1]:.0f})"),
        practice.Check("…and the shape is Gaussian, not merely the moments",
                       abs(result["skew"]) < 0.15 and abs(result["kurtosis"]) < 0.3
                       and result["ks"] < critical,
                       f"skew {result['skew']:+.4f}, excess kurtosis "
                       f"{result['kurtosis']:+.4f}, KS distance {result['ks']:.4f} < "
                       f"{critical:.4f}. Mean and sd alone would also fit a uniform "
                       f"distribution of the right width"),
        practice.Check("68% of walks land within one standard deviation",
                       0.64 < result["within_1sd"] < 0.72,
                       f"{result['within_1sd']:.1%} against the normal's 68.3%"),
        practice.Check("FINDING: every endpoint is even, so the support is half the integers",
                       result["even"] == N_WALKS,
                       f"all {result['even']} of {N_WALKS} final positions are even — a ±1 "
                       f"walk of {N_STEPS:,} steps has endpoint {N_STEPS} − 2·(down steps), "
                       f"which is always even. The limit is Gaussian but every finite walk "
                       f"is lattice-supported, so no continuous test can ever be exact"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
