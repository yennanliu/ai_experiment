"""Exercise 5 — 2.5x if the pre-period is part of post, and the variance ratio alone does not fix it.

    Apply CUPED to a pre-period with 60% of the variance of post. Compute the
    effective-sample-size boost.

Reading of the exercise: CUPED replaces Y with Y - theta (X - mean X),
theta = cov(X, Y) / var(X), which leaves var(Y)(1 - rho^2); the boost in
effective sample size is 1 / (1 - rho^2). "60% of the variance" is read as
var(X) = 0.6 var(Y) with post = pre + fresh noise -- the natural model of a
user's pre- and post-period metric -- and then varied, on 20,000 seeded
Gaussian users per case.

**ANSWER: 2.5x.** With post = pre + independent noise, cov(X, Y) = var(X),
so rho^2 = var(X) / var(Y) = 0.6: CUPED removes 60% of the variance, leaving
0.4, and 1 / 0.4 = 2.5. Measured: variance kept 0.401, boost 2.49x. Exercise
1's 207,702 users per arm becomes 83,081; with the x1.4 buffer, 290,782
becomes 116,313 -- CUPED more than pays for the lesson's noise buffer.

**FINDING: the 60% fixes the boost only through that model -- the boost is
set by the correlation, not the variance ratio.** Keep var(X) = 0.6 var(Y)
and make rho = 0.5: the boost is 1.33x; make X independent and it is 1.00x;
multiply X by 10 and it is the same 2.49x, to the last digit. CUPED is invariant to the scale of
the covariate, so "60% of the variance" is only an answer because post
contains pre. The lesson's "30-70%" reduction band is a 1.43x-3.33x boost.

**FINDING: the reference code has no CUPED.** Its docstring says it
"Illustrates CUPED-style variance reduction"; that sentence is the only
occurrence of "CUPED" in the module, and no function takes a pre-period.

Structure: `cuped()` is the estimator; `case()` draws one population with a
chosen var(X) and correlation and measures the variance CUPED keeps.
"""

from __future__ import annotations

import inspect
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "21-ab-testing-llm-features"
N, VAR_POST, RATIO = 20_000, 1.0, 0.6


def cuped(x, y):
    theta = statistics.covariance(x, y) / statistics.variance(x)
    mx = statistics.fmean(x)
    return [yi - theta * (xi - mx) for xi, yi in zip(x, y)]


def case(rho, scale=1.0, seed=0):
    """Variance kept by CUPED for var(X) = RATIO * var(Y) and corr(X, Y) = rho."""
    rng = random.Random(seed)
    b = rho * math.sqrt(VAR_POST / RATIO)
    noise = math.sqrt(VAR_POST - b * b * RATIO)
    x = [rng.gauss(0, math.sqrt(RATIO)) for _ in range(N)]
    y = [b * xi + rng.gauss(0, noise) for xi in x]
    return statistics.variance(
        cuped([scale * xi for xi in x], y)
    ) / statistics.variance(y)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    n = ref.fixed_sample_size(0.03, 0.05)
    kept = case(math.sqrt(RATIO))  # post = pre + noise: rho^2 = var ratio
    return {
        "kept": kept,
        "boost": 1 / kept,
        "theory": 1 / (1 - RATIO),
        "n": (
            n,
            round(n * (1 - RATIO)),
            int(n * 1.4),
            round(int(n * 1.4) * (1 - RATIO)),
        ),
        "rho_half": 1 / case(0.5),
        "independent": 1 / case(0.0),
        "scaled": 1 / case(math.sqrt(RATIO), scale=10.0),
        "band": (1 / (1 - 0.3), 1 / (1 - 0.7)),
        "mentions": inspect.getsource(ref).count("CUPED"),
    }


def verify(result):
    n = result["n"]
    return [
        practice.Check(
            "ANSWER: 2.5x",
            round(result["kept"], 2) == 0.40
            and abs(result["boost"] - 2.5) < 0.02
            and result["theory"] == 2.5
            and n == (207702, 83081, 290782, 116313),
            f"CUPED keeps {result['kept']:.3f} of the variance, boost {result['boost']:.2f}x "
            f"(1 / (1 - 0.6) = {result['theory']}); {n[0]} -> {n[1]} per arm, "
            f"buffered {n[2]} -> {n[3]}",
        ),
        practice.Check(
            "FINDING: the boost is set by the correlation, not the variance ratio",
            round(result["rho_half"], 2) == 1.33
            and round(result["independent"], 2) == 1.00
            and abs(result["scaled"] - result["boost"]) < 1e-9,
            f"same 60% variance ratio: rho 0.5 -> {result['rho_half']:.2f}x, independent -> "
            f"{result['independent']:.2f}x, X scaled 10x -> {result['scaled']:.2f}x; the "
            f"lesson's 30-70% band is {result['band'][0]:.2f}x-{result['band'][1]:.2f}x",
        ),
        practice.Check(
            "FINDING: the reference code has no CUPED",
            result["mentions"] == 1,
            f"'CUPED' appears {result['mentions']} time in the module, in its docstring",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
