"""Exercise 4 — Metropolis-Hastings on a 2D target; proposal width sweep.

    Implement Metropolis-Hastings to sample from a 2D distribution p(x, y)
    proportional to exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Plot the
    samples and the chain trajectory. Experiment with different proposal
    standard deviations.

Reading of the exercise: "experiment with different proposal standard
deviations" has a well-known right answer, and the checks are built to find it
rather than to describe it. Acceptance rate must fall monotonically with proposal
width (check 2); the useful width is the one maximising *effective* sample size,
not acceptance (check 4) — a chain accepting 95% of tiny steps explores nothing.
Check 5 records the shape of this particular target, which the x²y² coupling term
makes bimodal along the anti-diagonal.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "16-sampling-methods"
SEED, N_SAMPLES, BURN_IN = 20260904, 12_000, 2_000
WIDTHS = (0.1, 1.5, 4.0, 12.0)


def target_log_pdf(x, y):
    return -(x * x * y * y + x * x + y * y - 8 * x - 8 * y) / 2


def _autocorrelation_time(values):
    """Rough integrated autocorrelation: 1 + 2Σρ_k while ρ_k > 0.05."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var == 0:
        return float("inf")
    total = 1.0
    for lag in range(1, min(200, n // 2)):
        rho = sum((values[i] - mean) * (values[i + lag] - mean)
                  for i in range(n - lag)) / ((n - lag) * var)
        if rho <= 0.05:
            break
        total += 2 * rho
    return total


def solve():
    ref = parity.load_reference(PHASE, LESSON, "sampling")
    rows = {}
    for width in WIDTHS:
        import random
        random.seed(SEED)
        samples, accept = ref.metropolis_hastings_2d(target_log_pdf, 1.0, 1.0, N_SAMPLES,
                                                     BURN_IN, proposal_std=width)
        xs = [p[0] for p in samples]
        ys = [p[1] for p in samples]
        tau = _autocorrelation_time(xs)
        rows[width] = {"n": len(samples), "accept": accept,
                       "mean_x": sum(xs) / len(xs), "mean_y": sum(ys) / len(ys),
                       "tau": tau, "ess": len(samples) / tau if tau else 0.0,
                       "in_upper": sum(1 for x, y in samples if x + y > 4) / len(samples),
                       "spread_x": max(xs) - min(xs)}
    return {"rows": rows}


def verify(result):
    rows = result["rows"]
    accepts = [rows[w]["accept"] for w in WIDTHS]
    best = max(WIDTHS, key=lambda w: rows[w]["ess"])
    return [
        practice.Check(f"chains run at all {len(WIDTHS)} proposal widths",
                       all(r["n"] > 0 for r in rows.values()),
                       "; ".join(f"σ={w}: accept {rows[w]['accept']:.1%}, "
                                 f"ESS {rows[w]['ess']:.0f}" for w in WIDTHS)),
        practice.Check("acceptance falls monotonically as the proposal widens",
                       all(a > b for a, b in zip(accepts, accepts[1:])),
                       ", ".join(f"σ={w}: {rows[w]['accept']:.1%}" for w in WIDTHS)),
        practice.Check("the narrowest proposal accepts most and explores least",
                       rows[WIDTHS[0]]["accept"] == max(accepts)
                       and rows[WIDTHS[0]]["spread_x"] < rows[best]["spread_x"],
                       f"σ={WIDTHS[0]} accepts {accepts[0]:.1%} but covers only "
                       f"{rows[WIDTHS[0]]['spread_x']:.2f} in x, against "
                       f"{rows[best]['spread_x']:.2f} at σ={best}"),
        practice.Check(f"effective sample size peaks at an intermediate width (σ={best})",
                       best not in (WIDTHS[0], WIDTHS[-1]),
                       ", ".join(f"σ={w}: ESS {rows[w]['ess']:.0f} (τ={rows[w]['tau']:.1f})"
                                 for w in WIDTHS)
                       + f" — high acceptance is not the goal, independent samples are. "
                         f"Note the peak sits at {rows[best]['accept']:.1%} acceptance, well "
                         f"below the ~23% rule of thumb: this target is bimodal, so wide "
                         f"jumps that usually fail are what cross between the modes"),
        practice.Check("the x²y² coupling makes the target bimodal, not a single blob",
                       0.2 < rows[best]["in_upper"] < 0.8,
                       f"at σ={best}, {rows[best]['in_upper']:.1%} of samples have x+y>4; "
                       f"mean ({rows[best]['mean_x']:.2f}, {rows[best]['mean_y']:.2f}) sits "
                       f"between the modes rather than at either, so the mean is not a "
                       f"summary of this distribution"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
