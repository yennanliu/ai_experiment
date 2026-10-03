"""Exercise 1 — a quadratic needs two differences, but the lesson's check passes it after one.

    **Stationarity experiment.** Generate a series with a linear trend. Check
    stationarity with rolling statistics. Apply first differencing. Check again.
    How many rounds of differencing does it take for a quadratic trend?

Reading of the exercise: "check stationarity with rolling statistics" is the
lesson's own `check_stationarity`, whose verdict is the half-split test the doc
describes (half means differ by less than half a std, variance ratio under 2).
Every series is 300 points, noise N(0, 2^2), differenced by the lesson's own
`difference`, and the round count is the first round the verdict turns True.

**ANSWER: one round for a linear trend, two for a quadratic.** Slope 0.1 fails
at d=0 and passes at d=1; `0.01 t^2` fails at d=0 and d=1 and passes at d=2.
Without noise the second difference of `b t^2` is the constant `2b` exactly.

**FINDING: a gentle quadratic passes after one round on 100 of 100 seeds.** For
`0.001 t^2` the first difference still carries a slope of 0.002, but differencing
doubles the noise variance, and the half-split test only fires when the
remaining drift `L = 2bn` exceeds `sqrt(12/11) * sqrt(2) sigma`, i.e. `b > 0.0049`
at n=300, sigma=2. Measured pass rates after one round: 100/100 at b=0.001,
40/100 at b=0.005, 0/100 at b=0.01. "How many rounds" depends on the noise,
not only on the polynomial degree.

**FINDING: the check never says "too many".** White noise passes at d=0, 1, 2
and 3 alike, while each round multiplies its variance (1.96x, then 5.83x the
original) and plants a lag-1 autocorrelation of -0.49, then -0.66. The doc's
"differencing never hurts" is the opposite of what the second round does to
an already stationary series.

**CONTROL:** a random walk, the textbook non-stationary series, passes the
check on 13 of 100 seeds; white noise passes on 100 of 100.

Structure: `rounds_needed` differences until the lesson's verdict is True;
`pass_rate` counts verdicts over seeds.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "15-time-series"
N, SIGMA, SEEDS = 300, 2.0, 100
T = np.arange(N, dtype=float)


def rounds_needed(ref, series, max_rounds=4):
    """First differencing order at which check_stationarity says True."""
    for order in range(max_rounds + 1):
        if ref.check_stationarity(ref.difference(series, order) if order else series)[2]:
            return order
    return None


def noisy(seed, trend):
    return trend + np.random.RandomState(seed).normal(0, SIGMA, N)


def pass_rate(ref, make, order):
    """How many of SEEDS series pass the lesson's check after `order` differences."""
    hits = 0
    for seed in range(SEEDS):
        series = make(seed)
        hits += bool(ref.check_stationarity(ref.difference(series, order) if order else series)[2])
    return hits


def overdiff(ref):
    """Variance ratio and lag-1 ACF of white noise after 0..2 differences."""
    base = np.random.RandomState(7).normal(0, 1, 5000)
    rows = []
    for order in range(3):
        x = ref.difference(base, order) if order else base
        rows.append((x.var() / base.var(), ref.autocorrelation(x, 1)[1]))
    passes = [bool(ref.check_stationarity(ref.difference(base[:N], d) if d else base[:N])[2])
              for d in range(4)]
    return rows, passes


def solve():
    ref = parity.load_reference(PHASE, LESSON, "time_series")
    gentle = {b: pass_rate(ref, lambda s, b=b: noisy(s, b * T**2), 1) for b in (0.001, 0.005, 0.01)}
    rows, passes = overdiff(ref)
    return {
        "linear": rounds_needed(ref, noisy(0, 0.1 * T)),
        "quadratic": rounds_needed(ref, noisy(0, 0.01 * T**2)),
        "second_diff_spread": float(np.ptp(ref.difference(0.01 * T**2, 2))),
        "gentle": gentle,
        "boundary": np.sqrt(12 / 11) * np.sqrt(2) * SIGMA / (2 * N),
        "overdiff": rows, "noise_passes": passes,
        "walk": pass_rate(ref, lambda s: np.cumsum(np.random.RandomState(s).normal(0, 1, N)), 0),
        "white": pass_rate(ref, lambda s: noisy(s, 0.0), 0),
        "doc_never_hurts": "differencing never hurts" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    gentle, rows = result["gentle"], result["overdiff"]
    return [
        practice.Check(
            "ANSWER: one round for a linear trend, two for a quadratic",
            result["linear"] == 1 and result["quadratic"] == 2
            and result["second_diff_spread"] < 1e-9,
            f"slope 0.1 passes at d={result['linear']}; 0.01 t^2 passes at "
            f"d={result['quadratic']}; noise-free, the second difference of b t^2 is constant "
            f"(spread {result['second_diff_spread']:.1e})",
        ),
        practice.Check(
            "FINDING: a gentle quadratic passes after one round",
            all((gentle[0.001] == SEEDS, 15 < gentle[0.005] < 70, gentle[0.01] == 0,
                 0.004 < result["boundary"] < 0.006)),
            f"after one difference, pass rates are {gentle[0.001]}/{SEEDS} at b=0.001, "
            f"{gentle[0.005]}/{SEEDS} at b=0.005, {gentle[0.01]}/{SEEDS} at b=0.01; the "
            f"half-split test predicts the boundary at b = {result['boundary']:.4f}",
        ),
        practice.Check(
            "FINDING: the check never flags over-differencing",
            all((result["doc_never_hurts"], *result["noise_passes"], rows[2][0] > 5,
                 rows[2][1] < -0.6)),
            f"white noise passes at d=0..3 ({result['noise_passes']}); variance x "
            f"{rows[1][0]:.2f} and x {rows[2][0]:.2f}, lag-1 ACF {rows[1][1]:.2f} and "
            f"{rows[2][1]:.2f} after one and two rounds, against the doc's "
            "'differencing never hurts'",
        ),
        practice.Check(
            "CONTROL: white noise always passes; a random walk sometimes does",
            result["white"] == SEEDS and 0 < result["walk"] < 30,
            f"white noise {result['white']}/{SEEDS}, random walk {result['walk']}/{SEEDS}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
