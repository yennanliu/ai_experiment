"""Exercise 2 — the ACF's top lag is 28, not 7, and the gain comes from averaging four weeks.

    **Lag selection.** Compute ACF on a seasonal series (period=7). Which lags
    have the highest autocorrelation? Create lag features using only those lags
    (not consecutive lags). Does accuracy improve compared to using lags 1
    through 7?

Reading of the exercise: the seasonal series is the lesson's own
`make_seasonal_series(period=7)`; the ACF is the lesson's `autocorrelation` up to
lag 30, on the raw series and on its first difference (the lesson's own demo
uses the difference). "Those lags" are the top four by ACF of the differenced
series. Both feature sets are cut from one `make_lag_features(series, 30)` call so
they score the same rows, fitted by the lesson's `SimpleAR` and scored by
walk-forward MSE (`walk_forward_split`, 5 folds, min_train 60).

**ANSWER: lags 28, 7, 14, 21 (ACF 0.684, 0.676, 0.672, 0.664), and yes.**
Walk-forward MSE falls from 5.17 with lags 1-7 to 3.81 with those four, 26%
lower, against a noise floor of 1.5^2 = 2.25.

**FINDING: the strongest lag is 28, not the period.** On the raw series lag 28
(0.855) beats lag 7 (0.738), because the generator also carries a period-30
sine and 28 is the multiple of 7 nearest to 30. Ranked by |ACF| the next four
are all negative -- lags 4, 10, 17, 25 at about -0.64, the half-period
anti-phase -- so "highest autocorrelation" taken by magnitude picks lags that
predict the sign-flipped value.

**FINDING: selection is not what helps; averaging is.** Lag 7 alone scores
10.70 and lags 7 and 14 score 10.72, both twice as bad as lags 1-7. The four
seasonal copies win because each carries independent noise and the fit averages
it down (`sigma^2 (1 + 1/k)` for k copies).

**CONTROL:** all 30 lags score 3.10; the seasonal-naive forecast `y[t-7]`
scores 12.23.

Structure: `top_lags` ranks the ACF; `walk_forward` scores any column subset.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "15-time-series"
MAX_LAG, MIN_TRAIN, FLOOR = 30, 60, 1.5**2


def top_lags(acf, count, signed=True):
    values = acf[1:] if signed else np.abs(acf[1:])
    return [int(lag) + 1 for lag in np.argsort(-values)[:count]]


def walk_forward(ref, X, y, lags):
    cols = X[:, [lag - 1 for lag in lags]]
    scores = []
    for train, test in ref.walk_forward_split(len(cols), 5, MIN_TRAIN):
        model = ref.SimpleAR(len(lags)).fit(cols[train], y[train])
        scores.append(ref.mse(y[test], model.predict(cols[test])))
    return float(np.mean(scores))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "time_series")
    series = ref.make_seasonal_series(period=7)
    raw = ref.autocorrelation(series, MAX_LAG)
    diff = ref.autocorrelation(ref.difference(series), MAX_LAG)
    chosen = top_lags(diff, 4)
    X, y = ref.make_lag_features(series, MAX_LAG)
    sets = {"1-7": list(range(1, 8)), "chosen": chosen, "7": [7], "7,14": [7, 14],
            "all": list(range(1, MAX_LAG + 1))}
    naive = [ref.mse(y[te], X[te, 6]) for _, te in ref.walk_forward_split(len(X), 5, MIN_TRAIN)]
    return {
        "chosen": chosen, "chosen_acf": [float(diff[lag]) for lag in chosen],
        "raw_top": top_lags(raw, 2), "raw_acf": (float(raw[28]), float(raw[7])),
        "abs_next": top_lags(diff, 8, signed=False)[4:],
        "abs_next_acf": [float(diff[lag]) for lag in top_lags(diff, 8, signed=False)[4:]],
        "mse": {name: walk_forward(ref, X, y, lags) for name, lags in sets.items()},
        "naive": float(np.mean(naive)),
    }


def verify(result):
    mse, chosen = result["mse"], result["chosen"]
    return [
        practice.Check(
            "ANSWER: the top lags are 28, 7, 14, 21, and they beat lags 1-7",
            sorted(chosen) == [7, 14, 21, 28] and mse["chosen"] < 0.8 * mse["1-7"],
            f"top four by ACF of the differenced series: {chosen} "
            f"({', '.join(f'{v:.3f}' for v in result['chosen_acf'])}); walk-forward MSE "
            f"{mse['1-7']:.2f} with lags 1-7 against {mse['chosen']:.2f}, noise floor {FLOOR}",
        ),
        practice.Check(
            "FINDING: the strongest lag is 28, and |ACF| next picks the anti-phase lags",
            result["raw_top"][0] == 28 and all(v < -0.6 for v in result["abs_next_acf"]),
            f"raw ACF at 28 is {result['raw_acf'][0]:.3f} against {result['raw_acf'][1]:.3f} at 7 "
            f"(the generator adds a period-30 sine); by |ACF| the next four are "
            f"{result['abs_next']} at {', '.join(f'{v:.2f}' for v in result['abs_next_acf'])}",
        ),
        practice.Check(
            "FINDING: one or two seasonal lags lose to lags 1-7; the gain is averaging",
            min(mse["7"], mse["7,14"]) > 1.5 * mse["1-7"],
            f"lag 7 alone {mse['7']:.2f}, lags 7 and 14 {mse['7,14']:.2f}, against "
            f"{mse['1-7']:.2f} for lags 1-7 and {mse['chosen']:.2f} for four seasonal copies",
        ),
        practice.Check(
            "CONTROL: more lags approach the floor; seasonal naive does not",
            FLOOR < mse["all"] < mse["chosen"] and result["naive"] > 2 * mse["1-7"],
            f"all {MAX_LAG} lags {mse['all']:.2f}; seasonal naive y[t-7] {result['naive']:.2f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
