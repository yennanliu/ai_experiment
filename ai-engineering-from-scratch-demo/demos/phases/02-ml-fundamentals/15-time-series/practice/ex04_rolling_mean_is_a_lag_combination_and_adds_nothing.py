"""Exercise 4 — day-of-week does all the work; a 7-day rolling mean is a sum of the lags.

    **Feature engineering.** Add rolling mean (window=7), rolling std
    (window=7), and day-of-week features to the lag features. Compare accuracy
    with and without these extras using walk-forward validation.

Reading of the exercise: the series is the lesson's `make_seasonal_series`
(daily, weekly cycle), so day of week is `t mod 7`, one-hot with one level
dropped. The rolling window covers `y[t-1] .. y[t-7]`, never `y[t]`. Lags are the
lesson's `make_lag_features` with 7 lags, fitted by `SimpleAR` and scored by
walk-forward MSE (5 folds, min_train 60); 1, 3 and 14 lags are run as well.

**ANSWER: the extras help, 5.32 -> 3.82 MSE (28%), and day-of-week is all of
it.** Adding only day-of-week gives 3.81; adding only the rolling mean or only
the rolling std leaves 5.32 unchanged.

**FINDING: with 7 lags the rolling mean is not a new feature.** It is the
average of the seven lag columns exactly (max difference 0.0), so the design
matrix has rank 8 for 9 columns and least squares cannot use it. It does help
when the lags stop short: with 3 lags, 10.13 -> 5.96, because it reaches lags
4-7.

**FINDING: with 14 lags none of the extras help.** 4.00 with or without them
(4.01 with all three): the lags already contain the weekly phase. Extras
substitute for missing lags; they do not add information the lags lack.

**CONTROL: the alignment trap, in one line.** Let the window include `y[t]` and
walk-forward MSE drops to about 1e-25, because `y[t] = 7 * rolling_mean -
(y[t-1] + .. + y[t-6])` exactly -- the leak the doc warns about, reproduced.

Structure: `features` builds the blocks; `walk_forward` scores a column stack.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "15-time-series"
WINDOW = 7


def features(ref, series, n_lags, include_now=False):
    X, y = ref.make_lag_features(series, max(n_lags, WINDOW))
    t = np.arange(len(series))[len(series) - len(y):]
    window = np.column_stack([y, X[:, :WINDOW - 1]]) if include_now else X[:, :WINDOW]
    return {"lags": X[:, :n_lags], "mean": window.mean(1, keepdims=True),
            "std": window.std(1, keepdims=True), "dow": np.eye(7)[t % 7][:, 1:]}, y


def walk_forward(ref, blocks, names, y):
    X = np.column_stack([blocks[name] for name in names])
    scores = [ref.mse(y[te], ref.SimpleAR(X.shape[1]).fit(X[tr], y[tr]).predict(X[te]))
              for tr, te in ref.walk_forward_split(len(X), 5, 60)]
    return float(np.mean(scores))


def table(ref, series, n_lags):
    blocks, y = features(ref, series, n_lags)
    combos = {"lags": ["lags"], "+mean": ["lags", "mean"], "+std": ["lags", "std"],
              "+dow": ["lags", "dow"], "+all": ["lags", "mean", "std", "dow"]}
    return {name: walk_forward(ref, blocks, cols, y) for name, cols in combos.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "time_series")
    series = ref.make_seasonal_series(period=7)
    blocks, _ = features(ref, series, 7)
    design = np.column_stack([np.ones(len(blocks["lags"])), blocks["lags"], blocks["mean"]])
    leaky, y = features(ref, series, 7, include_now=True)
    return {
        "tables": {n: table(ref, series, n) for n in (3, 7, 14)},
        "mean_gap": float(np.abs(blocks["mean"][:, 0] - blocks["lags"].mean(1)).max()),
        "rank": (int(np.linalg.matrix_rank(design)), design.shape[1]),
        "leaky": walk_forward(ref, leaky, ["lags", "mean", "std", "dow"], y),
    }


def verify(result):
    t3, t7, t14 = (result["tables"][n] for n in (3, 7, 14))
    return [
        practice.Check(
            "ANSWER: the extras cut MSE by about a quarter, all of it day-of-week",
            t7["+all"] < 0.8 * t7["lags"] and abs(t7["+dow"] - t7["+all"]) < 0.05
            and abs(t7["+std"] - t7["lags"]) < 0.01,
            f"7 lags {t7['lags']:.2f}; +mean {t7['+mean']:.2f}, +std {t7['+std']:.2f}, "
            f"+day-of-week {t7['+dow']:.2f}, all three {t7['+all']:.2f}",
        ),
        practice.Check(
            "FINDING: the 7-day rolling mean is the mean of the 7 lag columns",
            all((result["mean_gap"] < 1e-12, result["rank"][0] == result["rank"][1] - 1,
                 abs(t7["+mean"] - t7["lags"]) < 1e-6, t3["+mean"] < 0.7 * t3["lags"])),
            f"max difference {result['mean_gap']:.1e}, design rank {result['rank'][0]} of "
            f"{result['rank'][1]}; with only 3 lags it does help, {t3['lags']:.2f} -> "
            f"{t3['+mean']:.2f}",
        ),
        practice.Check(
            "FINDING: with 14 lags no extra helps",
            all(abs(v - t14["lags"]) < 0.05 for v in t14.values()),
            f"14 lags {t14['lags']:.3f}; with extras "
            f"{', '.join(f'{k} {v:.3f}' for k, v in t14.items() if k != 'lags')}",
        ),
        practice.Check(
            "CONTROL: a window that includes y[t] predicts it exactly",
            result["leaky"] < 1e-12,
            f"walk-forward MSE {result['leaky']:.1e}: y[t] = 7 * mean - (y[t-1] + .. + y[t-6])",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
