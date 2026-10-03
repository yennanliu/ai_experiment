"""Exercise 3 — the random split flatters Ridge by 1.3x and kNN by 2x; the leak is the trend.

    **Walk-forward vs random split.** Train a Ridge regression on lag features.
    Evaluate with random 80/20 split and with walk-forward validation. How much
    does the random split overestimate performance?

Reading of the exercise: the series is the lesson's own
`make_synthetic_series(500)` with 10 lags from `make_lag_features`; walk-forward
is the lesson's `walk_forward_split(5 folds, min_train=100)`, as in its own
`demo_random_vs_walk_forward`. One shuffle is noisy, so the random 80/20 score
is averaged over 20 permutations. "Overestimate" is the ratio of the
walk-forward MSE to the random-split MSE.

**ANSWER: by about a third.** Ridge(alpha=1) scores 6.99 MSE under a random
split and 9.42 walk-forward: the random split makes the error look 26% smaller.

**FINDING: the size of the lie depends on the model, because the leak is the
trend.** A shuffled test point sits between training points in time, so a model
that cannot extrapolate is graded on interpolation. kNN (k=5) scores 9.12
random against 18.71 walk-forward, 2.05x. Remove the trend (subtract the
generator's `0.05 t`, same noise) and the gaps nearly close: kNN 4.94 against
5.26, Ridge 4.82 against 5.49.

**FINDING: the lesson's own printed ratio is a lucky shuffle.** Its demo
prints 5.92 random against 9.41 walk-forward, a ratio of 0.63, from one
`RandomState(42)` permutation. Over 200 permutations that 5.92 sits in the
luckiest 6% (mean 7.38, ratio 0.78).

**CONTROL:** Ridge(alpha=1) agrees with the lesson's `SimpleAR` (plain least
squares) to within 0.01 MSE on both protocols, so the comparison is not about
regularisation.

Structure: `scores` returns (random, walk-forward) MSE for any model factory.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "15-time-series"
N, LAGS, SHUFFLES = 500, 10, 20


def random_mse(ref, X, y, make, seed):
    idx = np.random.RandomState(seed).permutation(len(X))
    cut = int(0.8 * len(X))
    model = make().fit(X[idx[:cut]], y[idx[:cut]])
    return ref.mse(y[idx[cut:]], model.predict(X[idx[cut:]]))


def scores(ref, series, make):
    X, y = ref.make_lag_features(series, LAGS)
    rnd = np.mean([random_mse(ref, X, y, make, seed) for seed in range(SHUFFLES)])
    walk = [ref.mse(y[te], make().fit(X[tr], y[tr]).predict(X[te]))
            for tr, te in ref.walk_forward_split(len(X), 5, 100)]
    return float(rnd), float(np.mean(walk))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "time_series")
    series = ref.make_synthetic_series(N)
    flat = series - 0.05 * np.arange(N)
    models = {"ridge": lambda: Ridge(alpha=1.0), "knn": lambda: KNeighborsRegressor(5),
              "ar": lambda: ref.SimpleAR(LAGS)}
    X, y = ref.make_lag_features(series, LAGS)
    lucky = [random_mse(ref, X, y, models["ar"], seed) for seed in range(200)]
    return {
        "trend": {name: scores(ref, series, make) for name, make in models.items()},
        "flat": {name: scores(ref, flat, models[name]) for name in ("ridge", "knn")},
        "demo": lucky[42], "demo_rank": float(np.mean(np.array(lucky) < lucky[42])),
        "shuffle_mean": float(np.mean(lucky)),
    }


def verify(result):
    trend, flat = result["trend"], result["flat"]
    (r_rnd, r_wf), (k_rnd, k_wf) = trend["ridge"], trend["knn"]
    return [
        practice.Check(
            "ANSWER: the random split understates Ridge's error by about a quarter",
            1.2 < r_wf / r_rnd < 1.5,
            f"Ridge MSE {r_rnd:.2f} random (mean of {SHUFFLES} shuffles) against {r_wf:.2f} "
            f"walk-forward: {r_wf / r_rnd:.2f}x, the random split looks "
            f"{100 * (1 - r_rnd / r_wf):.0f}% better",
        ),
        practice.Check(
            "FINDING: kNN is flattered 2x, and removing the trend closes both gaps",
            k_wf / k_rnd > 1.8 and flat["knn"][1] / flat["knn"][0] < 1.15
            and flat["ridge"][1] / flat["ridge"][0] < 1.2,
            f"kNN {k_rnd:.2f} random against {k_wf:.2f} walk-forward ({k_wf / k_rnd:.2f}x); "
            f"detrended, kNN {flat['knn'][0]:.2f} against {flat['knn'][1]:.2f} and Ridge "
            f"{flat['ridge'][0]:.2f} against {flat['ridge'][1]:.2f}",
        ),
        practice.Check(
            "FINDING: the lesson's printed ratio comes from a lucky shuffle",
            result["demo_rank"] < 0.1,
            f"RandomState(42) gives {result['demo']:.2f}, below {100 * result['demo_rank']:.1f}% "
            f"of 200 shuffles (mean {result['shuffle_mean']:.2f}); ratio "
            f"{result['demo'] / trend['ar'][1]:.2f} printed against "
            f"{result['shuffle_mean'] / trend['ar'][1]:.2f} typical",
        ),
        practice.Check(
            "CONTROL: Ridge(alpha=1) is the lesson's SimpleAR to within 0.01",
            all(abs(a - b) < 0.01 for a, b in zip(trend["ridge"], trend["ar"])),
            f"SimpleAR {trend['ar'][0]:.3f} / {trend['ar'][1]:.3f} against Ridge "
            f"{r_rnd:.3f} / {r_wf:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
