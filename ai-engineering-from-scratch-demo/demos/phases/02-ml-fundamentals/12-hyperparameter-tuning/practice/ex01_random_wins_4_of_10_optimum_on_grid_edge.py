"""Exercise 1 — grid against random search at an equal budget, over ten seeds.

    Run grid search and random search with the same total budget (e.g., 50
    evaluations). Compare the best scores found. Run the experiment 10 times
    with different seeds. How often does random search win?

Reading of the exercise: the budget is the doc's own worked example -- the 3x3
grid lr {0.01, 0.1, 1.0} x max_depth {3, 5, 7}, 9 evaluations -- against 9
random draws over the same box (log-uniform lr, integer depth 3-7). "Different
seeds" reseeds both the data (`make_data(seed=s)`) and the random search, so the
grid is not a constant. The lesson's `grid_search` and `random_search` run
unchanged; only the model class they instantiate is swapped for scikit-learn's
`GradientBoostingRegressor` with `GBMForTuning`'s signature and defaults (50
trees), because 360 fits of the pure-Python GBM take minutes. The last check
confirms the swap does not change the conclusion.

**ANSWER: random search wins 4 of 10.** Mean best validation MSE is 0.6117 for
the grid and 0.6435 for random search.

**FINDING: the grid wins because the optimum sits on its edge.** The grid's best
cell has depth 3, its smallest value, on 10 of 10 seeds. Random search draws
depth 3 on only 19 of 90 trials (21%), and its winner has depth 3 on 3 seeds.
The doc's argument assumes the second hyperparameter barely matters; depth
matters here, and a grid always visits its corners.

**CONTROL: make the second axis unimportant and random search wins 10 of 10.**
The same contest with min_samples_split {2, 11, 20} in place of depth: mean best
0.4684 for random against 0.5889 for the grid. That is the doc's Bergstra-Bengio
case -- 9 distinct learning rates against 3 -- and it needs the second axis not
to matter.

**CONTROL: the lesson's own GBM agrees that depth 3 beats depth 5.** At lr 0.1
on seed 0, `GBMForTuning` scores 0.6836 against 0.9621 and the stand-in 0.7451
against 0.9011.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "12-hyperparameter-tuning"
SEEDS, BUDGET = range(10), 9
DEPTH_GRID = {"learning_rate": [0.01, 0.1, 1.0], "max_depth": [3, 5, 7]}  # the doc's own grid
DEPTH_DIST = {"learning_rate": ("log_float", 0.01, 1.0), "max_depth": ("int", 3, 7)}
SPLIT_GRID = {"learning_rate": [0.01, 0.1, 1.0], "min_samples_split": [2, 11, 20]}
SPLIT_DIST = {"learning_rate": ("log_float", 0.01, 1.0), "min_samples_split": ("int", 2, 20)}


def sklearn_gbm(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=5,
                subsample=1.0):
    """GBMForTuning's signature and defaults, fitted by scikit-learn."""
    return GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        min_samples_split=min_samples_split, subsample=subsample, random_state=0)


def contest(ref, grid, dist):
    """Best validation MSE of grid and of random search, per seed."""
    rows = []
    for seed in SEEDS:
        data = ref.make_data(seed=seed)[:4]
        grid_params, grid_score, _ = ref.grid_search(grid, *data)
        rand_params, rand_score, history = ref.random_search(dist, *data, n_iter=BUDGET, seed=seed)
        rows.append({"grid": -grid_score, "random": -rand_score, "grid_params": grid_params,
                     "random_params": rand_params, "tried": [p for p, _ in history]})
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "tuning")
    lesson_gbm = ref.GBMForTuning
    ref.GBMForTuning = sklearn_gbm  # grid_search/random_search run unchanged on top of it
    depth = contest(ref, DEPTH_GRID, DEPTH_DIST)
    split = contest(ref, SPLIT_GRID, SPLIT_DIST)
    X, y, X_val, y_val = ref.make_data(seed=0)[:4]
    fidelity = []
    for d in (3, 5):
        ours, theirs = lesson_gbm(max_depth=d), sklearn_gbm(max_depth=d).fit(X, y)
        ours.fit(X, y)
        fidelity.append((-ref.neg_mse(ours, X_val, y_val), -ref.neg_mse(theirs, X_val, y_val)))
    return {"depth": depth, "split": split, "fidelity": fidelity}


def tally(rows):
    wins = sum(r["random"] < r["grid"] for r in rows)
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("grid", "random")}
    return wins, mean


def verify(result):
    depth, split, (d3, d5) = result["depth"], result["split"], result["fidelity"]
    (wins, mean), (split_wins, split_mean) = tally(depth), tally(split)
    grid_d3 = sum(r["grid_params"]["max_depth"] == 3 for r in depth)
    tried = [p["max_depth"] for r in depth for p in r["tried"]]
    rand_d3 = sum(r["random_params"]["max_depth"] == 3 for r in depth)
    return [
        practice.Check(
            f"ANSWER: random search wins {wins} of {len(depth)} at a budget of {BUDGET}",
            wins <= 5,
            f"the doc's own 3x3 grid (lr 0.01/0.1/1.0 x depth 3/5/7) against {BUDGET} random draws "
            f"(log-uniform lr, depth 3-7), data and search reseeded together: mean best "
            f"validation MSE {mean['grid']:.4f} for grid, {mean['random']:.4f} for random",
        ),
        practice.Check(
            "FINDING: the grid wins because the optimum sits on its edge",
            grid_d3 == len(depth) and rand_d3 < len(depth) / 2,
            f"the grid's best cell has depth 3, its smallest, on {grid_d3} of {len(depth)} seeds; "
            f"random search draws depth 3 on {tried.count(3)} of {len(tried)} trials "
            f"({tried.count(3) / len(tried):.0%}) and its winner has depth 3 on {rand_d3} seeds. "
            "Depth is not an unimportant dimension here, and a grid always visits its corners",
        ),
        practice.Check(
            "CONTROL: make the second axis unimportant and random search wins every time",
            split_wins == len(split),
            f"the same contest with min_samples_split 2/11/20 in place of depth: random wins "
            f"{split_wins} of {len(split)}, mean best {split_mean['random']:.4f} against "
            f"{split_mean['grid']:.4f}. That is the doc's Bergstra-Bengio case; it needs the "
            "second axis not to matter",
        ),
        practice.Check(
            "CONTROL: the lesson's own GBM agrees that depth 3 beats depth 5",
            d3[0] < 0.8 * d5[0] and d3[1] < 0.9 * d5[1],
            f"at lr 0.1, 50 trees, seed 0: GBMForTuning {d3[0]:.4f} (depth 3) against "
            f"{d5[0]:.4f} (depth 5); the scikit-learn stand-in {d3[1]:.4f} against {d5[1]:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
