"""Exercise 5 — a histogram gradient booster against the lesson's from-scratch GBM.

    Run XGBoost on the same dataset with default parameters. Compare its
    accuracy to your from-scratch gradient boosting. Time both. How large is the
    speed difference?

Reading of the exercise: `xgboost` is not a dependency of this repo, so this
ships the scaled-down runnable DESIGN D11 asks for: scikit-learn's
`HistGradientBoostingClassifier` -- the same histogram-binned, second-order,
log-loss booster XGBoost and LightGBM popularised -- at its default parameters,
against the lesson's `GradientBoostingScratch` (100 depth-3 trees, lr 0.1,
squared error on +/-1 labels, sign to classify). "The same dataset" is the
lesson's own comparison data: `make_classification_data(500)`, split 400/100. A
fresh 4,000-row draw is added because 100 test rows cannot resolve the gap. The
real run is `pip install xgboost` then `XGBClassifier().fit(X, y)`. Timings are
machine-dependent, so the checks assert only orderings; numbers below are one
run on the author's machine.

**ANSWER: about 6x faster at the lesson's size, and 1.8 points more accurate.**
Fitting takes about 0.78 s for the scratch GBM and 0.13 s for the histogram
booster (best of 3). Accuracy on the 4,000 fresh rows is 0.9042 against 0.9220.

**FINDING: the speed gap is a function of n, not a constant.** Per tree, the
scratch GBM costs about 9 ms at 400 rows and 64 ms at 20,000; the histogram
booster about 1.2 ms and 3.4 ms. The ratio grows from about 7x to about 19x,
because the scratch tree pays numpy call overhead at small n and full passes
over the data at large n, while binning makes the booster's split search nearly
independent of n.

**FINDING: the lesson's 100-row test set triples the accuracy gap.** It scores
0.89 against 0.95, a 6-point gap; 4,000 rows put it at 1.8 points. One test row
is a full point and the standard error of a 100-row accuracy is ~3 points.

**FINDING: what is left of the gap is the loss, not the engineering.** Capped at
the scratch GBM's depth 3, the histogram booster still scores 0.9235, while
scikit-learn's exact `GradientBoostingRegressor` on the same squared-error-on-
labels objective scores 0.8985. Squared error on +/-1 targets keeps pulling
already-correct points back toward the label; log-loss stops caring once a point
is confidently right.

**CONTROL:** that squared-error regressor, an independent implementation of the
lesson's algorithm, agrees with the scratch GBM on 94.9% of the 4,000 rows.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "11-ensemble-methods"
BIG_N, SCRATCH_TREES_BIG = 20_000, 5
REAL_COMMAND = "pip install xgboost, then XGBClassifier().fit(X_train, y_train)"


def timed(fn, repeats=1):
    """Best wall-clock time of `repeats` calls, and the last result."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return out, best


def per_tree_seconds(ref, X, y, n_scratch):
    """Fit seconds per tree: the lesson's GBM against the histogram booster's default 100."""
    scratch = ref.GradientBoostingScratch(n_estimators=n_scratch)
    _, t_scratch = timed(lambda: scratch.fit(X, y.astype(float)))
    hist, t_hist = timed(lambda: HistGradientBoostingClassifier(random_state=0).fit(X, y), 3)
    return t_scratch / n_scratch, t_hist / hist.n_iter_


def solve():
    ref = parity.load_reference(PHASE, LESSON, "ensembles")
    X, y = ref.make_classification_data(n_samples=500)  # the lesson's comparison data
    X_tr, X_te, y_tr, y_te = ref.train_test_split(X, y)
    X_big, y_big = ref.make_classification_data(n_samples=4000, seed=7)
    scratch = ref.GradientBoostingScratch()
    _, t_scratch = timed(lambda: scratch.fit(X_tr, y_tr.astype(float)))
    hist, t_hist = timed(lambda: HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr), 3)
    shallow = HistGradientBoostingClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
    squared = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
    X_huge, y_huge = ref.make_classification_data(n_samples=BIG_N, seed=1)
    ratios = {n: per_tree_seconds(ref, Xs, ys, k) for n, Xs, ys, k in (
        (len(y_tr), X_tr, y_tr, 20), (BIG_N, X_huge, y_huge, SCRATCH_TREES_BIG))}
    scratch_big = np.sign(scratch.predict(X_big))

    def acc(pred, truth):
        return float((pred == truth).mean())

    return {
        "small": {"scratch": acc(np.sign(scratch.predict(X_te)), y_te),
                  "hist": acc(hist.predict(X_te), y_te), "n": len(y_te)},
        "big": {"scratch": acc(scratch_big, y_big), "hist": acc(hist.predict(X_big), y_big),
                "hist_depth3": acc(shallow.predict(X_big), y_big),
                "squared": acc(np.sign(squared.predict(X_big)), y_big), "n": len(y_big)},
        "agree_squared": acc(scratch_big, np.sign(squared.predict(X_big))),
        "fit": (t_scratch, t_hist), "per_tree": ratios,
    }


def verify(result):
    small, big = result["small"], result["big"]
    (t_scratch, t_hist), per_tree = result["fit"], result["per_tree"]
    ratio = {n: a / b for n, (a, b) in per_tree.items()}
    small_n, big_n = sorted(ratio)
    gap_small, gap_big = small["hist"] - small["scratch"], big["hist"] - big["scratch"]
    return [
        practice.Check(
            f"ANSWER: the booster fits {t_scratch / t_hist:.0f}x faster and scores "
            f"{gap_big:+.3f} on {big['n']:,} fresh rows",
            t_scratch > 1.5 * t_hist and gap_big > 0,
            f"fit: scratch {t_scratch:.3f} s, histogram booster {t_hist:.3f} s (best of 3). "
            f"Accuracy: {big['scratch']:.4f} against {big['hist']:.4f}. xgboost is not a "
            f"dependency here; the T3 command is: {REAL_COMMAND}",
        ),
        practice.Check(
            "FINDING: the speed gap grows with n",
            ratio[big_n] > ratio[small_n],
            f"seconds per tree: scratch {per_tree[small_n][0] * 1e3:.1f} ms at {small_n} rows "
            f"and {per_tree[big_n][0] * 1e3:.1f} ms at {big_n:,}; booster "
            f"{per_tree[small_n][1] * 1e3:.1f} ms and {per_tree[big_n][1] * 1e3:.1f} ms. The "
            f"ratio goes from {ratio[small_n]:.1f}x to {ratio[big_n]:.1f}x",
        ),
        practice.Check(
            "FINDING: the lesson's 100-row test set inflates the accuracy gap",
            small["n"] == 100 and gap_small > 2 * gap_big,
            f"on its {small['n']} rows: {small['scratch']:.2f} against {small['hist']:.2f}, a "
            f"{gap_small * 100:.1f}-point gap; on {big['n']:,} rows it is {gap_big * 100:.1f} "
            "points. One row is a full point there",
        ),
        practice.Check(
            "FINDING: the remaining gap is the loss function, not tree size or engineering",
            big["hist_depth3"] > big["scratch"] and big["squared"] < big["hist_depth3"],
            f"the booster capped at depth 3 scores {big['hist_depth3']:.4f}; scikit-learn's exact "
            f"GradientBoostingRegressor on squared error over +/-1 labels scores "
            f"{big['squared']:.4f}",
        ),
        practice.Check(
            "CONTROL: an independent squared-error GBM agrees with the scratch one",
            result["agree_squared"] > 0.9 and abs(big["squared"] - big["scratch"]) < 0.01,
            f"same predicted class on {result['agree_squared']:.1%} of {big['n']:,} rows; "
            f"accuracies {big['squared']:.4f} and {big['scratch']:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
