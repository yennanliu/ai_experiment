"""Exercise 2 — nested CV shows the shortcut flatters kNN; stratified folds dump the minority last.

    Build a nested cross-validation loop: the outer loop evaluates model
    performance, the inner loop tunes hyperparameters. Use it to compare two
    models fairly without leaking validation data into the evaluation.

Reading of the exercise: the two models are the lesson's `SimpleLogistic`
(tuned over epochs 1, 3, 10, 30) and a k-nearest-neighbour classifier written
here (tuned over k = 1, 3, 7, 15, 31). The outer loop is the lesson's
`kfold_split` (5 folds); the inner loop is the lesson's `cross_validate`
(3 folds) on the outer training part only. The leak it prevents is measured
against the usual shortcut, reporting the best grid point's 5-fold CV score,
on 8 datasets of 120 from `make_classification_data` and on the same inputs
with random labels, where the right answer is 50%.

| | logistic | kNN | gap |
|---|---:|---:|---:|
| nested CV, 8 real datasets | 0.8865 | 0.8583 | **0.0281** |
| best-of-grid CV, same data | 0.8969 | 0.8781 | 0.0187 |
| nested CV, 24 random-label sets | 0.5052 | 0.5031 | — |
| best-of-grid CV, random labels | 0.5038 | **0.5372** | — |

**ANSWER: nested CV ranks logistic above kNN by 2.8 points; the shortcut says
1.9.** Both agree on the winner, but not on the margin.

**FINDING: the shortcut's optimism belongs to kNN alone, which is what makes it
unfair.** On random labels kNN's best grid point scores 0.5372 against a nested
0.5031; logistic's grid (epochs 1-30 of a linear model) barely varies, 0.5038 vs
0.5052. On real labels the shortcut inflates kNN by 0.020 and logistic by
0.010, so it understates logistic's lead by a third. The leak scales with how
different the grid's models are, not with how many hyperparameters there are.

**FINDING: the lesson's `stratified_kfold_split` puts each class's remainder in
the last fold.** On `make_imbalanced_data(300, 0.05)`'s 13 positives it gives
[2, 2, 2, 2, 5] per fold, where round-robin would give 3, 3, 3, 2, 2 and plain
`kfold_split` happened to give [1, 4, 3, 2, 3]. With 4 positives it gives
[0, 0, 0, 0, 4] -- every minority sample in one fold, the exact failure
`docs/en.md` says stratification prevents.

**CONTROL:** the outer `kfold_split` folds partition the 120 rows, so each row
is evaluated once and never used to tune its own fold.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "09-model-evaluation"
DATASETS, NULL_DATASETS, N = 8, 24, 120


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y

    def predict(self, x):
        dist = [sum((a - b) ** 2 for a, b in zip(row, x)) for row in self.X]
        near = sorted(range(len(dist)), key=dist.__getitem__)[: self.k]
        return int(2 * sum(self.y[i] for i in near) > self.k)


def grids(ref):
    return {
        "logistic": [lambda e=e: ref.SimpleLogistic(lr=0.1, epochs=e) for e in (1, 3, 10, 30)],
        "knn": [lambda k=k: KNN(k) for k in (1, 3, 7, 15, 31)],
    }


def cv(ref, X, y, model_fn, k):
    return statistics.mean(ref.cross_validate(X, y, model_fn, k=k, metric_fn=ref.accuracy))


def nested(ref, X, y, grid):
    """Outer 5-fold estimate; each outer fold tunes on its own training part only."""
    scores = []
    for train, val in ref.kfold_split(len(X), 5, seed=1):
        X_tr, y_tr = [X[i] for i in train], [y[i] for i in train]
        inner = [cv(ref, X_tr, y_tr, fn, 3) for fn in grid]
        model = grid[inner.index(max(inner))]()
        model.fit(X_tr, y_tr)
        scores.append(statistics.mean(model.predict(X[i]) == y[i] for i in val))
    return statistics.mean(scores)


def evaluate(ref, null, count):
    """Mean (best-of-grid CV, nested CV) per model over `count` datasets."""
    out = {name: ([], []) for name in ("logistic", "knn")}
    for d in range(count):
        X, y = ref.make_classification_data(N, seed=200 + d)
        if null:
            y = random.Random(d).choices((0, 1), k=len(y))
        for name, grid in grids(ref).items():
            out[name][0].append(max(cv(ref, X, y, fn, 5) for fn in grid))
            out[name][1].append(nested(ref, X, y, grid))
    return {name: tuple(map(statistics.mean, pair)) for name, pair in out.items()}


def minority_per_fold(ref, y, k=5):
    return [sum(y[i] for i in val) for _, val in ref.stratified_kfold_split(y, k)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "evaluation")
    _, y_imb = ref.make_imbalanced_data(300, minority_ratio=0.05)
    folds = ref.kfold_split(N, 5, seed=1)
    return {
        "real": evaluate(ref, False, DATASETS),
        "null": evaluate(ref, True, NULL_DATASETS),
        "strat": minority_per_fold(ref, y_imb),
        "strat4": minority_per_fold(ref, [1] * 4 + [0] * 96),
        "plain": [sum(y_imb[i] for i in val) for _, val in ref.kfold_split(300, 5)],
        "doc": "might put all minority samples in one fold" in parity.doc_text(PHASE, LESSON),
        "covered": sorted(i for _, val in folds for i in val) == list(range(N)),
    }


def verify(result):
    real, null, strat, strat4 = result["real"], result["null"], result["strat"], result["strat4"]
    (lf, ln), (kf, kn) = real["logistic"], real["knn"]
    (nlf, nln), (nkf, nkn) = null["logistic"], null["knn"]
    return [
        practice.Check(
            "ANSWER: nested CV ranks logistic above kNN by 2.8 points, the shortcut by 1.9",
            all((ln > kn, lf > kf, ln - kn > lf - kf)),
            f"nested: logistic {ln:.4f}, kNN {kn:.4f} (gap {ln - kn:.4f}); best-of-grid CV: "
            f"{lf:.4f} vs {kf:.4f} (gap {lf - kf:.4f}), mean of {DATASETS} datasets of {N}",
        ),
        practice.Check(
            "FINDING: the shortcut's optimism is all kNN's, which is what makes it unfair",
            all((nkf - nkn > 0.025, abs(nlf - nln) < 0.01, abs(nkn - 0.5) < 0.02)),
            f"on {NULL_DATASETS} random-label datasets kNN's best-of-grid CV reports {nkf:.4f} "
            f"against nested {nkn:.4f}; logistic's grid barely varies, {nlf:.4f} vs {nln:.4f}. "
            f"On real labels the shortcut inflates kNN by {kf - kn:.3f} and logistic by "
            f"{lf - ln:.3f}, so it understates logistic's lead by a third",
        ),
        practice.Check(
            "FINDING: the lesson's stratified split puts the remainder in the last fold",
            all((result["doc"], strat == [2, 2, 2, 2, 5], strat4 == [0, 0, 0, 0, 4])),
            f"on make_imbalanced_data(300, 0.05)'s 13 positives stratified_kfold_split gives "
            f"{strat} per fold (round-robin would give 3,3,3,2,2; plain kfold gave "
            f"{result['plain']}); with 4 positives it gives {strat4}: all the minority in one "
            "fold, the case the doc says stratification exists to prevent",
        ),
        practice.Check(
            "CONTROL: the outer kfold_split validates every row exactly once",
            result["covered"],
            f"the 5 outer folds of {N} rows are a partition, so the nested estimate uses each "
            "row once for evaluation and never for its own fold's tuning",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
