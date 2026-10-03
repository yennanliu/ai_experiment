"""Exercise 4 — stacking logistic regression, a tree and k-NN under a logistic meta-learner.

    Build a stacking ensemble with three base models (logistic regression,
    decision tree, k-nearest neighbors) and a logistic regression meta-learner.
    Use 5-fold cross-validation to generate meta-features. Compare to each base
    model alone.

Reading of the exercise: five training sets of 400 rows are drawn from the
lesson's `make_classification_data` (seeds 0-4) and every model is scored on one
shared 4,000-row test draw (seed 7), because the lesson's own 100-row test split
has a standard error of about 3 points -- larger than the effect. Base models
are scikit-learn's: logistic regression, a depth-5 tree (the lesson's demo
depth) and 15-NN (about sqrt of a fold's 320 rows). Meta-features are each base
model's out-of-fold probability under shuffled 5-fold CV; the meta-learner is a
logistic regression on those three columns.

**ANSWER: stacking beats the best base model on 4 of 5 training sets,** by 0.78
points on average (0.9056 against k-NN's 0.8978); logistic regression averages
0.8601 and the tree 0.8733. On seed 0 k-NN alone wins, 0.9050 to 0.8995.

**FINDING: fitting the meta-learner in-sample doubles the tree's say and loses
every time.** With meta-features from base models scored on their own training
rows, the tree's weight rises from 1.74-2.04 to 4.25-4.97 -- a depth-5 tree is
confident on rows it memorised -- and accuracy falls on all 5 sets, to a mean of
0.8841. That is the leak the doc warns about, measured.

**FINDING: the lesson's StackingClassifier is a majority vote.** It feeds hard
+/-1 labels to a tanh meta-learner; with three voters and a small bias, any
weights where none exceeds the other two combined reproduce the vote. Its
predictions match the plain majority vote on 92.8-100% of test rows (exactly, on
2 of 5 sets), and it averages 0.8980 -- below the probability stack on 4 of 5.

**CONTROL:** scikit-learn's `StackingClassifier` with the same folds disagrees
with this stack on 0 of 20,000 test predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "11-ensemble-methods"
TRAIN_SEEDS, N_TRAIN, N_TEST = range(5), 400, 4000
BASES = {"logreg": LogisticRegression,
         "tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=0),
         "knn": lambda: KNeighborsClassifier(n_neighbors=15)}


class HardLabels:
    """A base model in the shape the lesson's StackingClassifier expects."""

    def __init__(self, make):
        self.make, self.model = make, None

    def fit(self, X, y):
        self.model = self.make().fit(X, y)

    def predict(self, X):
        return self.model.predict(X).astype(float)


def probabilities(models, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models])


def one_training_set(ref, seed, X_te, y_te):
    X, y = ref.make_classification_data(n_samples=N_TRAIN, seed=seed)
    folds = KFold(5, shuffle=True, random_state=0)
    fitted = {name: make().fit(X, y) for name, make in BASES.items()}
    acc = {name: float((m.predict(X_te) == y_te).mean()) for name, m in fitted.items()}
    test_feats = probabilities(fitted.values(), X_te)
    oof = np.column_stack([cross_val_predict(make(), X, y, cv=folds, method="predict_proba")[:, 1]
                           for make in BASES.values()])
    meta = LogisticRegression().fit(oof, y)
    leaky = LogisticRegression().fit(probabilities(fitted.values(), X), y)
    acc["stack"] = float(((stacked := meta.predict(test_feats)) == y_te).mean())
    acc["leaky"] = float((leaky.predict(test_feats) == y_te).mean())
    reference = StackingClassifier(list(zip(BASES, (m() for m in BASES.values()))),
                                   final_estimator=LogisticRegression(), cv=folds).fit(X, y)
    lesson = ref.StackingClassifier([lambda m=m: HardLabels(m) for m in BASES.values()])
    lesson.fit(X, y)
    vote = np.sign(np.sum([m.predict(X_te) for m in fitted.values()], axis=0))
    acc["lesson"] = lesson.accuracy(X_te, y_te)
    return acc, {"tree_w": meta.coef_[0][1], "leaky_tree_w": leaky.coef_[0][1],
                 "sk_disagree": int((reference.predict(X_te) != stacked).sum()),
                 "lesson_vs_vote": float((lesson.predict(X_te) == vote).mean())}


def summarise(accs, extras):
    """Means over training sets plus the per-set comparisons the checks quote."""
    col = {k: np.array([a[k] for a in accs]) for k in accs[0]}
    ext = {k: np.array([e[k] for e in extras]) for k in extras[0]}
    best_base = np.max([col[k] for k in BASES], axis=0)
    span = {k: (v.min(), v.max()) for k, v in ext.items()}
    return {"mean": {k: float(v.mean()) for k, v in col.items()}, "span": span,
            "wins": int((col["stack"] > best_base).sum()), "sets": len(accs),
            "gain": float((col["stack"] - best_base).mean()),
            "leaky_loses": int((col["leaky"] < col["stack"]).sum()),
            "lesson_below": int((col["lesson"] < col["stack"]).sum()),
            "vote_exact": int((ext["lesson_vs_vote"] == 1.0).sum()),
            "sk_disagree": int(ext["sk_disagree"].sum())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "ensembles")
    X_te, y_te = ref.make_classification_data(n_samples=N_TEST, seed=7)
    runs = [one_training_set(ref, seed, X_te, y_te) for seed in TRAIN_SEEDS]
    return summarise([a for a, _ in runs], [e for _, e in runs])


def verify(result):
    mean, n, span = result["mean"], result["sets"], result["span"]
    (w0, w1), (l0, l1), (v0, v1) = (span[k] for k in ("tree_w", "leaky_tree_w", "lesson_vs_vote"))
    return [
        practice.Check(
            f"ANSWER: stacking beats the best base model on {result['wins']} of {n} training sets",
            result["wins"] >= 4 and mean["stack"] > max(mean["logreg"], mean["tree"], mean["knn"]),
            f"mean test accuracy: stack {mean['stack']:.4f}, logreg {mean['logreg']:.4f}, "
            f"tree {mean['tree']:.4f}, knn {mean['knn']:.4f}; the gain over the per-set best "
            f"base model averages {result['gain']:+.4f}",
        ),
        practice.Check(
            "FINDING: in-sample meta-features double the tree's weight and lose every time",
            l0 > 2 * w1 and result["leaky_loses"] == n,
            f"tree weight {w0:.2f}-{w1:.2f} out-of-fold against {l0:.2f}-{l1:.2f} in-sample; the "
            f"leaky stack averages {mean['leaky']:.4f}, below the honest one on "
            f"{result['leaky_loses']} of {n} sets",
        ),
        practice.Check(
            "FINDING: the lesson's StackingClassifier is a majority vote",
            v0 > 0.9 and result["lesson_below"] >= 4,
            f"on hard +/-1 base labels its tanh meta-learner agrees with the plain 3-way vote on "
            f"{v0:.1%}-{v1:.1%} of test rows (exactly on {result['vote_exact']} sets) and "
            f"averages {mean['lesson']:.4f}, below the probability stack on "
            f"{result['lesson_below']} of {n} sets",
        ),
        practice.Check(
            "CONTROL: scikit-learn's StackingClassifier makes the same predictions",
            result["sk_disagree"] == 0,
            f"{result['sk_disagree']} disagreements over {n * N_TEST:,} test predictions, same "
            "folds and meta-learner",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
