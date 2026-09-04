"""Exercise 1 — KNN at K=1, 5, 15 and N: overfitting to underfitting.

    Implement KNN classification on a 2D dataset with 3 classes. Plot the
    decision boundary for K=1, K=5, K=15, and K=N. Observe the transition from
    overfitting to underfitting.

Reading of the exercise: the two ends are qualitatively different failures and
both are exactly characterisable, so neither needs a plot. K=1 scores **1.000**
on training data by construction — every point is its own nearest neighbour — so
that number is not evidence of anything. K=N predicts the global majority class
for every input, which on this 3-class split scores 0.217, *below* the 1/3 a coin
would get, because the test split's class balance differs from the training one.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "06-knn-and-distances"
SEED = 42


def solve():
    ref = parity.load_reference(PHASE, LESSON, "knn")
    X, y = ref.generate_classification_data(n_samples=300, n_classes=3, seed=SEED)
    X_train, y_train, X_test, y_test = ref.train_test_split(X, y, seed=SEED)
    ks = (1, 5, 15, len(y_train))
    rows = {}
    for k in ks:
        model = ref.KNN(k=k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows[k] = {"train": ref.accuracy(y_train, model.predict(X_train)),
                   "test": ref.accuracy(y_test, predictions),
                   "labels_used": len(set(predictions))}
    majority = max(y_train.count(c) for c in set(y_train)) / len(y_train)
    return {"rows": rows, "ks": ks, "n_train": len(y_train),
            "n_classes": len(set(y)), "train_majority": majority,
            "test_majority": max(y_test.count(c) for c in set(y_test)) / len(y_test)}


def verify(result):
    rows, ks = result["rows"], result["ks"]
    one, full = rows[1], rows[ks[-1]]
    best = max(ks, key=lambda k: rows[k]["test"])
    return [
        practice.Check("K=1 scores exactly 1.000 on training data, by construction",
                       one["train"] == 1.0,
                       f"every training point is its own nearest neighbour, so the label is "
                       f"returned unchanged — the number carries no information about the "
                       f"model at all. Test accuracy is {one['test']:.1%}"),
        practice.Check(f"K=N predicts one class for everything",
                       full["labels_used"] == 1,
                       f"{full['labels_used']} distinct label over the whole test set: with "
                       f"K = {ks[-1]} every query sees all {result['n_train']} training "
                       f"points, so the vote is the global majority regardless of input"),
        practice.Check("…and that scores below chance on this split",
                       full["test"] < 1 / result["n_classes"],
                       f"{full['test']:.1%} against {1 / result['n_classes']:.1%} for a coin "
                       f"over {result['n_classes']} classes. The training majority is "
                       f"{result['train_majority']:.1%} and the test majority "
                       f"{result['test_majority']:.1%} — predicting the wrong constant is "
                       f"worse than guessing"),
        practice.Check(f"ANSWER: test accuracy peaks at K={best}, between the two failures",
                       best not in (1, ks[-1]),
                       ", ".join(f"K={k}: train {rows[k]['train']:.1%} / test "
                                 f"{rows[k]['test']:.1%}" for k in ks)),
        practice.Check("the train-test gap closes as K grows, then both collapse",
                       one["train"] - one["test"] > rows[15]["train"] - rows[15]["test"],
                       f"gap at K=1 is {one['train'] - one['test']:+.1%} and at K=15 "
                       f"{rows[15]['train'] - rows[15]['test']:+.1%} — the classic variance "
                       f"reduction. But at K=N the gap is "
                       f"{full['train'] - full['test']:+.1%} with *both* terms near zero, "
                       f"so a small gap is no evidence of a good model"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
