"""Exercise 2 — a 3-class confusion matrix, per-class precision and recall.

    Implement a multi-class confusion matrix for the 3-class softmax model.
    Compute per-class precision and recall. Which class is hardest to classify?

Reading of the exercise: "hardest" has two answers and they need not agree —
lowest recall (the class most often missed) and lowest precision (the class most
often falsely claimed). Both are reported. The dataset is built with class 1
placed *between* 0 and 2 so its errors go both ways, which is what makes the two
measures diverge; check 5 shows the confusion is asymmetric.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "03-logistic-regression"
SEED, N_PER, EPOCHS, LR = 42, 120, 3_000, 0.2
CENTRES = ((0.0, 0.0), (1.6, 0.0), (3.2, 0.0))     # class 1 sits between 0 and 2
SPREAD = 0.9


def make_data(rng):
    X, y = [], []
    for label, (cx, cy) in enumerate(CENTRES):
        for _ in range(N_PER):
            X.append([rng.gauss(cx, SPREAD), rng.gauss(cy, SPREAD)])
            y.append(label)
    return X, y


def confusion(y_true, y_pred, k=3):
    matrix = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def per_class(matrix):
    rows = {}
    for c, row in enumerate(matrix):
        tp = row[c]
        predicted = sum(matrix[r][c] for r in range(len(matrix)))
        actual = sum(row)
        rows[c] = {"precision": tp / predicted if predicted else 0.0,
                   "recall": tp / actual if actual else 0.0}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "logistic_regression")
    rng = random.Random(SEED)
    X, y = make_data(rng)
    model = ref.SoftmaxRegression(n_features=2, n_classes=3, learning_rate=LR)
    model.fit(X, y, epochs=EPOCHS, print_every=EPOCHS + 1)
    predictions = [model.predict(x) for x in X]
    matrix = confusion(y, predictions)
    stats = per_class(matrix)
    total = sum(sum(row) for row in matrix)
    return {"matrix": matrix, "stats": stats,
            "accuracy": sum(matrix[c][c] for c in range(3)) / total,
            "lesson_accuracy": model.accuracy(X, y), "total": total,
            "worst_recall": min(stats, key=lambda c: stats[c]["recall"]),
            "worst_precision": min(stats, key=lambda c: stats[c]["precision"])}


def verify(result):
    matrix, stats = result["matrix"], result["stats"]
    return [
        practice.Check(f"the {result['total']} rows are all accounted for",
                       result["total"] == 3 * N_PER
                       and abs(result["accuracy"] - result["lesson_accuracy"]) < 1e-9,
                       f"matrix rows {matrix}; accuracy {result['accuracy']:.1%}, matching "
                       f"the lesson's own accuracy() exactly — the diagonal over the total"),
        practice.Check("every class has non-trivial precision and recall",
                       all(0.4 < s["precision"] < 1.0 and 0.4 < s["recall"] < 1.0
                           for s in stats.values()),
                       "; ".join(f"class {c}: P {s['precision']:.3f} R {s['recall']:.3f}"
                                 for c, s in stats.items())),
        practice.Check(f"ANSWER by recall: class {result['worst_recall']} is hardest",
                       result["worst_recall"] == 1,
                       f"class 1 recall {stats[1]['recall']:.3f} against "
                       f"{stats[0]['recall']:.3f} and {stats[2]['recall']:.3f} — it sits "
                       f"between the other two, so it loses points in both directions"),
        practice.Check("ANSWER by precision: the same class, but for a different reason",
                       result["worst_precision"] == 1,
                       f"class 1 precision {stats[1]['precision']:.3f}: when the model says "
                       f"1 it is wrong most often, because the neighbours of 1 are the "
                       f"classes that get mistaken *for* it"),
        practice.Check("…and the confusion is asymmetric — 0 and 2 are never confused",
                       matrix[0][2] + matrix[2][0] < matrix[0][1] + matrix[1][0],
                       f"0↔2 errors: {matrix[0][2] + matrix[2][0]}; 0↔1: "
                       f"{matrix[0][1] + matrix[1][0]}; 1↔2: "
                       f"{matrix[1][2] + matrix[2][1]}. A single accuracy number hides "
                       f"this entirely — the matrix is the point of the exercise"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
