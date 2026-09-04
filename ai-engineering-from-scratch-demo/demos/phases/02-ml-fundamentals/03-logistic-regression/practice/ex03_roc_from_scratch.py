"""Exercise 3 — an ROC curve over 100 thresholds, and AUC by the trapezoidal rule.

    Build an ROC curve from scratch. For 100 threshold values from 0 to 1,
    compute the true positive rate and false positive rate. Calculate the AUC
    (area under the curve) using the trapezoidal rule.

Reading of the exercise: a fixed grid of 100 thresholds does **not** give the
exact AUC — check 4 measures the 5e-05 gap against the rank-based (Mann-Whitney)
value, which is exact, and check 5 shows refinement closing it. The grid *does*
reach both corners unaided, because it spans 0 to 1 and every probability lies in
between; check 3 shows what a grid stopping at 0.05 and 0.95 loses instead.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "03-logistic-regression"
SEED, N_PER, EPOCHS, LR, N_THRESHOLDS = 42, 200, 2_000, 0.3, 100


def make_data(rng):
    X, y = [], []
    for label, centre in ((0, -1.0), (1, 1.0)):
        for _ in range(N_PER):
            X.append([rng.gauss(centre, 1.4), rng.gauss(0, 1.4)])
            y.append(label)
    return X, y


def roc(scores, labels, n_thresholds=N_THRESHOLDS, lo=0.0, hi=1.0):
    """FPR/TPR at each threshold in [lo, hi]. The exercise's grid spans 0 to 1,
    which already reaches both corners — no padding needed."""
    positives = sum(labels)
    negatives = len(labels) - positives
    points = []
    for i in range(n_thresholds):
        threshold = lo + (hi - lo) * i / (n_thresholds - 1)
        tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
        points.append((fp / negatives, tp / positives))
    return sorted(points)


def trapezoid(points):
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:]))


def exact_auc(scores, labels):
    """Mann-Whitney: P(score of a random positive > a random negative), ties at ½."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "logistic_regression")
    rng = random.Random(SEED)
    X, y = make_data(rng)
    model = ref.LogisticRegression(n_features=2, learning_rate=LR)
    model.fit(X, y, epochs=EPOCHS, print_every=EPOCHS + 1)
    scores = [model.predict_proba(x) for x in X]
    curve = roc(scores, y)
    narrow = roc(scores, y, lo=0.05, hi=0.95)
    exact = exact_auc(scores, y)
    fine = trapezoid(roc(scores, y, n_thresholds=10_000))
    return {"n_points": len(curve), "auc": trapezoid(curve),
            "corners": (curve[0], curve[-1]), "narrow_auc": trapezoid(narrow),
            "narrow_corners": (narrow[0], narrow[-1]),
            "exact": exact, "fine": fine,
            "monotone": all(a[1] <= b[1] + 1e-12 for a, b in zip(curve, curve[1:])),
            "accuracy": model.accuracy(X, y)}


def verify(result):
    auc, exact = result["auc"], result["exact"]
    return [
        practice.Check(f"{N_THRESHOLDS} thresholds give a monotone curve reaching both corners",
                       result["monotone"] and result["corners"] == ((0.0, 0.0), (1.0, 1.0)),
                       f"{result['n_points']} points from {result['corners'][0]} to "
                       f"{result['corners'][1]} — a grid spanning 0 to 1 hits the corners "
                       f"on its own, since every score is ≥ 0 and none is ≥ 1"),
        practice.Check("AUC is well above chance and consistent with accuracy",
                       0.75 < auc < 1.0,
                       f"AUC {auc:.4f} against an accuracy of {result['accuracy']:.1%} — "
                       f"AUC is threshold-free, accuracy is one threshold"),
        practice.Check("FINDING: a grid that stops short of 0 and 1 loses real area",
                       result["narrow_auc"] < auc - 0.01,
                       f"thresholds spanning only 0.05 to 0.95 reach "
                       f"{result['narrow_corners'][0]} and "
                       f"{result['narrow_corners'][1]}, giving AUC "
                       f"{result['narrow_auc']:.4f} against {auc:.4f} — short by "
                       f"{auc - result['narrow_auc']:.4f}. The corners are not decoration; "
                       f"they carry area"),
        practice.Check("FINDING: the 100-threshold grid is not the exact AUC",
                       abs(auc - exact) > 1e-6,
                       f"trapezoid over {N_THRESHOLDS} thresholds: {auc:.6f}; exact "
                       f"rank-based (Mann-Whitney) value: {exact:.6f} — off by "
                       f"{abs(auc - exact):.2e}. The recipe the exercise gives is an "
                       f"approximation of a quantity with a closed form"),
        practice.Check("…and refining the grid converges to the exact value",
                       abs(result["fine"] - exact) < abs(auc - exact),
                       f"10,000 thresholds give {result['fine']:.6f}, closer to "
                       f"{exact:.6f} than the 100-point {auc:.6f}. So the grid error is "
                       f"discretisation, not a different definition"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
