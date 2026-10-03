"""Exercise 3 — cosine annealing for the shrinkage of Lesson 11's gradient boosting.

    Add a learning rate scheduler (cosine annealing) to the gradient boosting
    implementation from Lesson 11. Does it help compared to a fixed learning
    rate?

Reading of the exercise: Lesson 11's gradient-boosting loop (its
`SimpleRegressionTree`, depth 3, 100 trees, initial prediction the mean) is run
with tree t shrunk by the doc's cosine formula lr * 0.5 * (1 + cos(pi t / T))
instead of a constant lr, on Lesson 11's own 320 training rows. "Does it help"
is answered at the lesson's default lr 0.1 and at an aggressive 0.5, on a fresh
400-row test draw (the lesson's 80-row split is too noisy for 5% effects). A
third run at the fixed rate with the same total shrinkage separates what the
schedule's shape does from what its smaller sum does.

**ANSWER: no.** At the lesson's lr 0.1, cosine raises test MSE by 46% (0.3941
against 0.2693). At lr 0.5 it lowers it by 4% (0.3049 against 0.3168).

**FINDING: in boosting, cosine annealing is a halved learning rate.** The cosine
shrinkages sum to lr (T+1)/2, so at lr 0.5 the schedule spends what a fixed
0.2525 spends -- and the two reach train MSE 0.02015 and 0.02011, 0.2% apart. At
lr 0.1 that halving under-fits: train MSE 0.1656 against 0.0623.

**FINDING: the fixed halved rate beats the schedule.** At lr 0.5 a constant
0.2525 reaches test MSE 0.2614, against the schedule's 0.3049. Same total step;
the schedule only moves it onto the first trees. Whatever cosine annealing buys
in neural-network training, here it is a learning-rate change in disguise, and a
worse one than the plain change.

**CONTROL:** at a constant 0.1 the loop reproduces Lesson 11's
`GradientBoostingScratch(n_estimators=10)` exactly (0.0e+00).
"""

from __future__ import annotations

import math

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "11-ensemble-methods"  # the exercise names Lesson 11
TREES = 100  # GradientBoostingScratch's default


def cosine(lr, T=TREES):
    """lr * 0.5 * (1 + cos(pi t / T)) for t = 0..T-1, the doc's formula."""
    return [lr * 0.5 * (1 + math.cos(math.pi * t / T)) for t in range(T)]


def boost(ref, data, rates):
    """Lesson 11's gradient boosting loop with tree t shrunk by rates[t] instead of a
    constant: train MSE, test MSE and the test predictions."""
    X, y, X_te, y_te = data
    fit, test = np.full(len(y), np.mean(y)), np.full(len(y_te), np.mean(y))
    for rate in rates:
        tree = ref.SimpleRegressionTree(max_depth=3)  # GradientBoostingScratch's default
        tree.fit(X, y - fit)
        fit, test = fit + rate * tree.predict(X), test + rate * tree.predict(X_te)
    return float(np.mean((fit - y) ** 2)), float(np.mean((test - y_te) ** 2)), test


def solve():
    ref = parity.load_reference(PHASE, LESSON, "ensembles")
    X, y = ref.make_regression_data(n_samples=400)
    X_tr, _, y_tr, _ = ref.train_test_split(X, y)  # the lesson's own 320 training rows
    X_te, y_te = ref.make_regression_data(n_samples=400, seed=7)
    data = (X_tr, y_tr, X_te, y_te)
    runs = {}
    for lr in (0.1, 0.5):
        runs[lr] = {"fixed": boost(ref, data, [lr] * TREES)[:2],
                    "cosine": boost(ref, data, cosine(lr))[:2]}
    half = sum(cosine(0.5)) / TREES
    runs[0.5]["half"] = boost(ref, data, [half] * TREES)[:2]
    lesson = ref.GradientBoostingScratch(n_estimators=10, learning_rate=0.1)
    lesson.fit(X_tr, y_tr)
    ours = boost(ref, data, [0.1] * 10)[2]
    return {"runs": runs, "half_rate": half,
            "parity": float(np.abs(ours - lesson.predict(X_te)).max())}


def verify(result):
    (f1, c1), (f5, c5, h5) = (r.values() for r in result["runs"].values())
    return [
        practice.Check(
            "ANSWER: no -- it hurts at the lesson's lr 0.1 and barely helps at 0.5",
            c1[1] > 1.2 * f1[1] and c5[1] < f5[1],
            f"test MSE over 400 fresh rows, {TREES} trees: lr 0.1 fixed {f1[1]:.4f}, cosine "
            f"{c1[1]:.4f} ({c1[1] / f1[1] - 1:+.0%}); lr 0.5 fixed {f5[1]:.4f}, cosine "
            f"{c5[1]:.4f} ({c5[1] / f5[1] - 1:+.0%})",
        ),
        practice.Check(
            "FINDING: in boosting, cosine annealing is a halved learning rate",
            abs(c5[0] / h5[0] - 1) < 0.01,
            f"cosine shrinkage sums to lr (T+1)/2, so at lr 0.5 it spends what a fixed "
            f"{result['half_rate']:.4f} spends; their train MSEs are {c5[0]:.5f} and "
            f"{h5[0]:.5f}, {abs(c5[0] / h5[0] - 1):.1%} apart. At lr 0.1 the same halving "
            f"under-fits: train MSE {c1[0]:.4f} against {f1[0]:.4f} fixed",
        ),
        practice.Check(
            "FINDING: the fixed halved rate beats the schedule",
            h5[1] < 0.9 * c5[1],
            f"at lr 0.5, fixed {result['half_rate']:.4f} reaches test MSE {h5[1]:.4f} against "
            f"cosine's {c5[1]:.4f}: same total step, but front-loaded onto the first trees "
            "instead of spread evenly",
        ),
        practice.Check(
            "CONTROL: with a constant rate the loop is Lesson 11's GBM",
            result["parity"] < 1e-12,
            f"10 trees at a constant 0.1 match GradientBoostingScratch(n_estimators=10) within "
            f"{result['parity']:.1e}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
