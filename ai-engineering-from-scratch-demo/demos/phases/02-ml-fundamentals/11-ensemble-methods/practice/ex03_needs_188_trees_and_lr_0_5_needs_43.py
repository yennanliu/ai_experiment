"""Exercise 3 — early stopping for the lesson's gradient boosting.

    In the gradient boosting implementation, add early stopping: track
    validation loss after each round and stop when it has not improved for 10
    consecutive rounds. How many trees does it actually need?

Reading of the exercise: the lesson's demo data (`make_regression_data(400)`) is
split by the lesson's own `train_test_split` into 240 fit / 80 validation / 80
test rows, so the stopping rule never sees the test set. The lesson's
`GradientBoostingScratch` is grown one `SimpleRegressionTree` at a time at its
default learning rate 0.1 and depth 3, and stops after 10 rounds with no new
validation minimum. "Needs" is the round of that minimum. The same run is
continued to patience 20 to see what the rule left on the table.

**ANSWER: 188 trees.** Validation MSE bottoms out at 0.4131 on round 188 and the
rule stops at round 198, keeping round 188, whose test MSE is 0.4210.

**FINDING: patience 10 stops inside a plateau, and it does not matter.** The
curve has a 13-round run without a new minimum; patience 20 rides through it to
a minimum at round 232, 1.4% lower on validation (0.4071) but level on test
(0.4205 against 0.4210). Those 44 trees fit the validation set's noise.

**FINDING: the lesson's default of 100 trees is half what it needs.** At 100
trees validation MSE is 0.4896 and test MSE 0.4475, 6% worse than stopping.

**FINDING: lr=0.5 needs 43 trees and reaches the same test MSE.** It stops at
round 53 with test MSE 0.4208 against 0.4210 at lr=0.1, with 4.4x fewer trees.
The lesson's demo prints "Lower learning rates need more trees but often
generalize better"; once each rate gets its own stopping point, the second half
of that does not show up here.

**CONTROL:** the first 20 trees grown here predict exactly what the lesson's
`GradientBoostingScratch(n_estimators=20)` predicts (max difference 0.0).
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "11-ensemble-methods"
PATIENCE, LONG_PATIENCE, CHECK_TREES = 10, 20, 20


def boost(ref, data, lr, patience):
    """Grow the lesson's GBM a tree at a time; stop after `patience` rounds without a
    new validation minimum. Returns the model and its validation/test MSE curves."""
    X, y, X_val, y_val, X_te, y_te = data
    model = ref.GradientBoostingScratch(n_estimators=0, learning_rate=lr)
    model.initial_pred = np.mean(y)
    fit, val, te = (np.full(len(t), model.initial_pred) for t in (y, y_val, y_te))
    val_curve, test_curve, since = [], [], 0
    while since < patience:
        tree = ref.SimpleRegressionTree(max_depth=model.max_depth)
        tree.fit(X, y - fit)
        fit, val, te = (p + lr * tree.predict(Z) for p, Z in ((fit, X), (val, X_val), (te, X_te)))
        model.trees.append(tree)
        val_curve.append(np.mean((val - y_val) ** 2))
        test_curve.append(np.mean((te - y_te) ** 2))
        since = 0 if val_curve[-1] == min(val_curve) else since + 1
    return model, np.array(val_curve), np.array(test_curve)


def stop_point(val_curve, patience):
    """(best round, round at which training stops) for a given patience, 1-based."""
    best, since = 0, 0
    for i, v in enumerate(val_curve):
        best, since = (i, 0) if v < val_curve[best] or i == 0 else (best, since + 1)
        if since == patience:
            return best + 1, i + 1
    return best + 1, len(val_curve)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "ensembles")
    X, y = ref.make_regression_data(n_samples=400)
    X_tr, X_te, y_tr, y_te = ref.train_test_split(X, y)
    X_fit, X_val, y_fit, y_val = ref.train_test_split(X_tr, y_tr, test_ratio=0.25, seed=0)
    data = (X_fit, y_fit, X_val, y_val, X_te, y_te)
    model, val, test = boost(ref, data, 0.1, LONG_PATIENCE)
    _, val_fast, test_fast = boost(ref, data, 0.5, PATIENCE)
    lesson = ref.GradientBoostingScratch(n_estimators=CHECK_TREES, learning_rate=0.1)
    lesson.fit(X_fit, y_fit)
    model.trees = model.trees[:CHECK_TREES]
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "ensembles.py").read_text()
    return {
        "val": val, "test": test, "stop": stop_point(val, PATIENCE),
        "stop_long": stop_point(val, LONG_PATIENCE),
        "fast": (stop_point(val_fast, PATIENCE), test_fast),
        "plateau": max(np.diff(np.flatnonzero(val <= np.minimum.accumulate(val)))) - 1,
        "prefix_gap": float(np.abs(model.predict(X_te) - lesson.predict(X_te)).max()),
        "claims_lower_lr": "Lower learning rates need more trees but often generalize better"
        in source,
    }


def verify(result):
    val, test = result["val"], result["test"]
    (best, stop), (best_long, _) = result["stop"], result["stop_long"]
    (fast_best, fast_stop), test_fast = result["fast"]
    return [
        practice.Check(
            f"ANSWER: {best} trees at the lesson's lr=0.1",
            150 <= best <= 220 and stop == best + PATIENCE,
            f"validation MSE bottoms out at {val[best - 1]:.4f} on round {best}; the rule stops "
            f"at round {stop} and keeps round {best}, whose test MSE is {test[best - 1]:.4f}",
        ),
        practice.Check(
            "FINDING: patience 10 stops inside a plateau, and it does not matter",
            result["plateau"] > PATIENCE and best_long > best
            and abs(test[best_long - 1] - test[best - 1]) < 0.01,
            f"the curve has a {result['plateau']}-round run with no new minimum; patience "
            f"{LONG_PATIENCE} rides through it to round {best_long}, validation "
            f"{val[best_long - 1]:.4f} ({1 - val[best_long - 1] / val[best - 1]:.1%} lower) but "
            f"test {test[best_long - 1]:.4f} against {test[best - 1]:.4f}",
        ),
        practice.Check(
            "FINDING: the lesson's default of 100 trees is half what it needs",
            test[99] > 1.03 * test[best - 1],
            f"at 100 trees validation MSE is {val[99]:.4f} and test MSE {test[99]:.4f}, "
            f"{test[99] / test[best - 1] - 1:.0%} worse than stopping at {best}",
        ),
        practice.Check(
            f"FINDING: lr=0.5 needs {fast_best} trees and generalises as well",
            fast_best < best / 3 and abs(test_fast[fast_best - 1] - test[best - 1]) < 0.01
            and result["claims_lower_lr"],
            f"it stops at round {fast_stop} with test MSE {test_fast[fast_best - 1]:.4f} against "
            f"{test[best - 1]:.4f}, {best / fast_best:.1f}x fewer trees; the lesson prints 'Lower "
            "learning rates need more trees but often generalize better'",
        ),
        practice.Check(
            "CONTROL: the tree-at-a-time model is the lesson's model",
            result["prefix_gap"] == 0,
            f"its first {CHECK_TREES} trees predict what GradientBoostingScratch("
            f"n_estimators={CHECK_TREES}) predicts, max difference {result['prefix_gap']:.1e}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
