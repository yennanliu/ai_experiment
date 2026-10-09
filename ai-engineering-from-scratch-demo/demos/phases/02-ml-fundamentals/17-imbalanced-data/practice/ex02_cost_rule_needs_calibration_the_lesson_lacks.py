"""Exercise 2 — the expected-cost rule needs calibrated probabilities; the lesson's are not.

    **Cost matrix optimization**: implement cost-sensitive learning where the
    cost matrix is a parameter. Create a function that takes a cost matrix and
    returns optimal predictions that minimize expected cost. Test with different
    cost ratios (1:10, 1:100, 1:1000) and plot how the precision-recall tradeoff
    changes.

Reading of the exercise: `cost[actual][predicted]` is a 2x2 matrix, and
`min_cost_predict` labels a row positive when its expected cost of predicting 1,
`(1-p) C[0][1] + p C[1][1]`, is below that of predicting 0. With a zero diagonal
that is the threshold `C_FP / (C_FP + C_FN)`: 0.0909, 0.0099 and 0.000999 for
1:10, 1:100 and 1:1000. Probabilities come from the lesson's
`logistic_regression_weighted` trained on the lesson's 800-row split, once with
the lesson's own settings (lr 0.1, 300 epochs) and once run to convergence (lr
1.0, 50,000 epochs); the test set is a fresh 10,000-row draw of
`make_imbalanced_data` (500 positives), so a 1:1000 cost is not decided by one
point. The "plot" is the printed table of precision and recall per ratio.

**ANSWER: precision collapses and recall saturates as the ratio grows.** On the
converged model, threshold 1/(1+r) gives precision 0.84, 0.60, 0.36 and recall
0.94, 1.00, 1.00 at 1:10, 1:100, 1:1000, at realised costs of 383, 428 and 875
against 260, 410 and 410 for the best threshold on the same scores.

**FINDING: on the lesson's own fit the rule flags almost everything.** Trained
the lesson's way the model is far from converged (w = [0.69, 0.64], b = -3.18,
against [3.40, 3.47], -13.30) and miscalibrated: its mean predicted probability
is 0.080 against a base rate of 0.050. The rule then flags 21.65%, 93.87% and
99.99% of rows, and its realised cost at 1:10 is 1665 against 263 for the best
threshold on the same scores -- 6.3x. Cost-optimal decisions need calibrated
probabilities, which the lesson never checks.

**FINDING: the lesson's threshold sweep cannot express these thresholds.**
`find_optimal_threshold` searches 0.05 to 0.95; the 1:100 and 1:1000 thresholds
lie below it. On the converged scores its best grid point costs 15,137 at 1:1000
against 410 reachable below the grid (37x), and 1637 against 410 at 1:100.

**CONTROL: at 1:1 the rule is the lesson's default threshold of 0.5**, and on
the converged model it costs 107 errors against a best of 104.

Structure: `min_cost_predict` is the requested function; `table` sweeps ratios.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "17-imbalanced-data"
RATIOS = (1, 10, 100, 1000)


def min_cost_predict(probs, cost):
    """1 where the expected cost of predicting positive is lower."""
    cost = np.asarray(cost, dtype=float)
    pos = (1 - probs) * cost[0][1] + probs * cost[1][1]
    neg = (1 - probs) * cost[0][0] + probs * cost[1][0]
    return (pos < neg).astype(int)


def realised(ref, y, pred, ratio):
    _, _, fp, fn = ref.confusion_matrix_values(y, pred)
    return fp + ratio * fn


def best_cost(ref, y, probs, ratio, grid):
    return min(realised(ref, y, (probs >= t).astype(int), ratio) for t in grid)


def table(ref, y, probs):
    rows = {}
    every = np.quantile(probs, np.linspace(0, 1, 2001))
    sweep = np.arange(0.05, 0.96, 0.01)  # the grid find_optimal_threshold uses
    for r in RATIOS:
        pred = min_cost_predict(probs, [[0, 1], [r, 0]])
        m = ref.compute_metrics(y, pred)
        rows[r] = {"precision": m["precision"], "recall": m["recall"],
                   "flagged": float(pred.mean()), "cost": realised(ref, y, pred, r),
                   "best": best_cost(ref, y, probs, r, every),
                   "sweep": best_cost(ref, y, probs, r, sweep)}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "imbalanced")
    X, y = ref.make_imbalanced_data(950, 50, seed=42)
    Xte, yte = ref.make_imbalanced_data(9500, 500, seed=7)
    out = {}
    for name, lr, epochs in (("lesson", 0.1, 300), ("converged", 1.0, 50000)):
        w, b = ref.logistic_regression_weighted(X[:800], y[:800], np.ones(800), lr, epochs)
        probs = ref.sigmoid(Xte @ w + b)
        out[name] = {"w": w, "b": b, "mean_p": float(probs.mean()), "rows": table(ref, yte, probs)}
    out["base"] = float(yte.mean())
    return out


def verify(result):
    lesson, conv = result["lesson"]["rows"], result["converged"]["rows"]
    return [
        practice.Check(
            "ANSWER: precision collapses and recall saturates as the cost ratio grows",
            conv[10]["precision"] > conv[100]["precision"] > conv[1000]["precision"]
            and conv[100]["recall"] > 0.99,
            "converged model, threshold 1/(1+r): " + "; ".join(
                f"1:{r} P={conv[r]['precision']:.2f} R={conv[r]['recall']:.2f} "
                f"cost {conv[r]['cost']} (best {conv[r]['best']})"
                for r in RATIOS[1:])),
        practice.Check(
            "FINDING: on the lesson's own fit the cost rule flags almost everything",
            lesson[100]["flagged"] > 0.9 and lesson[10]["cost"] > 4 * lesson[10]["best"],
            f"lesson fit w={np.round(result['lesson']['w'], 2)}, b={result['lesson']['b']:.2f} "
            f"(converged w={np.round(result['converged']['w'], 2)}, "
            f"b={result['converged']['b']:.2f}); mean p {result['lesson']['mean_p']:.3f} vs "
            f"base rate {result['base']:.3f}; flags "
            + ", ".join(f"{lesson[r]['flagged']:.2%}" for r in RATIOS[1:])
            + f"; cost at 1:10 is {lesson[10]['cost']} against a best of {lesson[10]['best']}"),
        practice.Check(
            "FINDING: the lesson's 0.05-0.95 sweep cannot reach the cost-optimal threshold",
            conv[1000]["sweep"] > 5 * conv[1000]["best"],
            f"converged scores at 1:1000: best grid point costs {conv[1000]['sweep']}, best "
            f"threshold anywhere {conv[1000]['best']}; at 1:100 {conv[100]['sweep']} vs "
            f"{conv[100]['best']}"),
        practice.Check(
            "CONTROL: at 1:1 the rule is the default 0.5 threshold and near-optimal",
            conv[1]["cost"] <= 1.1 * conv[1]["best"],
            f"1:1 costs {conv[1]['cost']} errors against a best of {conv[1]['best']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
