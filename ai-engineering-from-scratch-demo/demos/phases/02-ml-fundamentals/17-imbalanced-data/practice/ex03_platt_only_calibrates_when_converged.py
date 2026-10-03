"""Exercise 3 — Platt scaling keeps the ranking, but only a converged fit calibrates.

    **Threshold calibration**: implement Platt scaling (fit a logistic
    regression on the model's raw outputs to produce calibrated probabilities).
    Compare the precision-recall curve before and after calibration. Show that
    calibration does not change the ranking (AUC stays the same) but makes the
    probabilities more meaningful.

Reading of the exercise: the model is the lesson's own class-weighted logistic
regression from its threshold-tuning step (`compute_class_weights`, lr 0.1, 300
epochs, trained on the first 600 rows of the lesson's training split), the one
whose probabilities class weights deliberately distort. Platt scaling fits
`sigmoid(a z + c)` to its raw logits `z` on the lesson's 200-row validation
split, using the lesson's own `logistic_regression_weighted` as the fitter.
Everything is scored on a fresh 10,000-row draw of `make_imbalanced_data`.
"More meaningful" is measured as expected calibration error (10 bins), Brier
score, and the mean probability against the base rate.

**ANSWER: the ranking is untouched and the probabilities become honest.** Test
AUC is 0.99808 before and after, average precision 0.9660 before and after, and
the score order is identical row for row, so the PR curve is the same set of
points. Calibration error falls from 0.127 to 0.009, Brier from 0.0427 to
0.0101, and the mean probability from 0.175 to 0.059 against a base rate of
0.050.

**FINDING: Platt with the lesson's default fitter settings barely calibrates.**
`logistic_regression_weighted` defaults to lr 0.01 and 200 epochs; on the 1-D
logits that stops at slope a = 0.79 and calibration error 0.118, against a = 5.70
and 0.009 when run to convergence (lr 0.5, 5000 epochs). Platt scaling is a
maximum-likelihood fit, and an unconverged fit is not Platt scaling.

**FINDING: the threshold the lesson tunes is set by two validation points.**
The lesson's 200-row validation split holds 10 positives, and the model
separates them perfectly: the top negative scores 0.667 and the bottom positive
0.706, so validation F1 is 1.0 on every grid point between (0.67-0.70; 0.24-0.45
after calibration stretches the gap). `find_optimal_threshold` keeps the first
strict maximum, so it returns the low edge both times, hugging the highest
negative. On the test set that threshold scores F1 0.858, against 0.876 at the
gap's high edge and 0.896 at the best test threshold.

**CONTROL: ranking is preserved only because the slope is positive.** The same
scores pushed through a negative slope give AUC 0.0019 = 1 - 0.99808.

Structure: `platt` fits the scaler; `calibration` measures it; `plateau` finds
the validation F1 ties and the score gap that creates them.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "17-imbalanced-data"
GRID = np.arange(0.05, 0.96, 0.01)  # find_optimal_threshold's sweep


def auc(y, s):
    neg = np.sort(s[y == 0])
    lo, hi = np.searchsorted(neg, s[y == 1], "left"), np.searchsorted(neg, s[y == 1], "right")
    return float(np.mean(lo + 0.5 * (hi - lo)) / len(neg))


def average_precision(y, s):
    hits = y[np.argsort(-s, kind="mergesort")]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(np.sum(precision * hits) / hits.sum())


def ece(y, p, bins=10):
    which = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[which == k].sum() - y[which == k].sum()) for k in range(bins)) / len(y))


def platt(ref, z, y, lr, epochs):
    """(slope, intercept) of a logistic fit on the raw logits."""
    a, c = ref.logistic_regression_weighted(z[:, None], y, np.ones(len(y)), lr=lr, epochs=epochs)
    return float(a[0]), float(c)


def calibration(y, p):
    return {"ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)), "mean": float(p.mean()),
            "auc": auc(y, p), "ap": average_precision(y, p)}


def f1_at(ref, y, p, t):
    return ref.compute_metrics(y, (p >= t).astype(int))["f1"]


def plateau(ref, y, p):
    """Validation F1 ties across the sweep, and the gap that creates them."""
    f1s = [f1_at(ref, y, p, t) for t in GRID]
    ties = [t for t, f in zip(GRID, f1s) if f == max(f1s)]
    return {"best": max(f1s), "ties": len(ties), "lo": float(min(ties)), "hi": float(max(ties)),
            "chosen": float(ref.find_optimal_threshold(y, p)[0]), "positives": int(y.sum()),
            "gap": (float(p[y == 0].max()), float(p[y == 1].min()))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "imbalanced")
    X, y = ref.make_imbalanced_data(950, 50, seed=42)
    Xte, yte = ref.make_imbalanced_data(9500, 500, seed=7)
    Xtr, ytr, Xva, yva = X[:600], y[:600], X[600:800], y[600:800]
    w, b = ref.logistic_regression_weighted(Xtr, ytr, ref.compute_class_weights(ytr), 0.1, 300)
    zva, zte = Xva @ w + b, Xte @ w + b
    a, c = platt(ref, zva, yva, 0.5, 5000)
    a_lesson, c_lesson = platt(ref, zva, yva, 0.01, 200)
    raw, cal = ref.sigmoid(zte), ref.sigmoid(a * zte + c)
    tuned = plateau(ref, yva, ref.sigmoid(zva))
    return {
        "raw": calibration(yte, raw), "cal": calibration(yte, cal),
        "lesson": calibration(yte, ref.sigmoid(a_lesson * zte + c_lesson)),
        "slopes": (a, a_lesson), "base": float(yte.mean()),
        "same_order": bool(np.array_equal(np.argsort(raw, kind="stable"),
                                          np.argsort(cal, kind="stable"))),
        "flipped": auc(yte, ref.sigmoid(-a * zte + c)),
        "plateau_raw": tuned,
        "test_f1": (f1_at(ref, yte, raw, tuned["chosen"]), f1_at(ref, yte, raw, tuned["hi"]),
                    max(f1_at(ref, yte, raw, t) for t in np.quantile(raw, np.linspace(0, 1, 801)))),
        "plateau_cal": plateau(ref, yva, ref.sigmoid(a * zva + c)),
    }


def verify(result):
    raw, cal, les = result["raw"], result["cal"], result["lesson"]
    pr, pc, tf = result["plateau_raw"], result["plateau_cal"], result["test_f1"]
    return [
        practice.Check(
            "ANSWER: AUC and the PR curve are unchanged; the probabilities become honest",
            raw["auc"] == cal["auc"] and result["same_order"] and cal["ece"] < raw["ece"] / 5,
            f"AUC {raw['auc']:.5f} -> {cal['auc']:.5f}, AP {raw['ap']:.4f} -> {cal['ap']:.4f}, "
            f"identical row order; ECE {raw['ece']:.3f} -> {cal['ece']:.3f}, Brier "
            f"{raw['brier']:.4f} -> {cal['brier']:.4f}, mean p {raw['mean']:.3f} -> "
            f"{cal['mean']:.3f} against a base rate of {result['base']:.3f}"),
        practice.Check(
            "FINDING: Platt with the lesson's default fitter settings barely calibrates",
            les["ece"] > 5 * cal["ece"],
            f"lr 0.01 / 200 epochs stops at slope {result['slopes'][1]:.2f} and ECE "
            f"{les['ece']:.3f}; converged, slope {result['slopes'][0]:.2f} and ECE "
            f"{cal['ece']:.3f}"),
        practice.Check(
            "FINDING: the lesson's tuned threshold is set by two validation points",
            pr["best"] == 1.0 and pr["chosen"] == pr["lo"]
            and pc["chosen"] == pc["lo"] and tf[0] < tf[2] - 0.02,
            f"{pr['positives']} validation positives separate perfectly: the top negative scores "
            f"{pr['gap'][0]:.3f}, the bottom positive {pr['gap'][1]:.3f}, so F1 = 1.0 on "
            f"{pr['ties']} grid points ({pr['lo']:.2f}-{pr['hi']:.2f}; {pc['ties']} after "
            f"calibration, {pc['lo']:.2f}-{pc['hi']:.2f}) and the sweep returns the low edge "
            f"both times. Test F1 there is {tf[0]:.3f}, {tf[1]:.3f} at the high edge, "
            f"{tf[2]:.3f} at the best test threshold"),
        practice.Check(
            "CONTROL: ranking survives only because the Platt slope is positive",
            result["slopes"][0] > 0 and abs(result["flipped"] - (1 - raw["auc"])) < 1e-9,
            f"slope {result['slopes'][0]:.2f}; with the slope negated AUC is "
            f"{result['flipped']:.4f} = 1 - {raw['auc']:.5f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
