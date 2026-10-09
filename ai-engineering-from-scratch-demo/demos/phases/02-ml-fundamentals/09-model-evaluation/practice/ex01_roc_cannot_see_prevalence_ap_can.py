"""Exercise 1 — ROC AUC cannot see the class ratio; average precision falls with it.

    Implement precision-recall curves: plot precision vs recall at different
    thresholds. Compute the average precision (area under the PR curve). Compare
    the PR curve to the ROC curve on an imbalanced dataset and explain when each
    is more informative.

Reading of the exercise: the PR curve sweeps every distinct score as a
threshold, scoring each with the lesson's own `precision` and `recall`; average
precision is the step sum `sum (R_k - R_{k-1}) P_k`. The scorer is the lesson's
`SimpleLogistic`, trained as in its `__main__`. To compare the curves on
imbalanced data with the ranking held fixed, the negatives of a balanced pool
from the lesson's `make_classification_data` are replicated 1x, 9x and 19x:
each class's score distribution is unchanged, only the ratio moves.

| positives | AUC (lesson's `auc_roc`) | AP | precision at 80% recall | FPR at 80% recall |
|---|---:|---:|---:|---:|
| 48% | 0.9682 | 0.967 | 0.91 | 0.048 |
| 9% | 0.9682 | 0.846 | 0.53 | 0.048 |
| 5% | 0.9682 | 0.787 | 0.35 | 0.048 |

**ANSWER: with the ranking held fixed, AUC does not move at all while AP falls
0.967 -> 0.787.** Replicating negatives leaves TPR and FPR unchanged, so the ROC
curve is literally the same curve; precision divides by everything flagged, so
the PR curve sags. ROC is more informative when the question is ranking quality
or the deployed prevalence is unknown; PR is more informative when positives are
rare and what matters is how many alerts are real.

**FINDING: the ROC operating point looks identical while nearly two alarms in
three turn false.** At 80% recall the false-positive rate is 0.048 at every
ratio, but precision falls 0.91 -> 0.35 at 5% positives: FPR divides by the
negatives, so 19x more of them is invisible to it.

**FINDING: the lesson's own imbalanced data cannot show the difference.**
`make_imbalanced_data` puts the minority at (3, 3) against N(0, 1), nearly
separable, and the lesson's split leaves 3 positives in 60 test rows: AUC 1.0000
and AP 1.0000. Both curves are perfect, so the comparison needs overlapping
classes.

**CONTROL:** the lesson's `auc_roc` equals the Mann-Whitney probability
P(score_pos > score_neg) to 1e-12 (0.968159 both), so its AUC is ranking quality
as the doc says.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "09-model-evaluation"
COPIES = (1, 9, 19)


def pr_curve(ref, y, scores):
    """(recall, precision) at every distinct threshold, highest first."""
    points = []
    for t in sorted(set(scores), reverse=True):
        pred = [1 if s >= t else 0 for s in scores]
        points.append((ref.recall(y, pred), ref.precision(y, pred)))
    return points


def average_precision(points):
    area, last = 0.0, 0.0
    for r, p in points:
        area, last = area + (r - last) * p, r
    return area


def at_recall(ref, y, scores, target=0.8):
    """Precision and false-positive rate at the first threshold reaching `target` recall."""
    fpr, tpr, _ = ref.roc_curve(y, scores)
    k = next(i for i, r in enumerate(tpr) if r >= target)
    return dict(pr_curve(ref, y, scores))[tpr[k]], fpr[k]


def mann_whitney(y, scores):
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def compare(ref, model, copies):
    X, y = ref.make_classification_data(240, seed=11)
    scores = [model.predict_proba(x) for x in X]
    y_rep = y + [0] * (copies - 1) * y.count(0)
    s_rep = scores + [s for s, t in zip(scores, y) if t == 0] * (copies - 1)
    precision, fpr = at_recall(ref, y_rep, s_rep)
    return {"prevalence": sum(y_rep) / len(y_rep), "auc": ref.auc_roc(y_rep, s_rep),
            "ap": average_precision(pr_curve(ref, y_rep, s_rep)),
            "p_at_r80": precision, "fpr_at_r80": fpr, "mw": mann_whitney(y, scores)}


def lesson_split(ref):
    """The lesson's own imbalanced run: its split, its model, its test set."""
    X, y = ref.make_imbalanced_data(300, minority_ratio=0.05)
    X_tr, y_tr, _, _, X_te, y_te = ref.train_val_test_split(X, y)
    model = ref.SimpleLogistic(lr=0.5, epochs=500)
    model.fit(X_tr, y_tr)
    scores = [model.predict_proba(x) for x in X_te]
    return sum(y_te), len(y_te), ref.auc_roc(y_te, scores), average_precision(
        pr_curve(ref, y_te, scores))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "evaluation")
    X, y = ref.make_classification_data(300)
    model = ref.SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X, y)
    return {"runs": {c: compare(ref, model, c) for c in COPIES}, "lesson": lesson_split(ref)}


def verify(result):
    runs, (pos, n, l_auc, l_ap) = result["runs"], result["lesson"]
    base, rare = runs[1], runs[19]
    rows = "; ".join(f"{r['prevalence']:.0%} positive: AUC {r['auc']:.4f}, AP {r['ap']:.3f}, "
                     f"P@R0.8 {r['p_at_r80']:.2f}" for r in runs.values())
    return [
        practice.Check(
            "ANSWER: at the same ranking, AUC is unchanged and AP falls with prevalence",
            all((abs(rare["auc"] - base["auc"]) < 1e-12, base["ap"] - rare["ap"] > 0.15)),
            f"{rows}. Replicating negatives moves TPR and FPR not at all, so the ROC curve "
            "is the same curve; precision divides by everything flagged, so it moves",
        ),
        practice.Check(
            "FINDING: ROC's operating point looks identical while two alarms in three turn false",
            all((rare["fpr_at_r80"] == base["fpr_at_r80"], rare["p_at_r80"] < 0.4)),
            f"at 80% recall the false-positive rate is {base['fpr_at_r80']:.3f} at every ratio, "
            f"but precision is {base['p_at_r80']:.2f} at {base['prevalence']:.0%} positive and "
            f"{rare['p_at_r80']:.2f} at {rare['prevalence']:.0%}: FPR divides by the negatives, "
            "so it cannot see that there are 19x more of them",
        ),
        practice.Check(
            "FINDING: the lesson's own imbalanced data cannot show the difference",
            all((pos <= 5, l_auc > 0.99, l_ap > 0.99)),
            f"make_imbalanced_data puts the minority at (3, 3) against N(0, 1), nearly "
            f"separable, and the lesson's split leaves {pos} positives in {n} test rows: "
            f"AUC {l_auc:.4f}, AP {l_ap:.4f}. Both curves are perfect, so neither says anything",
        ),
        practice.Check(
            "CONTROL: the lesson's auc_roc is the Mann-Whitney ranking probability",
            abs(base["auc"] - base["mw"]) < 1e-12,
            f"auc_roc {base['auc']:.6f} vs P(score_pos > score_neg) {base['mw']:.6f} over every "
            "pair, so 'AUC is threshold-independent ranking quality' holds for this code",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
