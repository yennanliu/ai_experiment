"""Exercise 5 — the winner flips on one irrelevant column, and LOF ranks a fraud ring below chance.

    **Real-world evaluation.** Take a dataset with known anomalies (credit card
    fraud from Kaggle, for example). Evaluate all four methods using
    precision@100, precision@500, and AUPRC. Which method works best? Why?

Reading of the exercise: the Kaggle file needs a download and a login, so this
ships the scaled-down runnable D11 requires: a fixture shaped like that data.
It has 20,000 rows at Kaggle's 0.172% fraud rate (34 frauds), 28
PCA-like components, and a log-normal Amount column that is identical for both
classes. Frauds are shifted by -4 in 5 components. Three variants change only
the nuisance structure: Gaussian components without Amount, the same with
Amount, and heavy-tailed t(3) components with Amount, closest to the real
file. The four methods are the lesson's `zscore_detect`, `iqr_detect` and
`IsolationForest`, plus sklearn's LOF (k=20). The lesson's fitted trees are
walked vectorised for speed.

**ANSWER: no single method wins, so "which works best?" depends on the
nuisance features.** With heavy tails and Amount, Isolation Forest is best but
still weak: AUPRC 0.127 against at most 0.012 for the rest.

**FINDING: one irrelevant column decides the winner.** With Gaussian
components the z-score and IQR reach AUPRC 0.965 and 0.971, against 0.560 for
Isolation Forest. Adding only the Amount column, with the same distribution in
both classes, drops them to 0.114 and 0.015, and LOF (0.487) and Isolation
Forest (0.349) move ahead. `zscore_detect` and `iqr_detect` score a row by its
worst single feature, so one skewed column outvotes 5 informative ones.

**FINDING: LOF ranks a fraud ring below chance.** In the Gaussian variant the
34 frauds form their own tight cluster, and with k=20 < 34 each fraud's
neighbours are other frauds: AUROC 0.336, AUPRC 0.001. The same LOF reaches
AUPRC 0.456 at k=50 and 0.998 at k=100.

**FINDING: the doc's "AUROC is misleading" holds twice.** In the heavy-tailed
variant Isolation Forest has AUROC 0.967 and AUPRC 0.127. With Amount, the
z-score has AUROC 0.994 and AUPRC 0.114.

**CONTROL:** the vectorised walk reproduces the lesson's `anomaly_score`
exactly (difference 0.0 on every 97th row). P@k cannot exceed 34/k, so at
this scale P@100 is capped at 0.34 (z and IQR hit it in the Gaussian variant)
and P@500 at 0.068. On Kaggle's 492 frauds the caps are 1.0 and 0.984.

Structure: `make_fixture` builds a variant; `forest_scores` walks the lesson's
fitted trees; `evaluate` returns P@100, P@500 and AUPRC per method.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "16-anomaly-detection"
ROWS, RATE, DIMS, SHIFTED, SEED = 20_000, 0.00172, 28, 5, 10
VARIANTS = {"gaussian": (None, False), "amount": (None, True), "heavy": (3, True)}


def make_fixture(df, amount):
    """Labelled fraud-shaped rows: frauds are shifted -4 in the first SHIFTED components."""
    rng = np.random.RandomState(SEED)
    frauds = int(round(ROWS * RATE))
    draw = (lambda n: rng.standard_t(df, (n, DIMS))) if df else (lambda n: rng.randn(n, DIMS))
    X = np.vstack([draw(ROWS - frauds), draw(frauds) - 4.0 * (np.arange(DIMS) < SHIFTED)])
    if amount:
        X = np.column_stack([X, rng.lognormal(3, 1.5, ROWS)])
    return X, np.r_[np.zeros(ROWS - frauds), np.ones(frauds)].astype(int)


def forest_scores(ref, forest, X):
    """The lesson's anomaly_score, walking each fitted tree with index arrays."""
    total = np.zeros(len(X))
    for tree in forest.trees:
        stack = [(tree, np.arange(len(X)), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if node.is_leaf:
                total[idx] += depth + ref._c_factor(node.size)
                continue
            left = X[idx, node.feature] < node.threshold
            stack += [(node.left, idx[left], depth + 1), (node.right, idx[~left], depth + 1)]
    return 2.0 ** (-total / len(forest.trees) / ref._c_factor(min(forest.max_samples, len(X))))


def evaluate(ref, X, y):
    forest = ref.IsolationForest(100, 256, seed=42).fit(X)
    scores = {"z": ref.zscore_detect(X)[1], "iqr": ref.iqr_detect(X)[1],
              "iforest": forest_scores(ref, forest, X),
              "lof": -LocalOutlierFactor(20).fit(X).negative_outlier_factor_}
    sub = np.arange(0, len(X), 97)
    gap = float(np.abs(forest.anomaly_score(X[sub]) - scores["iforest"][sub]).max())
    table = {name: (ref.precision_at_k(y, s, 100), ref.precision_at_k(y, s, 500),
                    average_precision_score(y, s), roc_auc_score(y, s))
             for name, s in scores.items()}
    return table, gap


def solve():
    ref = parity.load_reference(PHASE, LESSON, "anomaly_detection")
    out = {name: evaluate(ref, *make_fixture(*spec)) for name, spec in VARIANTS.items()}
    X, y = make_fixture(None, False)
    ring = {k: average_precision_score(y, -LocalOutlierFactor(k).fit(X).negative_outlier_factor_)
            for k in (50, 100)}
    return {"tables": {n: t for n, (t, _) in out.items()}, "gap": max(g for _, g in out.values()),
            "ring": ring, "frauds": int(y.sum())}


def summary(tables):
    """The AUPRC leader per variant, and the heavy-tailed row as text."""
    best = {n: max(t, key=lambda m: t[m][2]) for n, t in tables.items()}
    row = " | ".join(f"{m} {p1:.2f}/{p5:.3f}/{ap:.3f}"
                     for m, (p1, p5, ap, _) in tables["heavy"].items())
    p500 = max(t[1] for t in tables["gaussian"].values())
    return best, row, p500


def verify(result):
    g, a, h = (result["tables"][n] for n in VARIANTS)
    best, row, p500 = summary(result["tables"])
    return [
        practice.Check(
            "ANSWER: on the Kaggle-like variant Isolation Forest wins, weakly",
            best["heavy"] == "iforest" and h["iforest"][2] < 0.3,
            f"heavy tails + Amount, P@100/P@500/AUPRC: {row}",
        ),
        practice.Check(
            "FINDING: one irrelevant Amount column flips the winner",
            best["gaussian"] in ("z", "iqr") and best["amount"] not in ("z", "iqr")
            and a["z"][2] < g["z"][2] / 4,
            f"AUPRC z/iqr/iforest {g['z'][2]:.3f}/{g['iqr'][2]:.3f}/{g['iforest'][2]:.3f} "
            f"without Amount, {a['z'][2]:.3f}/{a['iqr'][2]:.3f}/{a['iforest'][2]:.3f} with it, "
            f"where {best['amount']} leads at {a[best['amount']][2]:.3f}",
        ),
        practice.Check(
            "FINDING: LOF with k=20 ranks a 34-fraud ring below chance",
            g["lof"][3] < 0.5 and result["ring"][100] > 0.9,
            f"k=20 AUROC {g['lof'][3]:.3f}, AUPRC {g['lof'][2]:.3f}; AUPRC "
            f"{result['ring'][50]:.3f} at k=50 and {result['ring'][100]:.3f} at k=100",
        ),
        practice.Check(
            "FINDING: AUROC looks fine where AUPRC does not",
            min(h["iforest"][3], a["z"][3]) > 0.9 > 0.2 > max(h["iforest"][2], a["z"][2]),
            f"Isolation Forest, heavy tails: AUROC {h['iforest'][3]:.3f}, AUPRC "
            f"{h['iforest'][2]:.3f}; z-score with Amount: AUROC {a['z'][3]:.3f}, AUPRC "
            f"{a['z'][2]:.3f}",
        ),
        practice.Check(
            "CONTROL: the vectorised walk is the lesson's score; P@500 is capped",
            result["gap"] < 1e-12 and p500 <= result["frauds"] / 500,
            f"max difference {result['gap']:.1e}; P@500 <= {result['frauds']}/500 = "
            f"{result['frauds'] / 500:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
