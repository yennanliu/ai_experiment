"""Exercise 1 — the borderline rule is vacuous under overlap; no SMOTE moves the ranking.

    **Borderline-SMOTE**: modify the SMOTE implementation to only generate
    synthetic samples for minority points that are near the decision boundary
    (those whose k-nearest neighbors include majority class samples). Compare
    results with standard SMOTE on a dataset where classes overlap.

Reading of the exercise: the dataset is the lesson's `make_imbalanced_data`
(950 majority at the origin, 50 minority with sd 0.8) with the minority centre
moved from (2.5, 2.5) to (1.5, 1.5) so the classes overlap; 800/200 split, 5
seeds. A minority point is "borderline" if any of its 5 nearest neighbours in the
whole training set is majority (the exercise's rule). Synthetic points start from
borderline points only and interpolate towards their 5 nearest minority
neighbours, using the lesson's `find_k_neighbors`; each method tops the minority
up to parity and trains the lesson's `logistic_regression_weighted` (lr 0.1, 300
epochs), scored at threshold 0.5 by the lesson's `compute_metrics`.

**ANSWER: Borderline-SMOTE does not beat SMOTE: F1 0.324 against 0.335.** Both
far beat no resampling (0.179), and both buy that with the same trade: recall
0.96, precision 0.20.

**FINDING: the three models rank the test set identically.** Mean test AUC is
0.9454 untreated, 0.9458 with SMOTE and 0.9452 with Borderline-SMOTE. With a
linear model in 2-D, oversampling moves the intercept (-2.92 to -1.95 on seed 0)
and barely turns the weight vector, so every F1 difference above is a different
threshold on the same score, not a better classifier.

**FINDING: the exercise's rule stops selecting anything once classes overlap
heavily.** With the minority centre at (1, 1), every training minority point
(40.4 per seed on average) has a majority neighbour, so Borderline-SMOTE
reproduces the lesson's `smote` exactly (bit-identical synthetic arrays). And
16.8 of those 40.4 have only majority neighbours -- the points Han et al. (2005)
call noise and exclude, which the exercise's rule keeps.

**CONTROL: on the lesson's own, well-separated data the rule is selective.** At
(2.5, 2.5) it keeps 14.0 of 40.4 points, and Borderline-SMOTE's F1 (0.460) is below
SMOTE's (0.552), itself below no resampling (0.818).

Structure: `borderline` counts majority neighbours; `bsmote` is the modified
sampler; `trial` runs one seed of every method.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "17-imbalanced-data"
K, SEEDS = 5, 5


def overlap_data(shift, seed):
    """The lesson's generator with the minority centre at (shift, shift)."""
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(950, 2), rng.randn(50, 2) * 0.8 + shift])
    y = np.r_[np.zeros(950), np.ones(50)]
    order = rng.permutation(len(y))
    return X[order][:800], y[order][:800], X[order][800:], y[order][800:]


def borderline(ref, X, y):
    """(minority row indices, how many of each one's K neighbours are majority)."""
    rows = np.where(y == 1)[0]
    return rows, np.array([np.sum(y[ref.find_k_neighbors(X, i, K)] == 0) for i in rows])


def bsmote(ref, X_min, base, n, seed):
    """SMOTE whose seed points are restricted to `base` (positions in X_min)."""
    rng, out = np.random.RandomState(seed), []
    for _ in range(n):
        j = base[rng.randint(len(base))]
        nbrs = ref.find_k_neighbors(X_min, j, K)
        q = nbrs[rng.randint(len(nbrs))]
        out.append(X_min[j] + rng.random() * (X_min[q] - X_min[j]))
    return np.array(out)


def auc(y, s):
    neg = np.sort(s[y == 0])
    lo, hi = np.searchsorted(neg, s[y == 1], "left"), np.searchsorted(neg, s[y == 1], "right")
    return float(np.mean(lo + 0.5 * (hi - lo)) / len(neg))


def score(ref, X, y, Xte, yte):
    w, b = ref.logistic_regression_weighted(X, y, np.ones(len(y)), lr=0.1, epochs=300)
    p = ref.sigmoid(Xte @ w + b)
    m = ref.compute_metrics(yte, (p >= 0.5).astype(int))
    return [m["f1"], m["precision"], m["recall"], auc(yte, p), b]


def trial(ref, shift, seed):
    Xtr, ytr, Xte, yte = overlap_data(shift, seed)
    _, n_maj = borderline(ref, Xtr, ytr)
    X_min, need = Xtr[ytr == 1], int(np.sum(ytr == 0) - np.sum(ytr == 1))
    plain = ref.smote(X_min, K, need, seed)
    border = bsmote(ref, X_min, np.where(n_maj >= 1)[0], need, seed)
    grow = lambda s: (np.vstack([Xtr, s]), np.r_[ytr, np.ones(len(s))])  # noqa: E731
    return {"none": score(ref, Xtr, ytr, Xte, yte),
            "smote": score(ref, *grow(plain), Xte, yte),
            "border": score(ref, *grow(border), Xte, yte),
            "counts": [len(n_maj), np.sum(n_maj >= 1), np.sum(n_maj == K)],
            "identical": np.array_equal(bsmote(ref, X_min, np.arange(len(X_min)), need, seed),
                                        plain)}


def summary(ref, shift):
    runs = [trial(ref, shift, s) for s in range(SEEDS)]
    out = {k: np.mean([r[k] for r in runs], axis=0) for k in ("none", "smote", "border", "counts")}
    out["identical"] = all(r["identical"] for r in runs)
    out["b0"] = (runs[0]["none"][4], runs[0]["smote"][4])
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "imbalanced")
    return {shift: summary(ref, shift) for shift in (1.5, 1.0, 2.5)}


def verify(result):
    mid, heavy, apart = result[1.5], result[1.0], result[2.5]
    aucs = [mid[k][3] for k in ("none", "smote", "border")]
    return [
        practice.Check(
            "ANSWER: Borderline-SMOTE does not beat SMOTE on overlapping classes",
            mid["border"][0] <= mid["smote"][0] and mid["smote"][0] > mid["none"][0] + 0.1,
            f"F1 over {SEEDS} seeds: none {mid['none'][0]:.3f}, SMOTE {mid['smote'][0]:.3f}, "
            f"Borderline {mid['border'][0]:.3f}; Borderline precision {mid['border'][1]:.2f} "
            f"at recall {mid['border'][2]:.2f}"),
        practice.Check(
            "FINDING: all three rank the test set the same; only the intercept moves",
            max(aucs) - min(aucs) < 0.002,
            f"mean test AUC {aucs[0]:.4f} / {aucs[1]:.4f} / {aucs[2]:.4f}; seed 0 intercept "
            f"{mid['b0'][0]:.2f} untreated, {mid['b0'][1]:.2f} with SMOTE"),
        practice.Check(
            "FINDING: under heavy overlap the exercise's rule selects every minority point",
            heavy["counts"][1] == heavy["counts"][0] and heavy["identical"],
            f"at centre (1, 1) {heavy['counts'][1]:.1f} of {heavy['counts'][0]:.1f} minority "
            f"points qualify and the output is bit-identical to the lesson's smote; "
            f"{heavy['counts'][2]:.1f} have only majority neighbours (Han et al.'s noise)"),
        practice.Check(
            "CONTROL: on the lesson's separated data the rule is selective",
            apart["counts"][1] < apart["counts"][0] / 2
            and apart["border"][0] < apart["smote"][0] < apart["none"][0],
            f"at (2.5, 2.5) it keeps {apart['counts'][1]:.1f} of {apart['counts'][0]:.1f}; F1 "
            f"Borderline {apart['border'][0]:.3f} < SMOTE {apart['smote'][0]:.3f} < none "
            f"{apart['none'][0]:.3f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
