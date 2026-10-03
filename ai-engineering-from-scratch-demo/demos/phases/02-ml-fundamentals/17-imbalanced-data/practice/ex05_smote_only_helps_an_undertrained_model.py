"""Exercise 5 — SMOTE only "helps" at 99/1, and only because the lesson's model is under-trained.

    **Imbalance ratio experiment**: take a balanced dataset and progressively
    increase the imbalance ratio (50/50, 70/30, 90/10, 95/5, 99/1). For each
    ratio, train with and without SMOTE. Plot F1 vs imbalance ratio for both
    approaches. At what ratio does SMOTE start making a meaningful difference?

Reading of the exercise: the balanced dataset is the lesson's
`make_imbalanced_data(500, 500)`; each ratio keeps all 500 negatives and a random
subset of the positives (500, 214, 56, 26, 5). Each training set is fitted with
the lesson's `logistic_regression_weighted` with and without the lesson's
`smote` topping positives up to parity, and scored by F1 at threshold 0.5 on a
fresh 10,000-row test set drawn at the same ratio, averaged over 5 seeds. It is
run twice: with the lesson's training settings (lr 0.1, 300 epochs), and run to
convergence (lr 1.0, 3000 epochs). The "plot" is the printed table.

**ANSWER: with the lesson's model, SMOTE helps only at 99/1, and hurts before
that.** F1 without / with SMOTE at 50/50 to 99/1: 0.964 / 0.964, 0.954 / 0.923,
0.914 / 0.776, 0.806 / 0.632 and 0.000 / 0.240. SMOTE first makes a meaningful
difference at 90/10 -- a loss of 0.138 -- and its only gain is at 99/1.

**FINDING: the 99/1 "gain" is rescuing an unconverged fit, not the imbalance.**
With 5 positives, 300 gradient steps from zero leave the model scoring F1 0.000
at threshold 0.5. Trained to convergence, the same untreated model scores 0.767
at 99/1, and SMOTE lowers F1 at every imbalanced ratio (0.960 / 0.958,
0.918 / 0.895, 0.886 / 0.822, 0.767 / 0.536). On a linear model SMOTE drags the
0.5 boundary towards the majority, trading precision for recall -- a trade F1
at 0.5 does not reward once the model is fitted properly.

**CONTROL: SMOTE is no better than duplicating rows.** On the lesson's model the
lesson's own `random_oversample` scores within 0.02 of SMOTE at every ratio
(0.924, 0.769, 0.620, 0.228 against SMOTE's 0.923, 0.776, 0.632, 0.240), and at
50/50, where there is nothing to add, all three fits coincide at 0.964.

Structure: `subsample` builds a ratio; `trial` scores the three treatments.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "17-imbalanced-data"
RATIOS, SEEDS = (50, 70, 90, 95, 99), 5
SETTINGS = {"lesson": (0.1, 300), "converged": (1.0, 3000)}


def subsample(ref, majority, seed):
    """All 500 negatives of a balanced draw plus enough positives for the ratio."""
    X, y = ref.make_imbalanced_data(500, 500, seed=seed)
    keep_pos = np.random.RandomState(seed).choice(np.where(y == 1)[0],
                                                  round(500 * (100 - majority) / majority), False)
    keep = np.r_[np.where(y == 0)[0], keep_pos]
    return X[keep], y[keep]


def f1(ref, X, y, Xte, yte, setting):
    lr, epochs = SETTINGS[setting]
    w, b = ref.logistic_regression_weighted(X, y, np.ones(len(y)), lr=lr, epochs=epochs)
    return ref.compute_metrics(yte, (ref.sigmoid(Xte @ w + b) >= 0.5).astype(int))["f1"]


def trial(ref, majority, seed, setting):
    X, y = subsample(ref, majority, seed)
    Xte, yte = ref.make_imbalanced_data(100 * majority, 100 * (100 - majority), seed=1000 + seed)
    need = int(np.sum(y == 0) - np.sum(y == 1))
    if need == 0:
        base = f1(ref, X, y, Xte, yte, setting)
        return base, base, base
    synthetic = ref.smote(X[y == 1], k=5, n_synthetic=need, seed=seed)
    over = ref.random_oversample(X, y, seed=seed)
    return (f1(ref, X, y, Xte, yte, setting),
            f1(ref, np.vstack([X, synthetic]), np.r_[y, np.ones(need)], Xte, yte, setting),
            f1(ref, *over, Xte, yte, setting))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "imbalanced")
    return {setting: {m: np.mean([trial(ref, m, s, setting) for s in range(SEEDS)], axis=0)
                      for m in RATIOS} for setting in SETTINGS}


def verify(result):
    les, conv = result["lesson"], result["converged"]
    row = lambda part, ms: ", ".join(  # noqa: E731
        f"{m}/{100 - m} {part[m][0]:.3f}/{part[m][1]:.3f}" for m in ms)
    return [
        practice.Check(
            "ANSWER: on the lesson's model SMOTE hurts until 99/1, its only gain",
            les[99][1] > les[99][0] + 0.1 and les[90][1] < les[90][0] - 0.1,
            f"F1 without/with SMOTE: {row(les, RATIOS)}"),
        practice.Check(
            "FINDING: the 99/1 gain rescues an unconverged fit; converged, SMOTE always hurts",
            les[99][0] == 0 and conv[99][0] > 0.6
            and all(conv[m][1] < conv[m][0] for m in RATIOS[1:]),
            f"untreated at 99/1: F1 {les[99][0]:.3f} after 300 epochs, {conv[99][0]:.3f} "
            f"converged; converged without/with SMOTE: {row(conv, RATIOS[1:])}"),
        practice.Check(
            "CONTROL: SMOTE is no better than the lesson's random oversampling",
            all(abs(les[m][1] - les[m][2]) < 0.02 for m in RATIOS),
            "lesson model, SMOTE vs duplication: " + ", ".join(
                f"{m}/{100 - m} {les[m][1]:.3f}/{les[m][2]:.3f}" for m in RATIOS[1:])),
        practice.Check(
            "CONTROL: at 50/50 SMOTE adds nothing, so all three fits coincide",
            les[50][0] == les[50][1] == les[50][2],
            f"F1 {les[50][0]:.3f} for all three at 50/50"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
