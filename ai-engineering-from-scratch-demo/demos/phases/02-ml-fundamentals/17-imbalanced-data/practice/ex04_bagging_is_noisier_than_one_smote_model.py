"""Exercise 4 — balanced bagging is noisier than one SMOTE model, and both rank alike.

    **Ensemble with balanced bagging**: train multiple models, each on a
    balanced bootstrap sample (all minority + random subset of majority).
    Average their predictions. Compare this approach against a single model with
    SMOTE. Measure both performance and variance across runs.

Reading of the exercise: each balanced sample is the lesson's own
`random_undersample` (all 36 training positives plus 36 negatives drawn without
replacement), so the ensemble is 15 of the lesson's `logistic_regression_weighted`
models (lr 0.1, 300 epochs) with their probabilities averaged. The rival is one
model on the training set topped up by the lesson's `smote`. Both are scored at
threshold 0.5 on a fresh 10,000-row draw of `make_imbalanced_data`. "Variance
across runs" is measured twice over 10 runs: with the data fixed (the lesson's
seed 42) and only the resampling seed changing, and with a fresh training draw
each run.

**ANSWER: SMOTE scores slightly higher and varies far less.** With the data
fixed, F1 is 0.683 +- 0.002 for one SMOTE model and 0.659 +- 0.015 for the
15-model bag. Both run at recall ~1.00 and precision ~0.5.

**FINDING: bagging does cut the variance of undersampling, just not below
SMOTE's.** A single undersampled model swings by +- 0.051; averaging 15 brings
that to +- 0.015, 3.4x less, which is still 7.5x the spread of one SMOTE model,
because SMOTE keeps every negative and only its synthetic points move.

**FINDING: across fresh training sets the resampling noise vanishes into the
data noise.** With a new draw each run, F1 is 0.648 +- 0.020 for SMOTE and
0.623 +- 0.022 for the bag: the data, not the sampler, sets the spread.

**CONTROL: the three models rank the test set identically.** Mean test AUC is
0.9981 for the bag, SMOTE and the single undersampled model alike (0.9974); in
2-D a linear model's resampling moves its intercept, so every F1 gap here is a
threshold gap.

Structure: `bag` averages undersampled models; `run` scores one run of each.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "17-imbalanced-data"
MODELS, RUNS = 15, 10


def auc(y, s):
    neg = np.sort(s[y == 0])
    lo, hi = np.searchsorted(neg, s[y == 1], "left"), np.searchsorted(neg, s[y == 1], "right")
    return float(np.mean(lo + 0.5 * (hi - lo)) / len(neg))


def fit(ref, X, y, Xte):
    w, b = ref.logistic_regression_weighted(X, y, np.ones(len(y)), lr=0.1, epochs=300)
    return ref.sigmoid(Xte @ w + b)


def bag(ref, X, y, Xte, seed):
    """Average of MODELS models, each on all positives + an equal draw of negatives."""
    probs = [fit(ref, *ref.random_undersample(X, y, seed=seed + m), Xte) for m in range(MODELS)]
    return np.mean(probs, axis=0), probs[0]


def run(ref, data_seed, seed, Xte, yte):
    X, y = ref.make_imbalanced_data(950, 50, seed=data_seed)
    X, y = X[:800], y[:800]
    need = int(np.sum(y == 0) - np.sum(y == 1))
    synthetic = ref.smote(X[y == 1], k=5, n_synthetic=need, seed=seed)
    bagged, single = bag(ref, X, y, Xte, 1000 * seed)
    smoted = fit(ref, np.vstack([X, synthetic]), np.r_[y, np.ones(need)], Xte)
    out = {}
    for name, p in (("bag", bagged), ("smote", smoted), ("single", single)):
        m = ref.compute_metrics(yte, (p >= 0.5).astype(int))
        out[name] = [m["f1"], m["precision"], m["recall"], auc(yte, p)]
    return out


def spread(runs):
    return {k: (np.mean([r[k] for r in runs], axis=0), np.std([r[k] for r in runs], axis=0))
            for k in ("bag", "smote", "single")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "imbalanced")
    Xte, yte = ref.make_imbalanced_data(9500, 500, seed=7)
    return {"fixed": spread([run(ref, 42, s, Xte, yte) for s in range(RUNS)]),
            "fresh": spread([run(ref, 100 + s, s, Xte, yte) for s in range(RUNS)])}


def verify(result):
    fx, fr = result["fixed"], result["fresh"]
    f1 = lambda part, k: f"{part[k][0][0]:.3f} +- {part[k][1][0]:.3f}"  # noqa: E731
    return [
        practice.Check(
            "ANSWER: one SMOTE model edges the bag on F1 and varies far less",
            fx["smote"][0][0] > fx["bag"][0][0] and fx["smote"][1][0] < fx["bag"][1][0],
            f"data fixed, {RUNS} resampling seeds: SMOTE F1 {f1(fx, 'smote')}, {MODELS}-model "
            f"bag {f1(fx, 'bag')}; recall {fx['smote'][0][2]:.2f} / {fx['bag'][0][2]:.2f}, "
            f"precision {fx['smote'][0][1]:.2f} / {fx['bag'][0][1]:.2f}"),
        practice.Check(
            "FINDING: bagging cuts undersampling's variance, but not below SMOTE's",
            fx["single"][1][0] > 2 * fx["bag"][1][0] > 2 * fx["smote"][1][0],
            f"one undersampled model {f1(fx, 'single')}; bagged {f1(fx, 'bag')} "
            f"({fx['single'][1][0] / fx['bag'][1][0]:.1f}x less), still "
            f"{fx['bag'][1][0] / fx['smote'][1][0]:.1f}x SMOTE's spread"),
        practice.Check(
            "FINDING: across fresh training sets the data, not the sampler, sets the spread",
            fr["smote"][1][0] > 5 * fx["smote"][1][0],
            f"new training draw each run: SMOTE {f1(fr, 'smote')}, bag {f1(fr, 'bag')}"),
        practice.Check(
            "CONTROL: all three rank the test set the same",
            abs(fx["bag"][0][3] - fx["smote"][0][3]) < 1e-3
            and abs(fx["single"][0][3] - fx["smote"][0][3]) < 2e-3,
            f"mean test AUC: bag {fx['bag'][0][3]:.4f}, SMOTE {fx['smote'][0][3]:.4f}, single "
            f"undersampled {fx['single'][0][3]:.4f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
