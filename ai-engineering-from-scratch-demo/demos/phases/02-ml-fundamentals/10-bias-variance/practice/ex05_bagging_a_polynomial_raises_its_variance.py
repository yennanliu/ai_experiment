"""Exercise 5 — bagging a least-squares polynomial raises its variance, at every degree.

    Implement a simple bootstrap aggregating (bagging) wrapper: train 10 models
    on bootstrap samples and average predictions. Show that this reduces
    variance without increasing bias much.

Reading of the exercise: the base model is the lesson's own `fit_polynomial`
(least squares, n_train=30, noise 0.5), and bias^2 and variance are measured
the lesson's way, over 200 training sets drawn by its `generate_data` with
its seeds. The bagged model resamples each training set 10 times with
replacement, fits each resample, and averages the 10 predictions. "Show that"
is treated as a claim to test, at degrees 1, 3, 5, 8 and 15.

| degree | 1 | 3 | 5 | 8 | 15 |
|---|---:|---:|---:|---:|---:|
| variance, single fit | 0.0434 | 0.0365 | 0.0449 | 0.281 | 1.24e4 |
| variance, bagged x10 | 0.0495 | 0.0423 | 0.0798 | 36.7 | 1.13e15 |

**ANSWER: it does not. Bagging raises the variance at every degree tried**:
+14% at degree 1, +78% at degree 5, 130x at degree 8. `docs/en.md`'s practical
rule says the opposite: "high variance (deep trees, high-degree polynomials),
use bagging".

**FINDING: a bootstrap sample of 30 holds ~19 distinct points**, 19.1 on
average, so each bagged member is a degree-d fit to 19 points. At degree 15 that
nearly interpolates, and the median per-set error goes 0.52 -> 5.2e6, so the
blow-up is not one bad set.

**FINDING: even at degree 1 bagging gains nothing.** Bias^2 0.431 -> 0.425,
variance 0.0434 -> 0.0495. Least squares is a linear smoother, and the bootstrap
average of a linear fit is, to first order, the same fit, so there is nothing
to average away; bagging pays off for unstable, non-linear learners such as
deep trees, which is where the doc's random-forest example comes from.

**CONTROL: averaging itself works.** Averaging degree 5 over 10 *independent*
training sets cuts variance 0.0449 -> 0.0052 (8.6x) with bias^2 0.00110 ->
0.00096: it is resampling one set that fails. The single model reproduces the
lesson's degree-5 variance exactly.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "10-bias-variance"
DEGREES, DRAWS, BAGS, N_TRAIN = (1, 3, 5, 8, 15), 200, 10, 30
X_TEST = np.linspace(-2.5, 2.5, 100)


def single(ref, x, y, degree, rng):
    return ref.predict_polynomial(X_TEST, ref.fit_polynomial(x, y, degree))


def bagged(ref, x, y, degree, rng):
    """Average of BAGS fits, each on a bootstrap resample of (x, y)."""
    preds = []
    for _ in range(BAGS):
        idx = rng.randint(0, len(x), len(x))
        preds.append(ref.predict_polynomial(X_TEST, ref.fit_polynomial(x[idx], y[idx], degree)))
    return np.mean(preds, axis=0)


def fresh(ref, x, y, degree, rng):
    """Average of BAGS fits, each on an independent new training set: ideal bagging."""
    sets = [ref.generate_data(N_TRAIN, 0.5, seed=rng.randint(0, 100000)) for _ in range(BAGS)]
    return np.mean([single(ref, xs, ys, degree, rng) for xs, ys in sets], axis=0)


def decompose(ref, model, degree):
    """(bias^2, variance, median per-set error) over the lesson's 200 seeded sets."""
    seeds, rng = np.random.RandomState(42), np.random.RandomState(7)
    preds = []
    for _ in range(DRAWS):
        x, y = ref.generate_data(N_TRAIN, 0.5, seed=seeds.randint(0, 100000))
        preds.append(model(ref, x, y, degree, rng))
    preds, truth = np.array(preds), ref.true_function(X_TEST)
    bias = np.mean((preds.mean(axis=0) - truth) ** 2)
    median = np.median(np.mean((preds - truth) ** 2, axis=1))
    return float(bias), float(np.mean(preds.var(axis=0))), float(median)


def unique_share(draws=2000):
    rng = np.random.RandomState(0)
    return float(np.mean([len(set(rng.randint(0, N_TRAIN, N_TRAIN))) for _ in range(draws)]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bias_variance")
    return {
        "single": {d: decompose(ref, single, d) for d in DEGREES},
        "bagged": {d: decompose(ref, bagged, d) for d in DEGREES},
        "fresh": decompose(ref, fresh, 5),
        "lesson": ref.bias_variance_decomposition([5])[5]["variance"],
        "unique": unique_share(),
        "doc_rule": "high-degree polynomials), use bagging" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    one, bag, fresh = result["single"], result["bagged"], result["fresh"]
    rows = "; ".join(f"d={d}: {one[d][1]:.3g} -> {bag[d][1]:.3g}" for d in DEGREES)
    return [
        practice.Check(
            "ANSWER: it does not; bagged variance is higher at every degree tried",
            result["doc_rule"] and all(bag[d][1] > one[d][1] for d in DEGREES)
            and bag[5][1] > 1.5 * one[5][1],
            f"variance, single -> bagged: {rows}. docs/en.md's practical rule says the "
            "opposite: 'high variance (deep trees, high-degree polynomials), use bagging'",
        ),
        practice.Check(
            "FINDING: a bootstrap sample of 30 has ~19 distinct points, so high degrees blow up",
            18 < result["unique"] < 20 and bag[15][2] > 1e3 * one[15][2],
            f"a resample keeps {result['unique']:.1f} distinct points on average, and a "
            f"degree-15 fit to 19 points nearly interpolates them: the median per-set error "
            f"goes {one[15][2]:.2f} -> {bag[15][2]:.2g}, so this is not one outlier",
        ),
        practice.Check(
            "FINDING: even at degree 1 bagging gains nothing, because least squares is linear",
            abs(bag[1][0] / one[1][0] - 1) < 0.05 and bag[1][1] > one[1][1],
            f"degree 1: bias^2 {one[1][0]:.3f} -> {bag[1][0]:.3f}, variance {one[1][1]:.4f} -> "
            f"{bag[1][1]:.4f}. A linear smoother's bootstrap average is, to first order, the "
            "same fit, so bagging has nothing to average away; it pays off for unstable, "
            "non-linear learners such as deep trees",
        ),
        practice.Check(
            "CONTROL: averaging does cut variance ~10x when the 10 sets are independent",
            one[5][1] / fresh[1] > 6 and abs(one[5][1] - result["lesson"]) < 1e-12,
            f"degree 5 averaged over 10 fresh training sets: variance {one[5][1]:.4f} -> "
            f"{fresh[1]:.4f} ({one[5][1] / fresh[1]:.1f}x), bias^2 {one[5][0]:.5f} -> "
            f"{fresh[0]:.5f}. The averaging is sound; resampling one set is what fails. "
            f"(The single model reproduces the lesson's degree-5 variance exactly.)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
