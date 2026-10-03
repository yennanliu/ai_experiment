"""Exercise 3 — ridge on raw monomials penalises the wrong terms, so the sweep never settles.

    Add L2 regularization (Ridge regression) to the experiment. For a fixed
    high-degree polynomial (degree 15), sweep lambda from 0 to 100. Plot bias^2
    and variance as functions of lambda.

Reading of the exercise: the lesson's `fit_polynomial` already takes `lam`, so
the sweep is its own `bias_variance_decomposition([15], lam=...)` at lambda = 0,
1e-3, ..., 100, printed as a table rather than drawn. The same sweep is then
re-run with x rescaled to [-1, 1] before the lesson's `fit_polynomial`, through
a loop that reproduces the lesson's draws exactly.

| lambda | 0 | 1e-3 | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bias^2 (lesson) | 102 | 2.06 | 0.391 | 0.0602 | 0.334 | 0.330 | 0.588 |
| variance (lesson) | 12400 | 315 | 82.5 | 72.4 | 124 | 31.9 | 47.4 |
| bias^2 (x in [-1, 1]) | 102 | 0.0016 | 0.0158 | 0.0845 | 0.300 | 0.785 | 1.25 |
| variance (x in [-1, 1]) | 12400 | 0.073 | 0.048 | 0.042 | 0.041 | 0.039 | 0.043 |

**ANSWER: on the lesson's code the sweep bottoms out at lambda=10 with
bias^2 + variance = 32.2** -- 500 times the best unregularised degree (0.06).
Variance never drops below 31.9 anywhere in [0, 100].

**FINDING: variance is not monotone in lambda, and high lambda never gives the
"near-constant function" the doc promises.** It rises 72 -> 124 from lambda 0.1
to 1 and 32 -> 47 from 10 to 100. `fit_polynomial` penalises raw monomials:
x^15 reaches ~1e7 on [-3, 3], so its weight is tiny and its penalty negligible,
and the penalty lands on the low-order terms instead. On one dataset the x
coefficient falls 2.07 -> 0.08 from lambda 1e-3 to 100 while the fit still spans
2.74 units.

**FINDING: rescale x to [-1, 1] and the same sweep is the textbook U.** Bias^2
rises monotonically, variance falls 0.073 -> 0.039, and lambda=0.01 reaches
0.064, level with the best unregularised degree. (At lambda=0 both runs are the
same `lstsq` fit, which is scale-invariant.)

**CONTROL:** the replay loop reproduces `bias_variance_decomposition` exactly
(difference 0.0) at every lambda, so the rescaled run differs only in x units.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "10-bias-variance"
LAMBDAS, DEGREE = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0), 15


def sweep(ref, lam, scale):
    """The lesson's decomposition loop, with x divided by `scale` before fitting."""
    rng = np.random.RandomState(42)
    x_test = np.linspace(-2.5, 2.5, 100)
    preds = []
    for _ in range(200):
        x, y = ref.generate_data(n_samples=30, noise_std=0.5, seed=rng.randint(0, 100000))
        w = ref.fit_polynomial(x / scale, y, DEGREE, lam=lam)
        preds.append(ref.predict_polynomial(x_test / scale, w))
    preds = np.array(preds)
    bias = np.mean((preds.mean(axis=0) - ref.true_function(x_test)) ** 2)
    return float(bias), float(np.mean(preds.var(axis=0)))


def coefficients(ref, lam):
    """Linear coefficient and the fit's span on one dataset, at strength `lam`."""
    x, y = ref.generate_data(n_samples=30, noise_std=0.5, seed=1)
    w = ref.fit_polynomial(x, y, DEGREE, lam=lam)
    fit = ref.predict_polynomial(np.linspace(-2.5, 2.5, 100), w)
    return float(w[1]), float(fit.max() - fit.min())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bias_variance")
    lesson = {}
    for lam in LAMBDAS:
        r = ref.bias_variance_decomposition([DEGREE], lam=lam)[DEGREE]
        lesson[lam] = (r["bias_sq"], r["variance"])
    replayed = {lam: sweep(ref, lam, 1.0) for lam in LAMBDAS}
    scaled = {lam: sweep(ref, lam, 3.0) for lam in LAMBDAS}
    s_bias = [scaled[lam][0] for lam in LAMBDAS[1:]]
    return {
        "lesson": lesson,
        "drift": max(np.max(np.abs(np.subtract(lesson[k], replayed[k]))) for k in LAMBDAS),
        "scaled": scaled,
        "monotone": bool(np.all(np.diff(s_bias) > 0)),
        "coef": {lam: coefficients(ref, lam) for lam in (0.001, 100.0)},
        "doc_says": "becomes a near-constant function" in parity.doc_text(PHASE, LESSON),
    }


def table(rows):
    return "; ".join(f"lam={lam:g}: {b:.3g}/{v:.3g}" for lam, (b, v) in rows.items())


def verify(result):
    lesson, scaled, coef = result["lesson"], result["scaled"], result["coef"]
    var = [lesson[lam][1] for lam in LAMBDAS]
    best = min(LAMBDAS, key=lambda lam: sum(lesson[lam]))
    s_best = min(LAMBDAS, key=lambda lam: sum(scaled[lam]))
    drift = result["drift"]
    return [
        practice.Check(
            "ANSWER: the lesson's sweep (bias^2/variance) bottoms out at a total error of ~32",
            all((best == 10.0, sum(lesson[best]) > 30, min(var) > 30)),
            f"{table(lesson)}. The best lambda is {best:g}, at bias^2 + variance = "
            f"{sum(lesson[best]):.1f}; noise is 0.25, and unregularised degree 6 scores 0.06",
        ),
        practice.Check(
            "FINDING: variance is not monotone in lambda, and never gets near 'near-constant'",
            all((result["doc_says"], var[4] > var[3], var[6] > var[5], coef[100.0][1] > 2.0)),
            f"variance rises {var[3]:.0f} -> {var[4]:.0f} from lambda 0.1 to 1 and {var[5]:.0f} "
            f"-> {var[6]:.0f} from 10 to 100. x^15 is ~1e7 on [-3, 3], so its weight is tiny and "
            f"barely penalised; the penalty lands on the low terms instead: on one dataset the x "
            f"coefficient falls {coef[0.001][0]:.2f} -> {coef[100.0][0]:.2f} from lambda 1e-3 to "
            f"100, while the fit still spans {coef[100.0][1]:.2f} units -- not the 'near-constant "
            "function' docs/en.md promises at high alpha",
        ),
        practice.Check(
            "FINDING: on x scaled to [-1, 1] the same sweep is the textbook U",
            all((result["monotone"], s_best == 0.01, sum(scaled[s_best]) < 0.1,
                 scaled[0.001][1] > 1.5 * scaled[10.0][1])),
            f"{table(scaled)}. Bias^2 rises monotonically, variance falls "
            f"{scaled[0.001][1]:.3f} -> {scaled[10.0][1]:.3f}, and lambda={s_best:g} reaches "
            f"{sum(scaled[s_best]):.3f}, 500x below the raw sweep's best and level with the best "
            "unregularised degree (0.05-0.06)",
        ),
        practice.Check(
            "CONTROL: the replay loop is the lesson's own decomposition",
            drift < 1e-9,
            f"at scale 1 the replay matches bias_variance_decomposition within {drift:.1e} at "
            "every lambda, so the scaled run differs only in the x units",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
