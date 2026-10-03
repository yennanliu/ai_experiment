"""Exercise 4 — squares go after imputation and before scaling, and here they buy one fold's noise.

    Add a custom transformer to the pipeline that creates polynomial features
    (degree 2) for the two most important numeric columns. Where should it go in
    the pipeline?

Reading of the exercise: "most important" is measured, not assumed: drop each
numeric column from the lesson's own `FullPipeline` (logistic regression,
lr=0.1, 1000 steps) and see what 5-fold CV under the lesson's
`cross_validate_pipeline` loses. The custom transformer `Square` appends a^2,
ab and b^2 for the two winners, in the lesson's fit/transform/fit_transform
interface, and is tried in each of the three slots of the lesson's numeric
`TransformerPipeline`: before imputation, between imputation and scaling, and
after scaling.

**ANSWER: income and score; put it after imputation, before scaling.** Dropping
income costs 6.4 points of CV accuracy, score 4.0, age 0.2. With `Square`
between imputation and scaling CV is 0.766 against 0.758 without it.

**FINDING: before imputation it silently builds wrong features.** It does not
crash: NaN squared is NaN, and the lesson's `MedianImputer` fills the product
columns with their own medians. On 12 rows (every row with a missing income)
the imputed `income * score` is not the imputed income times score, and on
the same 12 the imputed income^2 is not income squared. CV happens to tie at
0.766.

**FINDING: after scaling the squares are left unscaled, and accuracy drops
below the no-square baseline** (0.754 against 0.758): the square of a
standardized lognormal income reaches 30.4, while every scaled column has unit
variance, so one appended column dominates the lesson's gradient descent.

**CONTROL: the gain is noise.** +0.8 points is 4 rows of 500 and smaller than
the 1.2-point spread across folds: `make_mixed_data` draws its label from a
boundary that is linear in age, income and score, so there is no curvature to find.

Structure: `Square` is the transformer; `cv` swaps it into the pipeline.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "13-ml-pipelines"
NUM, CAT, PAIR = ["age", "income", "score"], ["city", "plan"], (1, 2)
ORDERS = {"before impute": ("square", "impute", "scale"),
          "impute, square, scale": ("impute", "square", "scale"),
          "after scale": ("impute", "scale", "square")}


class Square:
    """Degree-2 terms a^2, ab, b^2 for two numeric columns, appended to the input."""

    def __init__(self, cols=PAIR):
        self.cols = cols

    def fit(self, X):
        return self

    def transform(self, X):
        a, b = X[:, self.cols[0]], X[:, self.cols[1]]
        return np.column_stack([X, a * a, a * b, b * b])

    def fit_transform(self, X):
        return self.transform(X)


def cv(ref, data, numeric=NUM, order=None):
    """Mean and fold spread of 5-fold CV, optionally with a custom numeric chain."""
    def make():
        pipe = ref.FullPipeline(ref.LogisticRegressionSimple(0.1, 1000), numeric, CAT)
        if order:
            parts = {"impute": ref.MedianImputer(), "scale": ref.StandardScaler(),
                     "square": Square()}
            pipe.num_pipeline = ref.TransformerPipeline([(k, parts[k]) for k in order])
        return pipe

    scores = ref.cross_validate_pipeline(make, data)
    return float(np.mean(scores)), float(np.std(scores))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pipeline")
    data = ref.make_mixed_data(500)
    base, spread = cv(ref, data)
    drops = {c: base - cv(ref, data, [n for n in NUM if n != c])[0] for c in NUM}
    raw = np.column_stack([data[c] for c in NUM])
    early = ref.MedianImputer().fit_transform(Square().transform(raw))
    scaled = ref.StandardScaler().fit_transform(ref.MedianImputer().fit_transform(raw))
    return {
        "base": base, "spread": spread, "drops": drops,
        "orders": {name: cv(ref, data, order=o)[0] for name, o in ORDERS.items()},
        "bad_cross": int(np.sum(~np.isclose(early[:, 4], early[:, 1] * early[:, 2]))),
        "bad_square": int(np.sum(~np.isclose(early[:, 3], early[:, 1] ** 2))),
        "missing_income": int(np.isnan(data["income"]).sum()),
        "max_square": float(np.max(scaled[:, 1] ** 2)),
    }


def verify(result):
    drops, orders, base = result["drops"], result["orders"], result["base"]
    top = sorted(drops, key=drops.get, reverse=True)[:2]
    middle = orders["impute, square, scale"]
    return [
        practice.Check(
            "ANSWER: income and score; the transformer goes after imputation, before scaling",
            top == ["income", "score"] and middle >= base,
            "dropping a column costs " + ", ".join(f"{c} {v:+.3f}" for c, v in drops.items())
            + f"; with Square between impute and scale CV is {middle:.3f} against {base:.3f}",
        ),
        practice.Check(
            "FINDING: before imputation the products are imputed, not computed",
            result["bad_cross"] == result["missing_income"] > 0 and result["bad_square"] > 0,
            f"no crash, but on {result['bad_cross']} rows (all {result['missing_income']} with a "
            f"missing income) imputed income*score is not imputed income x score, and on "
            f"{result['bad_square']} income^2 is not income squared; CV "
            f"{orders['before impute']:.3f}",
        ),
        practice.Check(
            "FINDING: after scaling the squares are unscaled and CV falls below baseline",
            orders["after scale"] < base and result["max_square"] > 10,
            f"impute -> scale -> Square scores {orders['after scale']:.3f} against {base:.3f}; "
            f"the squared z-score of income reaches {result['max_square']:.1f}",
        ),
        practice.Check(
            "CONTROL: the best placement's gain is within fold noise",
            0 <= middle - base < result["spread"],
            f"+{middle - base:.3f} is {round((middle - base) * 500)} rows of 500, under the "
            f"{result['spread']:.3f} fold spread; make_mixed_data's boundary is linear",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
