"""Exercise 1 — sklearn imputes a missing category; the lesson's encoder crashes on it.

    Build a pipeline that handles a dataset with 3 numeric columns and 2
    categorical columns. Use `ColumnTransformer` to apply median imputation +
    scaling to numerics and most-frequent imputation + one-hot encoding to
    categoricals. Train with 5-fold cross-validation.

Reading of the exercise: the dataset is the lesson's own `make_mixed_data(500)`
(age, income, score; city, plan), held as one object array so the
`ColumnTransformer` routes columns by index. The model is a logistic
regression, scored by shuffled 5-fold CV. The lesson already ships a
from-scratch version of the same pipeline (`FullPipeline`), so the sklearn
build is checked against it, and both are then given a categorical value that
is actually missing, which is the only case most-frequent imputation exists for.

**ANSWER: 5-fold CV accuracy 0.758 +/- 0.012** (folds 0.74 to 0.77). The lesson's
from-scratch `FullPipeline` with its own gradient-descent logistic regression
scores 0.756 under its own `cross_validate_pipeline`.

**FINDING: the lesson's pipeline has no categorical imputation, and its encoder
crashes on a missing category.** With 5% of `city` set to None, the lesson's
`OneHotEncoder.fit` raises TypeError (`sorted(set(...))` cannot order None
against str), and with NaN it raises the same error for float against str. The sklearn
pipeline imputes the most frequent city and scores 0.760 on the same data.

**CONTROL: on clean data the two preprocessors are the same map.** The lesson's
data has 26 missing ages, 12 missing incomes and no missing categories, so
most-frequent imputation is a no-op there, and the sklearn `ColumnTransformer`
output matches the lesson's `num_pipeline` + `cat_encoder` within 5.8e-15 over
all 11 columns.

Structure: `table` builds the object array; `sk_pipeline` is the build.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "13-ml-pipelines"
NUM, CAT = ["age", "income", "score"], ["city", "plan"]
CV = KFold(5, shuffle=True, random_state=42)


def table(data):
    out = np.empty((len(data["target"]), 5), dtype=object)
    for i, col in enumerate(NUM + CAT):
        out[:, i] = data[col]
    return out


def sk_pipeline():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", numeric, [0, 1, 2]), ("cat", categorical, [3, 4])])
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=1000))])


def scratch_error(ref, data, value):
    """The exception the lesson's FullPipeline raises when 5% of city is `value`."""
    holed = dict(data, city=data["city"].astype(object))
    holed["city"][::20] = value
    try:
        ref.FullPipeline(ref.DecisionTreeSimple(5), NUM, CAT).fit(holed)
    except Exception as exc:  # noqa: BLE001 -- the failure is the measurement
        return type(exc).__name__
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pipeline")
    data = ref.make_mixed_data(500)
    X, y = table(data), data["target"]
    scores = cross_val_score(sk_pipeline(), X, y, cv=CV)
    holed = X.copy()
    holed[::20, 3] = None
    mine = sk_pipeline()[0].fit_transform(X).astype(float)
    scratch = ref.FullPipeline(ref.LogisticRegressionSimple(0.05, 1000), NUM, CAT)
    theirs = np.hstack([
        scratch.num_pipeline.fit_transform(np.column_stack([data[c] for c in NUM])),
        scratch.cat_encoder.fit_transform(np.column_stack([data[c] for c in CAT])),
    ])
    return {
        "cv": scores.tolist(),
        "scratch_cv": float(np.mean(ref.cross_validate_pipeline(lambda: scratch, data))),
        "holed_cv": float(cross_val_score(sk_pipeline(), holed, y, cv=CV).mean()),
        "errors": [scratch_error(ref, data, v) for v in (None, np.nan)],
        "missing": [int(np.isnan(data[c]).sum()) for c in NUM]
        + [int(sum(v is None for v in data[c])) for c in CAT],
        "gap": float(np.abs(mine - theirs).max()),
        "width": mine.shape[1],
    }


def verify(result):
    cv = np.array(result["cv"])
    return [
        practice.Check(
            "ANSWER: the ColumnTransformer pipeline scores about 0.76 under 5-fold CV",
            len(cv) == 5 and 0.72 < cv.mean() < 0.80,
            f"5-fold CV accuracy {cv.mean():.3f} +/- {cv.std():.3f} (folds {cv.min():.2f} to "
            f"{cv.max():.2f}); the lesson's from-scratch FullPipeline scores "
            f"{result['scratch_cv']:.3f}",
        ),
        practice.Check(
            "FINDING: the lesson's encoder crashes on a missing category; sklearn imputes it",
            result["errors"] == ["TypeError", "TypeError"] and result["holed_cv"] > 0.72,
            f"with 5% of city set to None and to NaN the lesson's FullPipeline.fit raises "
            f"{result['errors']}: OneHotEncoder sorts a set of mixed types and the pipeline "
            f"has no categorical imputer. The sklearn pipeline scores {result['holed_cv']:.3f}",
        ),
        practice.Check(
            "CONTROL: on clean data the sklearn and from-scratch preprocessors agree",
            result["gap"] < 1e-12 and result["missing"][3:] == [0, 0],
            f"missing per column (age, income, score, city, plan) {result['missing']}, so "
            f"most-frequent imputation is a no-op on the lesson's data; the "
            f"{result['width']}-column ColumnTransformer output matches num_pipeline + "
            f"cat_encoder within {result['gap']:.1e}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
