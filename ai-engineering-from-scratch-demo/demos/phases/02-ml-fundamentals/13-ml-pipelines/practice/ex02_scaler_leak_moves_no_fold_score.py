"""Exercise 2 — a scaler fit on all the data moves no fold score; a label leak adds 24 points.

    Deliberately introduce data leakage: fit the scaler on the full dataset
    before splitting. Compare the cross-validation score (leaky) to the
    pipeline cross-validation score (clean). How large is the difference?

Reading of the exercise: the clean run is the lesson's own `FullPipeline` under
its own `cross_validate_pipeline`, which refits the imputer and scaler inside
every fold. The leaky run is the same pipeline with its numeric
imputer + scaler fit once on all 500 rows and frozen, so each fold reuses
statistics that include its validation rows. Both use the lesson's tree
(depth 5) and logistic regression, on two CV shuffles (seeds 42 and 99).

**ANSWER: the difference is exactly zero.** Leaky and clean CV agree fold for
fold in all four runs: 0.724 and 0.712 for the tree, 0.756 and 0.752 for
logistic regression. The leaked statistics barely differ: a fold's
scaler mean sits at most 0.04 standard deviations from the full-data mean.

**FINDING: the lesson's warning names the wrong leak.** `docs/en.md` says a
scaler fit before splitting "inflates accuracy estimates", and the lesson's own
`demo_data_leakage` prints a difference of +0.000. Scaling is an affine
map that does not see the labels: a tree's percentile thresholds move with it,
and gradient descent on features shifted by 0.04 sd lands on the same fold
accuracies.

**CONTROL: a leak that uses the labels does inflate the score.** On 200 rows
of pure noise (2000 features, random labels), picking the 20 features most
correlated with the label on the full dataset before CV scores 0.735 (mean of 3
seeds); the same selection inside the pipeline scores 0.490, i.e. chance.

Structure: `Frozen` is the leaky step; `cv` runs one configuration.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "13-ml-pipelines"
NUM, CAT, SEEDS = ["age", "income", "score"], ["city", "plan"], (42, 99)


class Frozen:
    """A transformer already fit on the full dataset: refitting is a no-op."""

    def __init__(self, fitted):
        self.fitted = fitted

    def fit(self, X):
        return self

    def transform(self, X):
        return self.fitted.transform(X)

    def fit_transform(self, X):
        return self.transform(X)


def cv(ref, data, model, leaky, seed):
    numeric = np.column_stack([data[c] for c in NUM])
    full = ref.TransformerPipeline([("impute", ref.MedianImputer()),
                                    ("scale", ref.StandardScaler())]).fit(numeric)

    def make():
        pipe = ref.FullPipeline(model(), NUM, CAT)
        if leaky:
            pipe.num_pipeline = Frozen(full)
        return pipe

    return ref.cross_validate_pipeline(make, data, seed=seed), full


def max_shift(ref, data, full):
    """Largest |fold scaler mean - full mean| in full-data sd, over 5 train folds."""
    idx, worst = np.random.RandomState(42).permutation(500), 0.0
    for fold in range(5):
        rows = np.concatenate([idx[: fold * 100], idx[fold * 100 + 100:]])
        part = ref.TransformerPipeline([("impute", ref.MedianImputer()),
                                        ("scale", ref.StandardScaler())])
        part.fit(np.column_stack([data[c][rows] for c in NUM]))
        mine, theirs = part.steps[1][1], full.steps[1][1]
        worst = max(worst, float(np.max(np.abs(mine.means - theirs.means) / theirs.stds)))
    return worst


def label_leak(seed):
    """(leaky, clean) CV accuracy of top-20 feature selection on pure noise."""
    rng = np.random.RandomState(seed)
    X, y = rng.randn(200, 2000), rng.randint(0, 2, 200)
    folds, select = KFold(5, shuffle=True, random_state=0), SelectKBest(f_classif, k=20)
    model = LogisticRegression(max_iter=1000)
    leaky = cross_val_score(model, select.fit_transform(X, y), y, cv=folds).mean()
    clean = cross_val_score(Pipeline([("select", select), ("m", model)]), X, y, cv=folds).mean()
    return leaky, clean


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pipeline")
    data = ref.make_mixed_data(500)
    models = {"tree": lambda: ref.DecisionTreeSimple(5),
              "logistic": lambda: ref.LogisticRegressionSimple(0.05, 1000)}
    runs = {}
    for name, model in models.items():
        for seed in SEEDS:
            leaky, full = cv(ref, data, model, True, seed)
            runs[f"{name}/{seed}"] = (leaky, cv(ref, data, model, False, seed)[0])
    controls = np.array([label_leak(s) for s in range(3)])
    return {
        "runs": {k: (float(np.mean(a)), float(np.mean(b)), float(np.max(np.abs(np.subtract(a, b)))))
                 for k, (a, b) in runs.items()},
        "shift": max_shift(ref, data, full),
        "doc_says": "This inflates accuracy estimates" in parity.doc_text(PHASE, LESSON),
        "control": controls.mean(axis=0).tolist(),
    }


def verify(result):
    runs = result["runs"]
    table = "; ".join(f"{k} {a:.3f} vs {b:.3f}" for k, (a, b, _) in runs.items())
    leaky, clean = result["control"]
    return [
        practice.Check(
            "ANSWER: fitting the scaler on the full dataset changes no fold score",
            all(gap == 0.0 for _, _, gap in runs.values()),
            f"leaky vs clean 5-fold CV, identical fold for fold: {table}; a fold's scaler "
            f"mean is at most {result['shift']:.2f} sd from the full-data mean",
        ),
        practice.Check(
            "FINDING: the lesson says this leak inflates accuracy; here it moves nothing",
            result["doc_says"] and result["shift"] < 0.2,
            "docs/en.md: 'This inflates accuracy estimates'. Scaling is affine and never sees "
            "the labels, so tree thresholds move with it and gradient descent on features "
            f"shifted by {result['shift']:.2f} sd reaches the same accuracy on every fold",
        ),
        practice.Check(
            "CONTROL: a leak that uses the labels does inflate CV, on pure noise",
            leaky > clean + 0.15 and abs(clean - 0.5) < 0.05,
            f"top-20 of 2000 noise features chosen on all 200 rows: CV {leaky:.3f}; chosen "
            f"inside each fold: {clean:.3f} (chance), mean of 3 seeds",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
