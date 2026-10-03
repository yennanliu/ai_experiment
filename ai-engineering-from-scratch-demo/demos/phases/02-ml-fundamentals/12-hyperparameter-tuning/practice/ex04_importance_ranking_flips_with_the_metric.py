"""Exercise 4 — which random-forest hyperparameters matter on breast cancer.

    Use Optuna to tune a RandomForestClassifier on a real dataset (e.g.,
    sklearn's breast cancer dataset). Use
    `optuna.visualization.plot_param_importances(study)` to see which
    hyperparameters matter most. Does it match the importance ranking from this
    lesson?

Reading of the exercise: `optuna` is not a dependency of this repo, so this
ships the scaled-down runnable DESIGN D11 asks for. 40 random-search trials are
drawn with the lesson's own `sample_param` over n_estimators, max_depth,
min_samples_leaf and max_features of scikit-learn's `RandomForestClassifier`,
plus a `dummy` hyperparameter that is never passed to the model, and each is
scored on a stratified 30% holdout of the breast-cancer data. Importance is what
Optuna's default evaluator computes: an fANOVA-style main effect -- the variance
of each parameter's marginal mean under a random-forest surrogate fitted to the
trials, normalised to sum to 1. The lesson's ranking is its "Hyperparameter
Importance" list and its practical table, whose Random Forest row reads
"n_estimators, max_depth, min_samples_leaf", with max_features under "Low
importance".

**ANSWER: no.** For holdout accuracy the ranking is min_samples_leaf 73.8%,
max_features 23.9%, dummy 2.0%, n_estimators 0.2%, max_depth 0.1%. The parameter
the lesson calls low-importance is second; its first two sit below the dummy.

**FINDING: change the metric and n_estimators goes from last to first.** On the
same 40 trials, scored by holdout log loss: n_estimators 59.4%, max_features
37.2%, max_depth 2.8%, min_samples_leaf 0.5%, dummy 0.0%. A small forest votes
in steps of 1/n_estimators, which log loss punishes and accuracy cannot see;
"which hyperparameters matter" has no answer until the metric is fixed.

**CONTROL:** the dummy scores 2.0% and 0.0% -- the noise floor of this
importance with 40 trials. The real run is `pip install optuna`, then
`optuna.visualization.plot_param_importances(study)`.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "12-hyperparameter-tuning"
TRIALS, SEED = 40, 0
SPACE = {"n_estimators": ("int", 10, 200), "max_depth": ("int", 2, 20),
         "min_samples_leaf": ("int", 1, 20), "max_features": ("float", 0.05, 1.0),
         "dummy": ("float", 0.0, 1.0)}  # sampled and logged, never passed to the model
REAL = "pip install optuna; optuna.visualization.plot_param_importances(study)"


def run_trials(ref):
    """Random-search trials (the lesson's sampler) scored on a stratified holdout."""
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    rng, params, scores = np.random.RandomState(SEED), [], []
    for _ in range(TRIALS):
        p = {k: ref.sample_param(v, rng) for k, v in SPACE.items()}
        model = RandomForestClassifier(
            n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]), max_features=p["max_features"],
            random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
        prob = np.clip(model.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)
        params.append([p[k] for k in SPACE])
        scores.append((model.score(X_te, y_te), log_loss(y_te, prob)))
    return np.array(params), np.array(scores)


def main_effects(params, score):
    """fANOVA-style importance: the variance of each parameter's marginal mean under a
    random-forest surrogate, normalised to sum to 1 (Optuna's default evaluator)."""
    surrogate = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(params, score)
    effects = []
    for j in range(params.shape[1]):
        marginal = []
        for value in np.quantile(params[:, j], np.linspace(0, 1, 15)):
            grid = params.copy()
            grid[:, j] = value
            marginal.append(surrogate.predict(grid).mean())
        effects.append(np.var(marginal))
    return dict(zip(SPACE, np.array(effects) / sum(effects)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "tuning")
    params, scores = run_trials(ref)
    doc = parity.doc_text(PHASE, LESSON)
    return {"accuracy": main_effects(params, scores[:, 0]),
            "log_loss": main_effects(params, scores[:, 1]),
            "acc_range": (scores[:, 0].min(), scores[:, 0].max()),
            "doc_low": "- Max features (for random forests)" in doc.split("**Low importance:**")[1],
            "doc_rf_row": "| Random Forest | n_estimators, max_depth, min_samples_leaf |" in doc}


def ranking(effects):
    return ", ".join(f"{k} {v:.1%}" for k, v in sorted(effects.items(), key=lambda kv: -kv[1]))


def verify(result):
    acc, loss = result["accuracy"], result["log_loss"]
    lo, hi = result["acc_range"]
    return [
        practice.Check(
            "ANSWER: no -- min_samples_leaf and max_features matter; the doc's top two do not",
            max(acc, key=acc.get) == "min_samples_leaf" and acc["max_features"] > 0.1
            and max(acc["n_estimators"], acc["max_depth"]) < acc["dummy"]
            and result["doc_low"] and result["doc_rf_row"],
            f"importance for holdout accuracy over {TRIALS} trials ({lo:.3f}-{hi:.3f}): "
            f"{ranking(acc)}. The doc ranks n_estimators and max_depth first for random forests "
            "and lists max_features under 'Low importance'",
        ),
        practice.Check(
            "FINDING: change the metric and n_estimators goes from last to first",
            max(loss, key=loss.get) == "n_estimators" and acc["n_estimators"] < 0.01,
            f"importance for holdout log loss on the same trials: {ranking(loss)}. A small "
            "forest votes in coarse steps of 1/n_estimators, which log loss punishes and "
            "accuracy cannot see; 'which hyperparameters matter' has no answer without a metric",
        ),
        practice.Check(
            "CONTROL: a hyperparameter the model never sees scores near zero",
            acc["dummy"] < 0.05 and loss["dummy"] < 0.05,
            f"'dummy' is sampled and logged like the others but never passed to the forest: "
            f"{acc['dummy']:.1%} (accuracy) and {loss['dummy']:.1%} (log loss). Under accuracy, "
            f"n_estimators ({acc['n_estimators']:.1%}) and max_depth ({acc['max_depth']:.1%}) "
            f"sit below that floor. Optuna is not a dependency; the real run is: {REAL}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
