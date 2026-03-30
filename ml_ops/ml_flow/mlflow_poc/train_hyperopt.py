"""Hyperparameter optimization example with GridSearchCV, logging all trials to MLflow."""

from __future__ import annotations

import os
import random
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Parameter grid for Random Forest
PARAM_GRID = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [3, 5, 10, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}


def log_grid_search_results(grid_search: GridSearchCV, parent_run_id: str) -> None:
    """Log all GridSearchCV trials as nested MLflow runs."""
    results = grid_search.cv_results_

    for i in range(len(results["params"])):
        params = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]
        rank = results["rank_test_score"][i]

        # Clean param names (remove 'clf__' prefix)
        clean_params = {k.replace("clf__", ""): v for k, v in params.items()}

        with mlflow.start_run(
            run_name=f"trial-{i + 1}-rank-{rank}", nested=True, parent_run_id=parent_run_id
        ):
            mlflow.log_params(clean_params)
            mlflow.log_metrics(
                {
                    "cv_mean_score": mean_score,
                    "cv_std_score": std_score,
                    "rank": rank,
                }
            )
            mlflow.set_tag("trial_number", i + 1)


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "mfpoc-hyperopt")
    mlflow.set_experiment(experiment_name)

    seed = int(os.environ.get("MFPOC_RANDOM_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)

    # Configuration
    cv_folds = int(os.environ.get("MFPOC_CV_FOLDS", "5"))
    n_jobs = int(os.environ.get("MFPOC_N_JOBS", "-1"))  # -1 = use all cores

    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    print(f"Dataset: Iris ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print(f"Parameter grid combinations: {np.prod([len(v) for v in PARAM_GRID.values()])}")
    print(f"CV folds: {cv_folds}")
    print("-" * 60)

    # Create pipeline
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=seed)),
        ]
    )

    # Grid search
    grid_search = GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Get best model metrics on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    print("-" * 60)
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Best params: {grid_search.best_params_}")

    # Log to MLflow
    with mlflow.start_run(run_name="hyperopt-gridsearch") as parent_run:
        # Log search configuration
        mlflow.log_params(
            {
                "search_type": "GridSearchCV",
                "cv_folds": cv_folds,
                "n_jobs": n_jobs,
                "random_seed": seed,
                "total_combinations": np.prod([len(v) for v in PARAM_GRID.values()]),
            }
        )

        # Log best parameters (clean names)
        best_params_clean = {
            k.replace("clf__", "best_"): v for k, v in grid_search.best_params_.items()
        }
        mlflow.log_params(best_params_clean)

        # Log metrics
        mlflow.log_metrics(
            {
                "best_cv_score": grid_search.best_score_,
                "test_accuracy": test_acc,
                "test_f1_weighted": test_f1,
            }
        )

        # Log tags
        mlflow.set_tags(
            {
                "framework": "sklearn",
                "model_type": "RandomForestClassifier",
                "task": "classification",
                "demo": "hyperparameter-optimization",
                "search_method": "grid",
            }
        )

        # Log parameter grid as JSON
        param_grid_serializable: dict[str, Any] = {}
        for k, v in PARAM_GRID.items():
            param_grid_serializable[k] = [str(x) if x is None else x for x in v]
        mlflow.log_dict(param_grid_serializable, "param_grid.json")

        # Log all trials as nested runs
        print("Logging all trials to MLflow...")
        log_grid_search_results(grid_search, parent_run.info.run_id)

        # Log best model
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        print(f"\nLogged to {tracking_uri}")
        print(f"Parent run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
