"""Regression example using California Housing dataset with multiple models."""

from __future__ import annotations

import io
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Model configurations: (name, estimator_class, hyperparams)
MODELS: list[tuple[str, type, dict[str, Any]]] = [
    ("LinearRegression", LinearRegression, {}),
    ("Ridge", Ridge, {"alpha": 1.0}),
    ("Lasso", Lasso, {"alpha": 0.1}),
    ("ElasticNet", ElasticNet, {"alpha": 0.1, "l1_ratio": 0.5}),
    ("SVR-RBF", SVR, {"C": 1.0, "kernel": "rbf"}),
    ("RandomForest", RandomForestRegressor, {"n_estimators": 100, "max_depth": 10}),
    ("GradientBoosting", GradientBoostingRegressor, {"n_estimators": 100, "max_depth": 5}),
]


def log_predictions_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Log predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2, label="Perfect"
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Predicted vs Actual - {model_name}")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    mlflow.log_image(buf, "predictions_scatter.png")


def log_residuals_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Log residuals histogram."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals distribution
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", label="Zero")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Residuals Distribution - {model_name}")
    axes[0].legend()

    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"Residuals vs Predicted - {model_name}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    mlflow.log_image(buf, "residuals_analysis.png")


def log_feature_importance(model: Any, feature_names: list[str], model_name: str) -> None:
    """Log feature importance for tree-based or linear models."""
    importance = None
    importance_type = None

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_type = "gini_importance"
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()
        importance_type = "coefficient_magnitude"

    if importance is None:
        return

    indices = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importance)), importance[indices])
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importance ({importance_type}) - {model_name}")
    ax.set_ylabel("Importance")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    mlflow.log_image(buf, "feature_importance.png")

    # Log as dict
    importance_dict = {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}
    mlflow.log_dict(importance_dict, "feature_importance.json")


def train_and_log_model(
    model_name: str,
    estimator_class: type,
    hyperparams: dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> dict[str, float]:
    """Train a regression model and log everything to MLflow."""
    # Build pipeline
    if "random_state" in estimator_class().get_params():
        clf = estimator_class(random_state=seed, **hyperparams)
    else:
        clf = estimator_class(**hyperparams)

    pipe = Pipeline([("scale", StandardScaler()), ("reg", clf)])

    # Train
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Calculate metrics
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Cross-validation score (negative MSE, convert to positive)
    cv_scores = cross_val_score(
        pipe, np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), cv=5, scoring="r2"
    )
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    with mlflow.start_run(run_name=f"{model_name}-california"):
        # Log parameters
        mlflow.log_params(
            {
                "model": model_name,
                "dataset": "california_housing",
                "random_seed": seed,
                **hyperparams,
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "cv_mean_r2": cv_mean,
                "cv_std_r2": cv_std,
            }
        )

        # Log tags
        mlflow.set_tags(
            {
                "framework": "sklearn",
                "model_type": model_name,
                "task": "regression",
                "demo": "california-housing",
            }
        )

        # Log visualizations
        log_predictions_scatter(y_test, y_pred, model_name)
        log_residuals_plot(y_test, y_pred, model_name)
        log_feature_importance(pipe.named_steps["reg"], feature_names, model_name)

        # Log model
        mlflow.sklearn.log_model(pipe, artifact_path="model")

    print(
        f"  {model_name}: R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} cv_R2={cv_mean:.4f}±{cv_std:.4f}"
    )

    return {"r2": r2, "rmse": rmse, "mae": mae, "cv_mean": cv_mean}


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "mfpoc-regression-california")
    mlflow.set_experiment(experiment_name)

    seed = int(os.environ.get("MFPOC_RANDOM_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)

    # Load California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print(f"Dataset: California Housing ({X.shape[0]} samples, {X.shape[1]} features)")
    print("Target: Median house value (in $100,000s)")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print("-" * 60)

    results: list[tuple[str, dict[str, float]]] = []

    for model_name, estimator_class, hyperparams in MODELS:
        metrics = train_and_log_model(
            model_name=model_name,
            estimator_class=estimator_class,
            hyperparams=hyperparams,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            seed=seed,
        )
        results.append((model_name, metrics))

    # Summary
    print("-" * 60)
    print("Summary (sorted by R2):")
    results.sort(key=lambda x: x[1]["r2"], reverse=True)
    for i, (name, metrics) in enumerate(results, 1):
        print(f"  {i}. {name}: R2={metrics['r2']:.4f}")

    best_model = results[0][0]
    print(f"\nBest model: {best_model}")


if __name__ == "__main__":
    main()
