"""Train and compare multiple classifiers, logging all to MLflow for comparison."""

from __future__ import annotations

import io
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Model configurations: (name, estimator_class, hyperparams)
MODELS: list[tuple[str, type, dict[str, Any]]] = [
    ("SVM-RBF", SVC, {"C": 1.0, "kernel": "rbf", "probability": True}),
    ("SVM-Linear", SVC, {"C": 1.0, "kernel": "linear", "probability": True}),
    ("LogisticRegression", LogisticRegression, {"C": 1.0, "max_iter": 1000}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 100, "max_depth": 5}),
    ("GradientBoosting", GradientBoostingClassifier, {"n_estimators": 100, "max_depth": 3}),
]

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
}


def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    """Log confusion matrix as an artifact image."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion Matrix")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    mlflow.log_image(buf, "confusion_matrix.png")


def log_feature_importance(model: Any, feature_names: list[str], model_name: str) -> None:
    """Log feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importance - {model_name}")
    ax.set_ylabel("Importance")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    mlflow.log_image(buf, "feature_importance.png")

    # Also log as dict
    importance_dict = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
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
    target_names: list[str],
    seed: int,
    dataset_name: str,
) -> dict[str, float]:
    """Train a model and log everything to MLflow."""
    # Build pipeline
    clf = estimator_class(random_state=seed, **hyperparams)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])

    # Train
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Calculate metrics
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))

    # Cross-validation score
    cv_scores = cross_val_score(
        pipe, np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), cv=5
    )
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # ROC-AUC (for multiclass, use OvR)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)
        roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
    else:
        roc_auc = 0.0

    with mlflow.start_run(run_name=f"{model_name}-{dataset_name}"):
        # Log parameters
        mlflow.log_params(
            {
                "model": model_name,
                "dataset": dataset_name,
                "random_seed": seed,
                **hyperparams,
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "roc_auc_weighted": roc_auc,
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std,
            }
        )

        # Log tags
        mlflow.set_tags(
            {
                "framework": "sklearn",
                "model_type": model_name,
                "task": "classification",
                "demo": "multimodel-comparison",
            }
        )

        # Log classification report as text
        report = classification_report(y_test, y_pred, target_names=target_names)
        mlflow.log_text(report, "classification_report.txt")

        # Log confusion matrix
        log_confusion_matrix(y_test, y_pred, target_names)

        # Log feature importance for tree-based models
        log_feature_importance(pipe.named_steps["clf"], feature_names, model_name)

        # Log model
        mlflow.sklearn.log_model(pipe, artifact_path="model")

    print(
        f"  {model_name}: acc={acc:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f} cv={cv_mean:.4f}±{cv_std:.4f}"
    )

    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc, "cv_mean": cv_mean}


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    dataset_name = os.environ.get("MFPOC_DATASET", "iris")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", f"mfpoc-multimodel-{dataset_name}")
    mlflow.set_experiment(experiment_name)

    seed = int(os.environ.get("MFPOC_RANDOM_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)

    # Load dataset
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(DATASETS.keys())}")

    data = DATASETS[dataset_name]()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
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
            target_names=target_names,
            seed=seed,
            dataset_name=dataset_name,
        )
        results.append((model_name, metrics))

    # Summary
    print("-" * 60)
    print("Summary (sorted by accuracy):")
    results.sort(key=lambda x: x[1]["accuracy"], reverse=True)
    for i, (name, metrics) in enumerate(results, 1):
        print(f"  {i}. {name}: {metrics['accuracy']:.4f}")

    best_model = results[0][0]
    print(f"\nBest model: {best_model}")


if __name__ == "__main__":
    main()
