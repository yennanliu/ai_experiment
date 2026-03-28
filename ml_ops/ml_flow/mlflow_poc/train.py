"""Minimal sklearn run that logs params, metrics, and a model to MLflow."""

from __future__ import annotations

import os
import random
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "mfpoc-iris"))

    seed = int(os.environ.get("MFPOC_RANDOM_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)

    C = float(os.environ.get("MFPOC_SVM_C", "1.0"))
    kernel = os.environ.get("MFPOC_SVM_KERNEL", "rbf")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    pipe: Pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", SVC(C=C, kernel=kernel, random_state=seed)),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    tags: dict[str, Any] = {"demo": "iris", "framework": "sklearn"}

    with mlflow.start_run(run_name=os.environ.get("MFPOC_RUN_NAME")):
        mlflow.log_params({"C": C, "kernel": kernel, "random_seed": seed})
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        mlflow.set_tags(
            {
                "tracking_uri": tracking_uri,
                "notes": "POC run; see ml_ops/ml_flow/README.md",
            }
        )

    print(f"Logged run to {tracking_uri} | accuracy={acc:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
