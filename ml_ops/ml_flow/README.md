# MLflow POC — Experiment Tracking Stack

A minimal **ML Ops** demonstration: run an [MLflow](https://mlflow.org/) tracking server in Docker, train scikit-learn models, log experiments, and inspect results in the UI.

## What This Does

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
├─────────────────────────────┬───────────────────────────────┤
│  mlflow service             │  Training services            │
│  ├─ Tracking server         │  ├─ train-demo (basic SVM)    │
│  ├─ SQLite backend          │  ├─ train-multimodel          │
│  ├─ Port 5000               │  ├─ train-regression          │
│  └─ Persistent volume       │  └─ train-hyperopt            │
└─────────────────────────────┴───────────────────────────────┘
```

## Available Examples

| Command | Description | Experiment Name |
|---------|-------------|-----------------|
| `mfpoc-train` | Basic SVM on Iris | `mfpoc-iris` |
| `mfpoc-multimodel` | Compare 5 classifiers (SVM, RF, LR, GB) | `mfpoc-multimodel-{dataset}` |
| `mfpoc-regression` | 7 regressors on California Housing | `mfpoc-regression-california` |
| `mfpoc-hyperopt` | GridSearchCV with nested runs | `mfpoc-hyperopt` |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Docker with Compose v2

## Quick Start (Docker)

```bash
# Build and run full stack (basic demo)
docker compose up --build
```

This starts:
- **MLflow UI** at [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **train-demo** job that trains a model and logs to MLflow

Once complete, open the UI → experiment `mfpoc-iris` → view run metrics and download model.

### Run Advanced Examples

```bash
# Multi-model comparison (5 classifiers)
docker compose up --build mlflow
docker compose run --rm --profile multimodel train-multimodel

# Regression with California Housing
docker compose run --rm --profile regression train-regression

# Hyperparameter optimization (GridSearchCV)
docker compose run --rm --profile hyperopt train-hyperopt

# Use Wine dataset instead of Iris
docker compose run --rm -e MFPOC_DATASET=wine --profile multimodel train-multimodel
```

### Common Operations

```bash
# Server only (no training job)
docker compose up --build mlflow

# Run training again
docker compose run --rm train-demo

# Run with custom hyperparameters
docker compose run --rm -e MFPOC_SVM_C=0.5 -e MFPOC_SVM_KERNEL=linear train-demo

# Stop everything
docker compose down

# Stop and remove data volume
docker compose down -v
```

## Local Development (without Docker)

```bash
uv sync

# Terminal A — start tracking server
mkdir -p data/artifacts
uv run mlflow server \
  --backend-store-uri sqlite:///data/mlflow.db \
  --default-artifact-root "$(pwd)/data/artifacts" \
  --host 127.0.0.1 \
  --port 5000

# Terminal B — run training examples
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

uv run mfpoc-train           # Basic SVM
uv run mfpoc-multimodel      # Compare 5 classifiers
uv run mfpoc-regression      # Regression models
uv run mfpoc-hyperopt        # Hyperparameter search
```

## What Gets Logged

### Basic Training (`mfpoc-train`)
- **Parameters:** `C`, `kernel`, `random_seed`
- **Metrics:** `accuracy`, `f1_weighted`
- **Artifacts:** Serialized sklearn pipeline

### Multi-Model Comparison (`mfpoc-multimodel`)
- **Models:** SVM-RBF, SVM-Linear, LogisticRegression, RandomForest, GradientBoosting
- **Metrics:** `accuracy`, `f1_weighted`, `precision`, `recall`, `roc_auc`, CV scores
- **Artifacts:** Confusion matrix PNG, feature importance chart, classification report

### Regression (`mfpoc-regression`)
- **Models:** LinearRegression, Ridge, Lasso, ElasticNet, SVR, RandomForest, GradientBoosting
- **Metrics:** `mse`, `rmse`, `mae`, `r2`, CV scores
- **Artifacts:** Predicted vs actual scatter, residuals analysis, feature importance

### Hyperparameter Search (`mfpoc-hyperopt`)
- **Parent run:** Best parameters, test metrics
- **Nested runs:** Each trial's params and CV score
- **Artifacts:** Best model, parameter grid JSON

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | varies | Experiment name |
| `MFPOC_RANDOM_SEED` | `42` | Random seed for reproducibility |
| `MFPOC_SVM_C` | `1.0` | SVM regularization (basic train) |
| `MFPOC_SVM_KERNEL` | `rbf` | SVM kernel (basic train) |
| `MFPOC_DATASET` | `iris` | Dataset for multimodel (`iris` or `wine`) |
| `MFPOC_CV_FOLDS` | `5` | Cross-validation folds (hyperopt) |
| `MFPOC_N_JOBS` | `-1` | Parallel jobs for GridSearch |

## Project Structure

```
ml_flow/
├── mlflow_poc/
│   ├── __init__.py
│   ├── train.py              # Basic SVM training
│   ├── train_multimodel.py   # Multi-model comparison
│   ├── train_regression.py   # Regression example
│   └── train_hyperopt.py     # Hyperparameter optimization
├── pyproject.toml            # Dependencies and build config
├── uv.lock                   # Locked dependencies
├── Dockerfile                # Multi-stage build with uv
└── docker-compose.yml        # Service definitions
```

## Technical Notes

**MLflow 2.17+ Host Header Security:** Recent MLflow versions include DNS rebinding protection. The compose file uses `--allowed-hosts` to permit internal Docker networking (`mlflow:5000`) and health checks (`127.0.0.1:5000`).

**Build System:** Uses [hatchling](https://hatch.pypa.io/) with entry points for all training scripts.

**Lockfile Maintenance:**
```bash
uv lock    # Update lockfile
uv sync    # Install from lockfile
```

## Limitations (POC)

This is a demonstration project. For production:
- Replace SQLite with PostgreSQL or MySQL
- Use S3/GCS/Azure Blob for artifact storage
- Add authentication (MLflow supports basic auth, OAuth)
- Deploy behind a reverse proxy with TLS
