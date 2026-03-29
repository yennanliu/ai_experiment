# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLflow POC — a minimal ML tooling demonstration showing experiment tracking with MLflow, scikit-learn model training, and Docker-based deployment. Uses **uv** for Python package management with a lockfile for reproducibility.

## Common Commands

### Local Development (requires uv)

```bash
# Install dependencies
uv sync

# Start MLflow tracking server (Terminal A)
mkdir -p data/artifacts
uv run mlflow server \
  --backend-store-uri sqlite:///data/mlflow.db \
  --default-artifact-root "$(pwd)/data/artifacts" \
  --host 127.0.0.1 --port 5000

# Run training job (Terminal B)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run mfpoc-train

# Lint
uv run ruff check .
uv run ruff format .

# Update lockfile after dependency changes
uv lock && uv sync
```

### Docker Compose

```bash
# Full stack (server + demo training job)
docker compose up --build

# Server only
docker compose up --build mlflow

# Re-run trainer against running server
docker compose run --rm train-demo
```

MLflow UI available at http://127.0.0.1:5000

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Docker Compose Stack                     │
├────────────────────────┬────────────────────────────────┤
│  mlflow service        │  train-demo service            │
│  - Tracking server     │  - SVM on Iris dataset         │
│  - SQLite backend      │  - Logs params/metrics/model   │
│  - Port 5000           │  - Depends on mlflow healthy   │
│  - mlflow_data volume  │                                │
└────────────────────────┴────────────────────────────────┘
```

**Training script** (`mlflow_poc/train.py`):
- Trains sklearn SVM pipeline (StandardScaler → SVC)
- Logs to MLflow: params (`C`, `kernel`, `random_seed`), metrics (`accuracy`, `f1_weighted`), and serialized model artifact

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `mfpoc-iris` | Experiment name |
| `MFPOC_RUN_NAME` | (auto) | Run identifier |
| `MFPOC_SVM_C` | `1.0` | SVM regularization |
| `MFPOC_SVM_KERNEL` | `rbf` | SVM kernel type |
| `MFPOC_RANDOM_SEED` | `42` | Reproducibility seed |

## Technical Notes

- **MLflow 2.17+ requires `--allowed-hosts`** for Docker networking (DNS rebinding protection)
- Python 3.12+, Ruff for linting (line-length: 100), hatchling build system
