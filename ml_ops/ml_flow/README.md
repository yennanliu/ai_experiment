# MLflow POC — Experiment Tracking Stack

A minimal **ML Ops** demonstration: run an [MLflow](https://mlflow.org/) tracking server in Docker, train a scikit-learn model, log experiments, and inspect results in the UI.

## What This Does

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
├─────────────────────────────┬───────────────────────────────┤
│  mlflow service             │  train-demo service           │
│  ├─ Tracking server         │  ├─ Loads Iris dataset        │
│  ├─ SQLite backend          │  ├─ Trains SVM classifier     │
│  ├─ Port 5000               │  ├─ Logs params & metrics     │
│  └─ Persistent volume       │  └─ Uploads model artifact    │
└─────────────────────────────┴───────────────────────────────┘
```

**Training logs to MLflow:**
- **Parameters:** `C`, `kernel`, `random_seed`
- **Metrics:** `accuracy`, `f1_weighted`
- **Artifacts:** Serialized sklearn pipeline (StandardScaler → SVC)

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Docker with Compose v2

## Quick Start (Docker)

```bash
# Build and run full stack
docker compose up --build
```

This starts:
- **MLflow UI** at [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **train-demo** job that trains a model and logs to MLflow

Once complete, open the UI → experiment `mfpoc-iris` → view run metrics and download model.

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

# Terminal B — run training
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run mfpoc-train
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `mfpoc-iris` | Experiment name |
| `MFPOC_RUN_NAME` | (auto-generated) | Run identifier |
| `MFPOC_SVM_C` | `1.0` | SVM regularization parameter |
| `MFPOC_SVM_KERNEL` | `rbf` | SVM kernel (`rbf`, `linear`, `poly`) |
| `MFPOC_RANDOM_SEED` | `42` | Random seed for reproducibility |

## Project Structure

```
ml_flow/
├── mlflow_poc/
│   ├── __init__.py
│   └── train.py          # Training script with MLflow logging
├── pyproject.toml        # Dependencies and build config
├── uv.lock               # Locked dependencies
├── Dockerfile            # Multi-stage build with uv
└── docker-compose.yml    # Two-service stack
```

## Technical Notes

**MLflow 2.17+ Host Header Security:** Recent MLflow versions include DNS rebinding protection. The compose file uses `--allowed-hosts` to permit internal Docker networking (`mlflow:5000`) and health checks (`127.0.0.1:5000`).

**Build System:** Uses [hatchling](https://hatch.pypa.io/) with a single entry point `mfpoc-train`.

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
