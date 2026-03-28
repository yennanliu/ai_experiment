# MLflow POC — experiment tracking stack

Small **ML tooling engineering** demo: run an [MLflow](https://mlflow.org/) tracking server in Docker, log a sklearn baseline from Python, and inspect runs in the UI. Packaging matches `ml_ops/data_version_system` (**uv** + lockfile + slim **Docker** image).

**Docs this POC maps to** (`ml_ops/doc/`): experiment/metrics registry, reproducible jobs, and a one-command demo (`docker compose`) as recommended for portfolio PoCs.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for local Python workflows
- Docker (and Compose v2) for the bundled stack

## Local quick start (uv only)

```bash
cd ml_ops/ml_flow
uv sync

# Terminal A — tracking server (SQLite + local artifacts under ./data)
mkdir -p data/artifacts
uv run mlflow server \
  --backend-store-uri sqlite:///data/mlflow.db \
  --default-artifact-root "$(pwd)/data/artifacts" \
  --host 127.0.0.1 \
  --port 5000
```

```bash
# Terminal B — log a run
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run mfpoc-train
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) → experiment `mfpoc-iris` → compare metrics and downloaded model artifact.

Optional env vars for the trainer: `MLFLOW_EXPERIMENT_NAME`, `MFPOC_SVM_C`, `MFPOC_SVM_KERNEL`, `MFPOC_RANDOM_SEED`, `MFPOC_RUN_NAME`.

## Docker / Compose

Build uses [uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/) (`uv sync --frozen --no-dev`).

```bash
cd ml_ops/ml_flow
docker compose up --build
```

- **mlflow** — UI and API on [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **train-demo** — one-shot job that logs params/metrics/model to the server; backend + artifacts persist in the `mlflow_data` volume.

Tracking server only (no automatic demo job):

```bash
docker compose up --build mlflow
```

Run the trainer again after `compose up`:

```bash
docker compose run --rm train-demo
```

Lockfile maintenance:

```bash
uv lock
uv sync
```

## Limitations (POC)

SQLite and a single-node artifact directory are fine for demos only. Production would use a remote metadata store (e.g. Postgres) and object storage for artifacts, plus auth in front of the UI.
