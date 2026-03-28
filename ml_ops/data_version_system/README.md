# data_version_system — mini “DVC++” POC

Hash-based **dataset snapshots**, **diff across versions**, and **experiment records** that point at a dataset version + command (reproduce hints).

- Python 3.12+ (stdlib only)
- Docker optional wrapper around the same CLI

## Quick start (local)

```bash
cd ml_ops/data_version_system
export PYTHONPATH="$PWD"
mkdir -p workspace/demo_data && echo '{"a":1}' > workspace/demo_data/x.json
cd workspace
python -m dvs init
python -m dvs add demo_data --tag demo-v1
echo '{"a":2}' > demo_data/x.json
python -m dvs add demo_data --tag demo-v2
python -m dvs list
python -m dvs diff <id1> <id2>   # paste two ids from list
python -m dvs exp create --dataset <id1> --name train -- python -c "print('train')"
python -m dvs exp list
python -m dvs exp show <exp_id>
python -m dvs exp verify <exp_id>
```

## Docker

```bash
cd ml_ops/data_version_system
docker compose build
docker compose run --rm dvs init
docker compose run --rm dvs add demo_data --tag v1
docker compose run --rm dvs list
```

Mount semantics: host `./workspace` → container `/workspace`; `.dvs` lives under whatever directory you use as `-C` / `cwd`.

## Design

| Piece | Behavior |
|--------|----------|
| **Version id** | `sha256` of canonical JSON of sorted relative paths → content hashes (first 12 hex chars). Same bytes ⇒ same id. |
| **Store** | `.dvs/datasets/*.json`, `.dvs/experiments/*.json` under the chosen project root. |
| **Skipped paths** | `.dvs`, `.git`, `__pycache__`, `.venv`, `venv`, `node_modules`, and path segments starting with `.` |
| **Experiment id** | Hash of dataset version + argv + cwd (deterministic replay key). |

## Limitations (POC)

No remote sync, no large-file streaming optimizations, no partial checkout—just **manifest + lineage** you can build on.

Reference JD context: dataset/model lineage and reproducibility are core to ML tooling roles; this POC demonstrates small, inspectable building blocks.
