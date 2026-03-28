# data_version_system — mini “DVC++” POC

Hash-based **dataset snapshots**, **diff across versions**, and **experiment records** that point at a dataset version + command (reproduce hints).

Packaging and envs use **[uv](https://docs.astral.sh/uv/)** (lockfile + `uv run`). Runtime dependencies: **none** (stdlib only). **Ruff** is optional in the `dev` dependency group.

## Prereqs

- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

## Quick start (local)

```bash
cd ml_ops/data_version_system
uv sync                    # creates .venv, installs package + dev tools
# or: uv sync --no-dev     # project only

mkdir -p workspace/demo_data && echo '{"a":1}' > workspace/demo_data/x.json
cd workspace
uv run dvs init
uv run dvs add demo_data --tag demo-v1
echo '{"a":2}' > demo_data/x.json
uv run dvs add demo_data --tag demo-v2
uv run dvs list
uv run dvs diff <id1> <id2>   # paste two ids from list
uv run dvs exp create --dataset <id1> --name train -- python -c "print('train')"
uv run dvs exp list
uv run dvs exp show <exp_id>
uv run dvs exp verify <exp_id>
```

After `uv sync`, you can also activate `.venv` and run `dvs` directly.

Lockfile maintenance:

```bash
uv lock          # refresh uv.lock after dependency changes
uv sync          # apply lock to .venv
```

## Docker

The image runs `uv sync --frozen --no-dev` at build time (see [uv + Docker](https://docs.astral.sh/uv/guides/integration/docker/)).

```bash
cd ml_ops/data_version_system
docker compose build   # first time only
# Uses sample data in workspace/demo_data (or: mkdir -p workspace/demo_data && echo '{"a":1}' > workspace/demo_data/x.json)
docker compose run --rm dvs init
docker compose run --rm dvs add demo_data --tag v1
docker compose run --rm dvs list
```

Container `cwd` is `/workspace` (host folder `workspace/`), so `demo_data` means `workspace/demo_data` on your machine.

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
