<!-- generated:start -->
# 00-setup-and-tooling / 07-docker-for-ai

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/07-docker-for-ai/) · upstream spec
`phases/00-setup-and-tooling/07-docker-for-ai/docs/en.md`

```bash
uv run demo practice run 07-docker-for-ai --ex 1
uv run demo explain 07-docker-for-ai --ex 1
uv run pytest demos/phases/00-setup-and-tooling/07-docker-for-ai
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Build the Dockerfile and run `python -c "import torch; print(torch.__version__)"` inside the… | code | T0 | `ex01_only_the_torch_trio_is_pinned.py` |
| 2 | Start the docker-compose stack and verify Qdrant is accessible from the AI container at `http… | code | T0 | `ex02_the_docs_qdrant_test_cannot_import.py` |
| 3 | Add `flask` to the Dockerfile, rebuild, and run a simple API server on port 5000. Map the por… | code | T0 | `ex03_adding_flask_rebuilds_nine_unpinned_libs.py` |
| 4 | Measure the image size with `docker images`. Try switching the base image from `devel` to `ru… | code | T0 | `ex04_runtime_saves_at_most_29_percent.py` |
<!-- generated:end -->

## Answers

The lesson's code is a `Dockerfile` and a `docker-compose.yml`, and CI has no
docker daemon, so every exercise reads those two files from the reference
`phases/` tree — the Dockerfile split into its 13 instructions, the compose file
through the harness's stdlib YAML subset — and derives what `docker build`,
`docker compose up` and `docker images` would show. All four are **T0**, stdlib only.

### 1 — the container prints `2.6.0+cu124`, but only 3 of 15 packages are pinned

**ANSWER: `2.6.0+cu124`** — the image's only torch is `torch==2.6.0+cu124`, and
`python` is deadsnakes 3.12 via `update-alternatives`.

**FINDING: 3 of the 15 pip packages are pinned.** torch, torchvision and
torchaudio are; pip, setuptools, wheel and all 9 libraries (numpy, transformers,
datasets, accelerate, …) resolve on build day. A built image is reproducible;
the Dockerfile is not.

**FINDING: the prose promises a different torch.** `docs/en.md` says "PyTorch
2.3" **5** times, including every box of the "Same image everywhere" diagram;
the Dockerfile pins 2.6.0.

**CONTROL:** CUDA 12.4.1 base, `+cu124` wheels and the `cu124` index agree;
torchvision 0.21.0 / torchaudio 2.6.0 are torch 2.6.0's companions; get-pip is
sha256-pinned; the doc's walk-through equals `code/Dockerfile`.

### 2 — Qdrant is reachable, but the doc's Python test cannot import its client

**ANSWER: yes, with curl.** Neither service declares `networks:`, so both join
`code_default` (the project is named after the `code/` directory) and `qdrant`
resolves there on port 6333. `docker compose exec ai-dev curl -s
http://qdrant:6333/collections` works because `curl` is in the apt list.

**FINDING: the lesson's own test, `from qdrant_client import QdrantClient`,
raises ModuleNotFoundError** — the image never installs `qdrant-client`.

**FINDING: `../../../` mounts `phases/`, not the repository.** The doc's
`docker run -v $(pwd):/workspace` from the repo root mounts the root, so
`/workspace` is a different tree depending on which launch path you used.

**CONTROL:** the doc's YAML equals the file; Qdrant is pinned to `v1.12.5`;
no service has `depends_on`.

### 3 — adding flask to the list re-resolves 9 unpinned libraries

A cache simulation: the first changed instruction and everything after it re-run.

| edit | first changed (of 13) | re-run | packages installed by the changed RUN |
|---|---:|---:|---|
| append `flask` to the library RUN | 8 | 5 | 10, all unpinned |
| add `RUN pip install flask` after it | 9 | 5 of 14 | flask |
| bump torch 2.6.0 → 2.5.1 | 7 | 6 | torch trio |
| base `devel` → `runtime` | 0 | 13 | everything |

**ANSWER: add flask as its own RUN**, then
`docker run --rm -p 5000:5000 -v $(pwd):/workspace ai-dev flask --app app run --host 0.0.0.0 --port 5000`.

**FINDING: editing the list silently upgrades 9 other libraries** — the edited
layer re-runs pip over transformers, datasets and the rest, none of them pinned.

**FINDING: bind 0.0.0.0.** Flask defaults to 127.0.0.1, which `-p 5000:5000`
cannot reach; the lesson already pays this for Jupyter — **3 of 3** of its
Jupyter commands pass `--ip=0.0.0.0`.

### 4 — `devel` → `runtime` is free, but saves at most 29%, not 62%

Sizes are the doc's own Step 3 table; `docker images` is not available in CI.

| | GB |
|---|---:|
| `nvidia/cuda:12.4.1-devel` | ~4 |
| `nvidia/cuda:12.4.1-runtime` | ~1.5 |
| torch stack (PyTorch runtime image ~6 − CUDA runtime) | ~4.5 |

**ANSWER: the swap is safe and saves the base's 2.5 GB** — the Dockerfile uses
nothing `devel` adds: no nvcc, flash-attn, bitsandbytes or CUDA build.

**FINDING: on the finished image that is at most 2.5 / 8.5 = 29.4%**, not the
bare base's 62.5%, because both images carry the same ~4.5 GB torch stack.

**FINDING: the one-token swap re-runs all 13 instructions**, torch download
included, so measuring it costs a full second build.

**CONTROL:** exactly one instruction changes, and the runtime tag keeps CUDA
12.4.1 / ubuntu22.04.
