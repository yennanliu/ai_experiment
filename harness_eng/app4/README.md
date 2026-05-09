# app4 — Multi-Project Orchestrator

Projects 11 · 12 · 13 from `doc/projects.md`.

## What's in here

| File | Project | What it does |
|------|---------|-------------|
| `harness/vector_memory.py` | **11** | ChromaDB persistent semantic store — replaces the flat `store.json` K/V from app1-3. Embeds documents with `all-MiniLM-L6-v2` (local ONNX), no API key needed. |
| `harness/graph.py` + `nodes.py` | **12** | LangGraph `StateGraph` with three async nodes. `decompose` (Director) breaks a goal into sub-projects; `process_project` fan-outs concurrently (one per sub-project); `aggregate` synthesises a final report. |
| `harness/dashboard.py` | **13** | Rich TUI with `Live` + `Layout`. Shows project statuses, scores, and streams LLM tokens live from the evaluator via `astream_events`. |

## Graph topology

```
START → decompose → [Send × N] → process_project → aggregate → END
                    (parallel fan-out, one per sub-project)
```

Each `process_project` invocation:
1. Retrieves semantic context from ChromaDB
2. Plans (LLM) → Generates artifact (LLM) → Evaluates (streaming LLM)
3. Stores artifact summary back into ChromaDB for future runs

## Run

```bash
# install
uv sync

# First run downloads the ONNX embedding model (~80 MB, cached in ~/.cache/chroma/)
uv run python main.py
```

## Config (`.env`)

```
PROVIDER=anthropic         # or openai
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
MAX_TOKENS=2048
TEMPERATURE=0
```

## Outputs

- `artifacts/<slug>.md` — one artifact per sub-project
- `artifacts/_report.md` — final orchestration report with scores
- `memory/chroma/` — persisted ChromaDB vector store (grows across runs)
