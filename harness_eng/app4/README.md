# app4 — Multi-Project Orchestrator

Implements **Projects 11 · 12 · 13** from `doc/projects.md` using LangGraph, LangChain, and ChromaDB.

---

## What each file does

| File | Project | Role |
|------|---------|------|
| `harness/vector_memory.py` | **11** | ChromaDB semantic store — replaces flat K/V `store.json` from app1-3 |
| `harness/graph.py` + `nodes.py` | **12** | LangGraph `StateGraph` — Director → concurrent fan-out → aggregate |
| `harness/dashboard.py` | **13** | Rich `Live` TUI — project status, scores, live LLM token stream |

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │             OrchestratorState            │
                         │  goal, gen_id, projects, results,        │
                         │  final_report                            │
                         └─────────────────────────────────────────┘

 START
   │
   ▼
┌──────────┐   JSON plan    ┌───────────────────────────────────────┐
│ decompose│ ─────────────► │          _fan_out() router            │
│ (Director│                │  returns [Send("process_project", …)] │
│  LLM)    │                └───────────────────────────────────────┘
└──────────┘                         │           │           │
      │ writes decomposition          ▼           ▼           ▼    (concurrent)
      │ to ChromaDB          ┌──────────┐ ┌──────────┐ ┌──────────┐
      │                      │ process  │ │ process  │ │ process  │
      │                      │ project  │ │ project  │ │ project  │
      │                      │ A        │ │ B        │ │ C        │
      │                      └────┬─────┘ └────┬─────┘ └────┬─────┘
      │                           │             │             │
      │                      each node:         │             │
      │                       1. search ChromaDB│             │
      │                       2. plan (LLM)     │             │
      │                       3. design doc (LLM)            │
      │                       4. code gen (LLM) │             │
      │                       5. evaluate (LLM, streaming)    │
      │                       6. write to ChromaDB            │
      │                           │             │             │
      │                           └──────┬──────┘
      │                                  │ results[] accumulated
      │                                  │ via operator.add reducer
      │                                  ▼
      │                          ┌──────────────┐
      └─────────────────────────►│  aggregate   │
                                 │  (Director   │
                                 │   LLM)       │
                                 └──────┬───────┘
                                        │
                                       END
                          artifacts/{gen_id}/_report.md
```

### Per-run output layout

```
artifacts/
  20260509_175030/        ← gen_id (timestamp of the run)
    ingestion.md          ← design document
    ingestion.py          ← generated Python implementation
    storage.md
    storage.py
    query_api.md
    query_api.py
    _report.md            ← integration report with scores
  20260509_191500/        ← second run, nothing overwritten
    …
memory/
  chroma/                 ← ChromaDB vector store (grows across runs)
```

---

## Key ideas

### 1. LangGraph `Send` for concurrent fan-out

The conditional edge after `decompose` returns a list of `Send` objects — one per sub-project. LangGraph runs each target node invocation concurrently in separate async tasks. No `ThreadPoolExecutor`, no manual task tracking: the graph runtime handles scheduling.

```python
def _fan_out(state) -> list[Send]:
    return [
        Send("process_project", {"goal": state["goal"], "gen_id": state["gen_id"], "project": p})
        for p in state["projects"]
    ]
```

### 2. `Annotated[list, operator.add]` as a fan-out reducer

The `results` field uses `operator.add` as its reducer. Each concurrent `process_project` node returns `{"results": [one_item]}`. LangGraph applies the reducer to merge them: `[] + [A] + [B] + [C] = [A, B, C]`. The `aggregate` node then sees all results at once.

```python
class OrchestratorState(TypedDict):
    results: Annotated[list[dict], operator.add]   # fan-out writes, aggregate reads
```

### 3. `astream_events` for live token capture

`graph.astream_events(inputs, version="v2")` yields a stream of structured events from every component inside the graph — including per-token LLM stream chunks. The dashboard subscribes to `on_llm_stream` events to feed the live stream panel.

```python
async for event in graph.astream_events(inputs, version="v2"):
    if event["event"] == "on_llm_stream":
        chunk = event["data"]["chunk"]          # AIMessageChunk
        dash.push_token(chunk.content)
```

### 4. Filter `name == node`, not just `node`

LangGraph's routing functions (`_fan_out`) fire their own `on_chain_end` events with the same `metadata["langgraph_node"]` as the parent. For `decompose`, `_fan_out` fires first with `output = [Send(…)]` — a list. Matching on `event["name"] == "decompose"` in addition to the metadata selects only the actual node's event.

```python
# ✗ too broad — catches _fan_out's event too
if etype == "on_chain_end" and node == "decompose":

# ✓ matches only the decompose node itself
if etype == "on_chain_end" and name == "decompose" and node == "decompose":
```

### 5. Versioned artifact runs (gen_id)

Every `main()` call generates `gen_id = datetime.now().strftime("%Y%m%d_%H%M%S")`. The ID flows through the full graph state and each `process_project` writes under `artifacts/{gen_id}/`. Multiple runs never overwrite each other, so you can compare outputs side-by-side or track improvement over time.

### 6. ChromaDB semantic memory with local embeddings

ChromaDB's default embedding function uses `all-MiniLM-L6-v2` via ONNX (downloaded once to `~/.cache/chroma/`). No API key required. Documents are upserted with a stable key so repeated runs refine the stored summaries rather than duplicating them.

```python
# Before planning: retrieve relevant prior knowledge
ctx_hits = vector_memory.search(task, n=3)

# After generating: store summary for future runs
vector_memory.add(f"artifact:{gen_id}:{slug}", summary, {"type": "artifact"})
```

---

## Key knowledge

| Topic | What to know |
|-------|-------------|
| **LangGraph `Send`** | Fan-out primitive. The dict in `Send(node, dict)` is the full input state for that invocation — it does NOT inherit from the main state automatically. Pass every field the node needs explicitly. |
| **State reducers** | Fields without a reducer are **overwritten** on each write; fields with `Annotated[T, reducer]` are **merged**. Use `operator.add` for lists you want to accumulate across parallel nodes. |
| **`astream_events` v2** | Prefer `version="v2"` — v1 has different event shapes. Events include `on_chain_start/end` (nodes, chains), `on_llm_stream` (tokens), `on_tool_start/end`. Each has `metadata["langgraph_node"]` telling you which graph node it came from. |
| **Routing fn events** | Conditional edge functions are themselves LangChain chains and fire their own `on_chain_start/end` events. Their metadata inherits the parent node's `langgraph_node`. Always filter by `event["name"]` to avoid ambiguity. |
| **Async nodes** | LangGraph runs async node functions with `asyncio`. Concurrent `Send` invocations share the same event loop — no threads needed. Use `await llm.ainvoke()` rather than `.invoke()` inside async nodes. |
| **ChromaDB distances** | The default ONNX model returns L2 (Euclidean) distances, not cosine similarities. Lower = more similar. `score = 1 - distance` is an approximation; for strict cosine sim, configure the collection with `hnsw:space = "cosine"`. |
| **Streaming `ainvoke`** | Setting `streaming=True` on a LangChain chat model makes `ainvoke` emit `on_llm_stream` events token-by-token via LangSmith callbacks, even though `ainvoke` itself still returns only when complete. |

---

## Run

```bash
uv sync          # install deps (first time)
uv run python main.py
```

The first run downloads the ONNX embedding model (~80 MB, cached in `~/.cache/chroma/`). Subsequent runs are instant.

Each run appends to ChromaDB, so the semantic memory improves with repeated runs on similar goals.

---

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
