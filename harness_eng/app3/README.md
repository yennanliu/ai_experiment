# app3 — Parallel Agent Fan-out

> Harness Engineering series · [app1 (basics)](../app1) · [app2 (advanced)](../app2) · **app3 (fan-out)**

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │              Planner                │
                        │   task → JSON plan (components[])   │
                        └──────────────┬──────────────────────┘
                                       │ plan
                         ┌─────────────▼─────────────┐
                         │         Fan-out            │
                         │   ThreadPoolExecutor       │
                         │   (one thread / component) │
                         └──┬──────────┬──────────┬───┘
                  mem_copy  │  mem_copy│  mem_copy│
                 ┌──────────▼┐ ┌───────▼──┐ ┌────▼──────┐
                 │ Generator │ │Generator │ │ Generator │
                 │ component │ │component │ │ component │
                 │     A     │ │    B     │ │     C     │
                 └──────────┬┘ └───────┬──┘ └────┬──────┘
                  artifact  │  artifact│  artifact│
                            └─────┬────┘          │
                                  │  merge deltas │
                         ┌────────▼───────────────▼──┐
                         │          Merger            │
                         │  reads artifacts/ on disk  │
                         │  writes _merged.md         │
                         └────────────┬───────────────┘
                                      │
                         ┌────────────▼───────────────┐
                         │         Evaluator          │
                         │  scores all artifacts      │
                         │  including _merged.md      │
                         └────────────────────────────┘
```

**Module map:**

| File | Role |
|---|---|
| `harness/fan_out.py` | Thread pool; isolated memory per agent; merge on join |
| `harness/generator.py` | Tool loop (takes `mem: dict`; caller owns lifecycle) |
| `harness/merger.py` | Synthesises parallel artifacts into one integration summary |
| `harness/planner.py` | Breaks task into components JSON |
| `harness/evaluator.py` | Rubric scoring (completeness / correctness / quality) |
| `harness/provider.py` | Anthropic + OpenAI behind one interface |
| `harness/tools.py` | Design-artifact tool schemas + dispatcher |
| `harness/memory.py` | JSON file-backed K/V store |

---

## Key Ideas

### 1. Concurrent generation
Each plan component runs in its own thread simultaneously. A 4-component plan that takes 20 s sequentially completes in ~20 s wall-clock instead of ~80 s — the bottleneck is the slowest agent, not the sum.

### 2. Memory isolation
Every thread receives `dict(shared_mem)` — a shallow copy of the shared memory snapshot. Agents write to their own copy without racing. After all threads join, deltas are merged back with last-write-wins per key.

```
shared_mem ──copy──► thread A mem  ──delta──┐
           ──copy──► thread B mem  ──delta──┼──► merged_mem
           ──copy──► thread C mem  ──delta──┘
```

### 3. Partial-failure handling
A component that throws an exception produces a `ComponentResult(error=...)` instead of crashing the pipeline. The merger notes the gap explicitly; the evaluator still runs against the successful artifacts.

### 4. Merger as a distinct agent
Parallel agents produce independent documents — they never see each other's work. The **Merger** is a separate, dedicated pass that reads all artifacts and writes a cross-cutting integration summary (`_merged.md`). This surfaces interface mismatches and risks that no single generator could see.

### 5. `verbose=False` during fan-out
Generators suppress per-tool-call prints while running concurrently to avoid interleaved output. The results table (printed after all threads join) shows elapsed time per component instead.

---

## How to Run

**Prerequisites:** Python 3.11+, `uv` (or `pip`)

```bash
# 1. Clone / navigate
cd harness_eng/app3

# 2. Install dependencies
uv sync
# or: pip install anthropic openai python-dotenv

# 3. Configure credentials
cp .env.example .env
# edit .env — set PROVIDER and the matching API key

# 4. Run
uv run python main.py
```

**Expected output:**

```
══════════════════════════════════════════════════════════════
  Parallel Agent Fan-out Demo  (app3)
  Provider : anthropic
  Model    : claude-haiku-4-5-20251001
  Tier     : high
══════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────
  Distributed Cache Layer
  [Planner] Decomposing task...
  Components (4): eviction_policy, replication, client_interface, observability
  Workers       : up to 4 parallel threads

  [Fan-out] Launching 4 generator agents in parallel...
  [Fan-out] All agents finished in 18.3s total
  Component                    Status   Time  Artifact
  ─────────────────────────── ──────── ──────  ────────────────────
  eviction_policy              OK       16.1s  eviction_policy
  replication                  OK       18.3s  replication
  client_interface             OK       14.9s  client_interface
  observability                OK       15.7s  observability

  [Merger] Synthesising artifacts into integration summary...
  [Evaluator] Scoring all artifacts...
```

**Outputs:**

| Path | Contents |
|---|---|
| `artifacts/<slug>.md` | One design doc per component |
| `artifacts/_merged.md` | Integration summary written by the Merger |
| `memory/store.json` | Merged memory across all agents |

**Configuration (`.env`):**

```bash
PROVIDER=anthropic          # or openai
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
MAX_WORKERS=4               # max parallel threads
MAX_TOKENS=2048
TEMPERATURE=0
```
