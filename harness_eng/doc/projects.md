# Harness Engineering — Learning Projects

> Philosophy: **learn by doing**. Each project builds one concrete harness component.
> Progress from a bare agent loop to a production-grade multi-agent pipeline.

---

## Tier 1 — Foundations (already built)

| # | Project | What you learn | Repo |
|---|---------|---------------|------|
| 1 | Single-agent loop with memory | Tool dispatch, persistent K/V memory, session boundaries | `app1/` |
| 2 | Multi-agent pipeline (Planner → Generator → Evaluator) | Agent roles, feedback loop, rubric scoring | `app2/` |
| 3 | Multi-session handoff | Save/resume progress across restarts without re-running completed work | `app2/` |
| 4 | Constraint enforcement loop | External rules file, violation detection, auto-correction round | `app2/` |
| 5 | Software Eng agent (code, not docs) | Scoped tool surfaces (`CODE_SCHEMAS` vs `SCHEMAS`), workspace vs artifacts | `app2/` |

---

## Tier 2 — Intermediate POCs

### 6. Parallel Agent Fan-out
Run multiple Generator agents concurrently (one per component), then merge results.  
Introduces: concurrency management, partial-failure handling, merge strategy.  
**Output:** `app3/` with `asyncio`-based fan-out harness.

### 7. Human-in-the-Loop Gate
After evaluation, pause and print a diff; require `y/n` approval before writing the artifact.  
Introduces: interrupt points, structured human feedback re-injection, approval state saved to handoff.  
**Output:** CLI approval flow wired into the pipeline.

### 8. Self-healing Agent (auto-retry with root-cause memory)
If a tool call fails or returns a bad result, the agent writes a "failure note" to memory and retries with a different approach.  
Introduces: error taxonomy, memory-guided retry, max-retry budget.  
**Output:** `harness/self_healer.py` wrapping the tool loop.

### 9. Evaluation Calibration
Run the same task 5× with `TEMPERATURE=0.7` and record score variance.  
Plot mean ± σ per dimension; surface which rubric items are unstable.  
Introduces: meta-evaluation, rubric reliability, temperature as a lever.  
**Output:** `scripts/calibrate.py` + `artifacts/calibration_report.md`.

### 10. Provider A/B Test
Run Task D with Anthropic Haiku vs OpenAI `gpt-4o-mini`; compare scores and cost.  
Introduces: provider abstraction test, cost tracking (`input_tokens × price`), capability tiers.  
**Output:** `scripts/ab_test.py` + comparison table in `artifacts/`.

---

## Tier 3 — Advanced POCs

### 11. Persistent Vector Memory
Replace `store.json` K/V with a local embedding store (ChromaDB).  
Agents use semantic `search_memory` instead of exact key lookup.  
Introduces: RAG inside a harness, embedding freshness, retrieval quality.  
**Output:** `harness/vector_memory.py` with drop-in replacement for `memory.py`.

### 12. Multi-Project Orchestrator
A top-level "Director" agent decomposes a large goal into sub-projects, spins up a full Planner–Generator–Evaluator pipeline for each, and aggregates a final report.  
Introduces: recursive harness composition, shared memory namespace, cross-project constraint checking.  
**Output:** `app4/` with `director.py`.

### 13. Streaming Evaluator Dashboard
Stream agent output tokens to a terminal dashboard (Rich TUI) showing live scores, artifact word counts, and tool-call counts updating in real time.  
Introduces: streaming API, observable harness, live telemetry.  
**Output:** `harness/dashboard.py` using `rich.live`.

### 14. Harness-as-API (FastAPI wrapper)
Expose the pipeline over HTTP: `POST /run` triggers a task, `GET /status/{id}` polls progress, `GET /artifacts/{id}` retrieves results.  
Introduces: async task queue, session isolation, REST interface for harness consumers.  
**Output:** `app5/` with `server.py`.

### 15. Adversarial Constraint Injector
A "Red Team" agent deliberately generates artifacts that violate constraints.  
The constraint checker must catch every violation; failed catches are logged.  
Introduces: harness robustness testing, constraint coverage metrics.  
**Output:** `scripts/red_team.py` + violation miss report.

---

## Suggested Order

```
1 → 2 → 3 → 4 → 5      # Tier 1: done
↓
6 → 7 → 8               # add concurrency, human gates, self-healing
↓
9 → 10                  # measure and compare
↓
11 → 12 → 13 → 14 → 15  # production-grade features
```

Each project should be self-contained enough to run independently.  
Reuse `harness/provider.py`, `harness/memory.py`, and `harness/config.py` across all apps — that shared layer *is* the harness.
