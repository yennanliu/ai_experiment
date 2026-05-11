# CrewAI Expert Examples

Three expert-level patterns that go beyond app2. Each example isolates a distinct advanced concept.

---

## Examples at a Glance

| # | Example | Key Concept | Run Command |
|---|---------|-------------|-------------|
| 1 | Flow with Router | `@start` / `@router` / `@listen` multi-crew state machine | `uv run python -m src.example1_flow.main "quantum computing"` |
| 2 | Memory | `memory=True` — agents recall facts across crew runs | `uv run python -m src.example2_memory.main` |
| 3 | Guardrails + Callbacks | Output validation + real-time step/task monitoring | `uv run python -m src.example3_guardrails.main "AI in healthcare"` |

---

## Example 1 — Flow with Router

**What it shows:** A `Flow` is a stateful pipeline that can branch, route, and coordinate multiple crews. The `@router` decorator dynamically dispatches to different crews based on runtime data — no `if/else` logic in the caller.

```
topic (user input)
     │
     ▼
@start classify_topic ──► direct LLM call → "technical" | "business"
     │
     ▼
@router route_topic
     │
     ├──► "technical" ──► run_technical_crew  ─┐
     │                    (Technical Analyst)   │ sets state.raw_content
     └──► "business"  ──► run_business_crew   ─┘
                          (Business Analyst)
                                │
                                ▼
              @listen(both) polish_article
                          (Senior Editor)
                                │
                                ▼
                         state.final_article
```

**Key wiring in `flow.py`:**

```python
class ArticleState(BaseModel):
    topic: str = ""
    content_type: str = ""
    raw_content: str = ""
    final_article: str = ""

class SmartArticleFlow(Flow[ArticleState]):

    @start()
    def classify_topic(self):
        response = LLM("openai/gpt-4o-mini").call(f"Classify '{self.state.topic}'...")
        self.state.content_type = "technical" if "tech" in response.lower() else "business"
        return self.state.content_type   # ← router reads this return value

    @router(classify_topic)
    def route_topic(self, content_type: str) -> str:
        return content_type              # emits "technical" or "business" as route key

    @listen("technical")
    def run_technical_crew(self): ...   # only fires when route = "technical"

    @listen("business")
    def run_business_crew(self): ...    # only fires when route = "business"

    @listen(or_(run_technical_crew, run_business_crew))  # OR — fires when either completes
    def polish_article(self): ...
```

State (`ArticleState`) is a Pydantic model — all methods read/write via `self.state`. The flow's state is fully inspectable after `kickoff()`.

**Run:**
```bash
# Technical route
uv run python -m src.example1_flow.main "transformer neural networks"

# Business route
uv run python -m src.example1_flow.main "venture capital in Southeast Asia"
```

---

## Example 2 — Memory

**What it shows:** `memory=True` enables three memory layers:

| Layer | Storage | Scope | Purpose |
|---|---|---|---|
| Short-term | RAM | Current run | RAG context injection between tasks |
| Long-term | SQLite (`.crewai/`) | Cross-run | Persist learned facts across sessions |
| Entity | ChromaDB (`.crewai/`) | Cross-run | Track named entities (people, places, products) |

When you run three related questions in sequence, each crew run stores its findings. Later runs recall prior answers — the third answer explicitly references facts from the first two.

**Key wiring in `crew.py`:**

```python
Crew(
    agents=[researcher, synthesizer],
    tasks=[t1, t2],
    process=Process.sequential,
    memory=True,        # ← all three memory layers enabled automatically
    verbose=True,
)
```

No other changes needed. The crew handles extraction, storage, and retrieval automatically.

**Demo — three linked questions:**
```
Q1: What is the Python GIL and why does it exist?
Q2: How does the GIL affect multithreaded programs?   ← recalls Q1 facts
Q3: What are alternatives for true CPU parallelism?   ← recalls Q1 + Q2
```

**Run:**
```bash
# Default: 3 linked questions (best demo of memory recall)
uv run python -m src.example2_memory.main

# Single custom question
uv run python -m src.example2_memory.main "What is RLHF in LLM training?"
```

Memory is stored in `.crewai/` — delete it to reset:
```bash
rm -rf .crewai/
```

---

## Example 3 — Guardrails + Callbacks

**What it shows:** Two orthogonal quality-control mechanisms:

- **Guardrails** — per-task output validators. If validation fails, the agent retries automatically with the failure reason injected into the prompt.
- **Callbacks** — side-effects that fire on every agent step (`step_callback`) and every completed task (`task_callback`). Used here for structured logging.

```
analysis_task ──► guardrail: validate_analysis
                  • rejects if < 80 words
                  • rejects if no analytical language
                  • agent retries with failure message if rejected
      │
      ▼ (PASSED)
report_task ───► guardrail: validate_report
                  • rejects if ## Introduction or ## Conclusion missing
                  • rejects if < 150 words
      │
      ▼ (PASSED)
   final output

Throughout:
  step_callback → output/steps.log   (every agent thought/action)
  task_callback → output/tasks.log   (every completed task)
```

**Key wiring in `tasks.py`:**

```python
def validate_analysis(result: TaskOutput) -> Tuple[bool, Any]:
    word_count = len(result.raw.split())
    if word_count < 80:
        return (False, f"Too brief ({word_count} words). Need 80+ with specific findings.")
    if not any(t in result.raw.lower() for t in ["because", "therefore", "data", "shows"]):
        return (False, "Add evidence-based language: 'data shows', 'therefore', etc.")
    return (True, result.raw.strip())   # (True, transformed_output)

Task(..., guardrail=validate_analysis)  # ← attached per task
```

**Key wiring in `crew.py`:**

```python
def on_step(step) -> None:
    # fires on every agent thought, tool call, or observation
    log("steps.log", f"{type(step).__name__} | {str(step)[:200]}")

def on_task(output: TaskOutput) -> None:
    # fires once per completed (guardrail-passed) task
    log("tasks.log", f"DONE | agent={output.agent} | words={len(output.raw.split())}")

Crew(..., step_callback=on_step, task_callback=on_task)
```

**Run:**
```bash
uv run python -m src.example3_guardrails.main "remote work productivity trends"
uv run python -m src.example3_guardrails.main "electric vehicle adoption barriers"

# After running, inspect logs:
cat output/steps.log
cat output/tasks.log
```

---

## Project Structure

```
app3/
├── src/
│   ├── example1_flow/
│   │   ├── flow.py      # Flow class: @start, @router, @listen
│   │   └── main.py
│   ├── example2_memory/
│   │   ├── agents.py
│   │   ├── tasks.py
│   │   ├── crew.py      # memory=True
│   │   └── main.py
│   └── example3_guardrails/
│       ├── agents.py
│       ├── tasks.py     # validate_* guardrail functions
│       ├── crew.py      # step_callback + task_callback
│       └── main.py
├── output/              # logs from example 3
├── .crewai/             # memory DB (auto-created by example 2, gitignored)
├── .env.example
├── .gitignore
├── .python-version      # 3.12
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
cd app3
cp .env.example .env
# Add: OPENAI_API_KEY=sk-...

uv sync
```

---

## Concept Ladder

| App | Patterns covered |
|-----|-----------------|
| app1 | Sequential crew, context passing |
| app2 | Custom `@tool`, Pydantic output, Hierarchical crew |
| **app3** | **Flow + routing, Memory, Guardrails, Callbacks** |

---

## Key Differences from app2

| Feature | app2 | app3 |
|---|---|---|
| Multi-crew coordination | Manual wiring | `Flow` manages state + routing |
| Output typing | `output_pydantic` (one task) | `guardrail` validates + retries |
| Observability | None | `step_callback` + `task_callback` |
| Memory | None | Short-term + Long-term + Entity |
