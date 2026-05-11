# CrewAI Advanced Examples

Three self-contained advanced patterns built on top of the `app1` foundation. Each example isolates one concept so you can study and extend it independently.

---

## Examples at a Glance

| # | Example | Key Concept | Run Command |
|---|---------|-------------|-------------|
| 1 | Custom Tools | `@tool`-decorated functions given to agents | `uv run python -m src.example1_custom_tools.main` |
| 2 | Structured Output | Tasks return validated Pydantic models | `uv run python -m src.example2_structured_output.main "fintech"` |
| 3 | Hierarchical Crew | Manager LLM delegates to specialist agents | `uv run python -m src.example3_hierarchical.main "SaaS productivity tool"` |

---

## Example 1 — Custom Tools

Agents receive Python functions decorated with `@tool`. The framework automatically calls these functions when the agent decides it needs them — no manual invocation.

```
User Input (filepath)
       │
       ▼
┌──────────────────────────────────────┐
│  Agent: Document Analyst             │
│  Tools: read_text_file, count_words  │──► reads file + stats
└──────────────────┬───────────────────┘
                   │ analysis findings
┌──────────────────▼───────────────────┐
│  Agent: Report Writer                │
│  Tools: save_report                  │──► writes output/report.md
└──────────────────────────────────────┘
```

**Key file:** `src/example1_custom_tools/tools.py`

```python
from crewai.tools import tool

@tool("Read Text File")
def read_text_file(filepath: str) -> str:
    """Reads and returns the full content of a text file."""
    with open(filepath) as f:
        return f.read()
```

The docstring becomes the tool's description — agents read it to decide when to use the tool.

**Run:**
```bash
# Uses auto-created sample.txt
uv run python -m src.example1_custom_tools.main

# Analyze your own file
uv run python -m src.example1_custom_tools.main path/to/your/file.txt
```

Output saved to `output/report.md`.

---

## Example 2 — Structured Output (Pydantic)

Tasks can return validated Pydantic objects instead of raw strings. Downstream code gets typed, machine-readable data — no parsing required.

```
User Input (market)
       │
       ▼
┌──────────────────────────────┐
│  Agent: Market Analyst       │──► raw research text
└──────────────┬───────────────┘
               │ context
┌──────────────▼───────────────┐
│  Agent: Business Strategist  │
│  output_pydantic=MarketReport│──► MarketReport(topic, key_facts,
└──────────────────────────────┘     opportunities, risks,
                                     recommendation, confidence_score)
```

**Key file:** `src/example2_structured_output/models.py`

```python
class MarketReport(BaseModel):
    topic: str
    key_facts: List[str]
    opportunities: List[str]
    risks: List[str]
    recommendation: str
    confidence_score: float  # 0.0 – 1.0
```

**Key wiring** in `tasks.py`:
```python
Task(..., output_pydantic=MarketReport)
```

Accessing the result:
```python
result = crew.kickoff()
report: MarketReport = result.pydantic  # fully typed
print(report.confidence_score)
print(report.model_dump())             # → dict / JSON
```

**Run:**
```bash
uv run python -m src.example2_structured_output.main "electric vehicles"
uv run python -m src.example2_structured_output.main "B2B SaaS"
```

---

## Example 3 — Hierarchical Crew

A manager LLM orchestrates multiple specialist agents. The manager breaks down the goal, assigns work, reviews outputs, and synthesizes a final answer — without you wiring any `context=` chains manually.

```
User Input (product idea)
       │
       ▼
┌─────────────────────────────────────────────┐
│       Manager LLM (gpt-4o)                  │  ← orchestrates
│       decides who does what and when        │
└──────┬──────────────┬────────────────┬──────┘
       │              │                │
       ▼              ▼                ▼
┌────────────┐ ┌────────────┐ ┌──────────────────┐
│  Market    │ │ Technical  │ │ Marketing        │
│  Analyst   │ │ Lead       │ │ Specialist       │
│            │ │            │ │                  │
│ Market     │ │ Tech stack │ │ GTM strategy     │
│ analysis   │ │ & roadmap  │ │ & 90-day plan    │
└────────────┘ └────────────┘ └──────────────────┘
       │              │                │
       └──────────────┴────────────────┘
                      │
                      ▼
             Manager synthesizes
             final product launch plan
```

**Key wiring** in `crew.py`:
```python
Crew(
    agents=[analyst, tech, marketer],
    tasks=[task1, task2, task3],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4o"),  # stronger model manages
)
```

The manager LLM (`gpt-4o`) is separate from the worker LLMs (`gpt-4o-mini`) — you can tune cost vs. quality at each tier independently.

**Run:**
```bash
uv run python -m src.example3_hierarchical.main "AI-powered personal finance app"
uv run python -m src.example3_hierarchical.main "smart home security system"
```

---

## Project Structure

```
app2/
├── src/
│   ├── example1_custom_tools/
│   │   ├── tools.py     # @tool-decorated functions
│   │   ├── agents.py    # agents with tools=[]
│   │   ├── tasks.py
│   │   ├── crew.py
│   │   └── main.py
│   ├── example2_structured_output/
│   │   ├── models.py    # Pydantic output schema
│   │   ├── agents.py
│   │   ├── tasks.py     # output_pydantic=MarketReport
│   │   ├── crew.py
│   │   └── main.py
│   └── example3_hierarchical/
│       ├── agents.py
│       ├── tasks.py
│       ├── crew.py      # Process.hierarchical + manager_llm
│       └── main.py
├── output/              # generated by example 1
├── .env.example
├── .gitignore
├── .python-version      # 3.12
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
# From the app2/ directory
cp .env.example .env
# Add your key: OPENAI_API_KEY=sk-...

uv sync
```

---

## Key Concepts Compared

| Pattern | Where it lives | Why it matters |
|---|---|---|
| `@tool` decorator | `tools.py` | Lets agents call arbitrary Python code (APIs, files, DBs) |
| `output_pydantic=` | task definition | Guarantees typed output — no fragile string parsing |
| `Process.hierarchical` | crew definition | Manager handles delegation; you define tasks, not execution order |
| `manager_llm=` | crew definition | Separate, stronger model for orchestration vs. worker execution |
