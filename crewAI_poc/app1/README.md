# CrewAI Research Crew

A minimal multi-agent system built with [CrewAI](https://crewai.com/). Two specialized AI agents collaborate to research a topic and produce a polished report.

---

## Architecture

```
User Input (topic)
       │
       ▼
┌─────────────────────────────────────────┐
│               Crew (Sequential)          │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  Agent 1: Senior Research Analyst│   │
│  │  Tool: SerperDev (web search)    │   │
│  │  Task: Research the topic        │   │
│  └──────────────┬───────────────────┘   │
│                 │ findings              │
│  ┌──────────────▼───────────────────┐   │
│  │  Agent 2: Content Writer         │   │
│  │  Task: Write structured report   │   │
│  └──────────────┬───────────────────┘   │
│                 │                       │
└─────────────────┼───────────────────────┘
                  │
                  ▼
           Markdown Report
```

### Key Ideas

| Concept | Description |
|---|---|
| **Agent** | An autonomous AI actor with a role, goal, backstory, and optional tools |
| **Task** | A unit of work assigned to an agent, with a clear expected output |
| **Crew** | Orchestrates agents + tasks; here using `Process.sequential` (one task feeds the next) |
| **Context passing** | The Writer's task receives the Researcher's output via `context=[research_task]` |
| **LLM** | Both agents share a single `gpt-4o-mini` instance (swap to any OpenAI model) |

---

## Project Structure

```
app1/
├── src/
│   ├── agents.py   # Agent definitions
│   ├── tasks.py    # Task definitions
│   ├── crew.py     # Crew assembly
│   └── main.py     # Entry point
├── .env.example    # API key template
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your key:
#   OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
# Default topic
uv run python -m src.main

# Custom topic
uv run python -m src.main "quantum computing breakthroughs 2025"
```

---

## Extending

- **Add an agent:** Define a new function in `agents.py`, create a task in `tasks.py`, add both to the crew in `crew.py`.
- **Change LLM:** Edit the model string in `crew.py` (e.g. `"openai/gpt-4o"`).
- **Add tools:** Import from `crewai_tools` and pass to any agent's `tools=[]` list.
- **Parallel execution:** Change `Process.sequential` → `Process.hierarchical` and add a manager LLM.
