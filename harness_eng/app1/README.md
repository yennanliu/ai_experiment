# Harness Engineering — Demo App

A minimal Python implementation of the five harness engineering components described in [rar.design/posts/harness-engineering-guide](https://rar.design/posts/harness-engineering-guide).

> "The model is the engine. The harness is the car. Right now, most people are trying to drive an engine across the highway."

---

## What Is Harness Engineering?

Prompt engineering tells a model *what to do*. Harness engineering builds the **infrastructure around the model** so it can work reliably: documented instructions, enforced constraints, tools, persistent memory, and an independent evaluation loop.

This app demonstrates all five components in ~150 lines of Python, switchable between Claude (Anthropic) and GPT (OpenAI) via a single `.env` line.

---

## Architecture

```
app1/
├── AGENTS.md            # [1] Repository structure — agent's standing instructions
├── harness/
│   ├── config.py        #     Loads .env (provider, model, API keys)
│   ├── provider.py      # [3] Tools & provider abstraction (Anthropic / OpenAI)
│   ├── tools.py         # [2] Architectural constraint — safe tool allow-list
│   ├── agent.py         # [4] Context management — agentic loop + memory I/O
│   ├── memory.py        #     Persistent JSON store across sessions
│   └── evaluator.py     # [5] Evaluation loop — separate model grades every answer
├── memory/
│   └── store.json       #     Facts persisted between runs
├── main.py              #     Entry point / demo questions
├── .env.example         #     Config template
└── pyproject.toml       #     uv project (Python 3.11+)
```

### The Five Components

| # | Component | Where | What it does |
|---|---|---|---|
| 1 | **Repository Structure** | `AGENTS.md` | Loaded as the system prompt — the agent's single source of truth |
| 2 | **Architectural Constraint** | `tools.py` | Restricts `eval` to a safe math scope; no arbitrary code execution |
| 3 | **Tools** | `provider.py` | `calculate`, `remember`, `recall` via a tool-use loop |
| 4 | **Context Management** | `memory.py` + `agent.py` | Facts stored to `memory/store.json`, reloaded on every run |
| 5 | **Evaluation Loop** | `evaluator.py` | A *separate* model call scores each answer — generators can't grade their own work |

### How the Agent Loop Works

```
user question
     │
     ▼
 LLM call ──► tool_use? ──yes──► execute tool(s) ──► LLM call (loop)
     │
    no
     │
     ▼
 final answer ──► evaluator scores it ──► print result
```

The `provider.py` layer normalises Anthropic and OpenAI differences so `agent.py` and `evaluator.py` are provider-agnostic.

---

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd app1
uv sync
```

---

## Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
# "anthropic" or "openai"
PROVIDER=anthropic

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001   # optional, this is the default

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini                    # optional, this is the default
```

Only the key for the chosen provider needs to be set.

---

## Run

```bash
uv run python main.py
```

**Example output:**

```
────────────────────────────────────────────────────
Q: What is 18% of 240? Store the result as 'tip'.
A: 18% of 240 is 43.2. I've stored this as 'tip'.
Eval [5/5]: Correct calculation, clearly stated, stored as requested.

────────────────────────────────────────────────────
Q: Recall 'tip'. What is the total if the original bill was $240?
A: The tip is $43.20, so the total bill is $283.20.
Eval [5/5]: Accurate recall and correct addition.

────────────────────────────────────────────────────
Done. Session memory saved to memory/store.json
```

To switch providers, change one line in `.env` and re-run — no code changes needed.

---

## Key Ideas

**AGENTS.md is the harness, not the model.**  
Rules, tool descriptions, and constraints live in a file the agent always reads. Anything not in-file doesn't exist from the agent's perspective.

**The evaluator is always a separate call.**  
A model rating its own output consistently overestimates quality. `evaluator.py` uses an independent LLM call with a different system prompt.

**Memory outlives the session.**  
`memory/store.json` persists facts between runs. This is the simplest form of long-term context management — the agent can recall something stored yesterday.

**Provider is a config detail, not an architectural decision.**  
The `Conversation` abstraction in `provider.py` normalises tool-call formats between Anthropic and OpenAI so the rest of the codebase never sees API differences.
