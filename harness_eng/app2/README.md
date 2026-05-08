# app2 — Advanced Harness Engineering Examples

Extends `app1` with three advanced patterns from the [Harness Engineering Guide](https://rar.design/posts/harness-engineering-guide).

## What's new vs. app1

| Capability | app1 | app2 |
|---|---|---|
| Agent architecture | Single agent + evaluator | **Three-agent pipeline** (Planner + Generator + Evaluator) |
| Tool set | calculate / remember / recall | + `write_artifact` / `read_artifact` / `list_artifacts` / `search_memory` |
| Evaluator | 1–5 single score | **4-dimensional rubric** (completeness / correctness / constraint adherence / quality) |
| Session continuity | None | **Handoff artifacts** — resume across sessions |
| Quality enforcement | Post-hoc scoring | **Constraint checker** — dedicated "find-problems" agent + correction loop |

## Patterns demonstrated

### A · Three-Agent Pipeline
```
Planner  →  Generator  →  Constraint Checker  →  Evaluator
                ↑                                     |
                └──────── retry if score < 3.5 ───────┘
```
The Planner decomposes a task into a JSON spec. The Generator uses tools to write structured artifacts. A dedicated Evaluator — never the Generator itself — scores quality across four dimensions and triggers a retry loop when needed.

### B · Multi-Session Handoff
Progress is written to `artifacts/handoffs/progress_{session_id}.json` after each component completes. On the next run, the session is resumed: completed components are skipped, and work picks up exactly where it left off. Eliminates redundant LLM calls and demonstrates cross-session memory.

### C · Constraint Enforcement Loop
`CONSTRAINTS.md` defines five architectural rules (security, interface definition, error handling, scalability, documentation). A dedicated constraint-checker agent — designed specifically to find problems — scans every artifact after generation. Violations trigger a targeted correction pass, then re-verification.

## File structure

```
app2/
├── AGENTS.md              # Generator standing instructions
├── PLANNER.md             # Planner JSON-output instructions
├── CONSTRAINTS.md         # Architectural rules enforced by checker
├── main.py                # Orchestrates all three patterns
├── harness/
│   ├── config.py          # Provider / model config (reused from app1)
│   ├── provider.py        # Anthropic + OpenAI abstraction (reused from app1)
│   ├── memory.py          # JSON session memory (reused from app1)
│   ├── tools.py           # Extended tool set (new: artifact I/O, search_memory)
│   ├── planner.py         # Planner agent — task → JSON plan
│   ├── generator.py       # Generator agent — plan component → artifact
│   ├── evaluator.py       # Evaluator — 4-dimensional rubric scoring
│   ├── constraint_checker.py  # Checker — flags architectural violations
│   └── handoff.py         # Progress save/resume across sessions
├── artifacts/             # Generated artifacts (git-ignored)
└── memory/                # Session memory store (git-ignored)
```

## Setup

```bash
cd app2
cp .env.example .env      # add your API key
uv sync                   # or: pip install -e .
uv run python main.py
```

Configure provider in `.env`:

```
PROVIDER=anthropic          # or openai
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

## Output

Running `main.py` executes all three patterns sequentially and prints live progress:

```
══════════════════════════════════════════════════════════════
  Advanced Harness Engineering Demo  (app2)
  Provider : anthropic
  Model    : claude-haiku-4-5-20251001
══════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────
  A. Three-Agent Pipeline (Planner → Generator → Evaluator loop)
  [Planner] Decomposing task...
  [Generator] Implementing (attempt 1)...
    Tool : list_artifacts()
    Tool : write_artifact(name='rate_limit_algorithm', ...)
  [Constraint Checker] Scanning artifacts...
  [Evaluator] Scoring with rubric...
  Completeness      : 4/5
  Correctness       : 5/5
  Constraint Adh.   : 4/5
  Quality           : 4/5
  Overall           : 4.2/5
```

Artifacts are saved to `artifacts/` as markdown files for inspection.
