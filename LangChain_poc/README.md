# LangChain PoC

Interactive examples using LangChain + LangGraph + OpenAI.

## Setup

```bash
uv sync
# Add your OpenAI API key to .env
```

## Run Examples

```bash
uv run main.py chat          # Simple chat
uv run main.py multi_step    # Research → structured summary
uv run main.py router        # Auto-routes to code/math/general expert
uv run main.py critique      # Generate → critique → revise loop
uv run main.py parallel      # Parallel text analysis → merged report
```

Each example prints its workflow graph on startup.

## Structure

```
core.py              — shared helpers (LLM, chain builder, graph printer)
main.py              — entry point, picks example by name
examples/
  chat.py            — basic conversation (LangChain)
  multi_step.py      — two-step pipeline (LangGraph)
  router.py          — conditional routing by intent (LangGraph)
  critique.py        — cyclic generate/critique/revise (LangGraph)
  parallel.py        — fan-out analysis → fan-in merge (LangGraph)
```

Add new examples by dropping a `.py` file in `examples/` — auto-discovered.
