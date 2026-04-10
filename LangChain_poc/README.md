# LangChain PoC

A simple interactive chat app using LangChain and OpenAI.

## Setup

```bash
uv sync
# Add your OpenAI API key to .env
```

## Run Examples

```bash
# Simple chat
uv run main.py chat

# Multi-step: research a topic → structured summary
uv run main.py multi_step
```

## Structure

```
core.py              — shared LLM + chain helpers
main.py              — entry point, picks example by name
examples/
  chat.py            — basic conversation
  multi_step.py      — research → summary pipeline
```

Add new examples by creating a file in `examples/` and registering it in `main.py`.
