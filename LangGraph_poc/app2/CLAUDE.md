# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (requires uv)
uv sync

# Run development server
uv run uvicorn main:app --reload

# Seed ChromaDB with example email→reply pairs
uv run python seed_rag.py

# Re-seed (wipe and re-seed)
uv run python seed_rag.py --reset

# Run smoke tests against the running server
uv run python test_mock.py
uv run python test_mock.py --url http://localhost:8000
```

## Architecture

This is a FastAPI + LangGraph service for B2B email reply generation. The core flow is:

```
POST /generate-draft
  → LangGraph pipeline (agent/graph.py):
      classify → select_template → retrieve_examples → generate_draft → build_checklist → score_draft
  → JSON response
```

**Key files:**
- `main.py` — FastAPI app; history is stored in-memory (lost on restart)
- `agent/state.py` — `AgentState` TypedDict flows through all graph nodes
- `agent/nodes.py` — 6 LangGraph node functions; all call GPT-4o via `openai.OpenAI()`
- `agent/templates.py` — Per-type reply templates and `RISK_KEYWORDS` list
- `agent/graph.py` — Wires nodes into a linear `StateGraph`; exports `email_agent`
- `agent/retriever.py` — ChromaDB RAG layer; persists to `./chroma_db/`; uses `text-embedding-3-small`
- `seed_rag.py` — Seeds ChromaDB with hardcoded historical examples
- `mock_data.py` — Sample emails exposed via `GET /mock-emails`
- `static/` — Vanilla HTML UI (index, history, templates pages)

**Email types:** `New PO`, `Change Order`, `Wrong Order / Quality Issue`, `Inventory Inquiry`, `Shipment Inquiry`, `New Quote`, `Other`

**RAG behavior:** `retrieve_examples` queries ChromaDB for top-3 similar historical emails, preferring same email type. Falls back to global search if no same-type results. Examples are injected as few-shot context into the `generate_draft` prompt.

**Config:** Requires `OPENAI_API_KEY` in `.env`.
