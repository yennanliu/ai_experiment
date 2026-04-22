# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Running

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Run the demo playground
uv run python main.py
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. `COHERE_API_KEY` is optional (only needed if Demo 4 is switched to Cohere reranking).

## Architecture

`main.py` is an interactive CLI menu that dynamically imports and runs a demo module from `demos/`. Global LLM and embedding model settings are configured once in `main.py` via `Settings.llm` and `Settings.embed_model`.

Each file in `demos/` is a self-contained demo exposing a single `run()` function:

| Demo | File | What it shows |
|------|------|---------------|
| 1 | `sub_question.py` | `SubQuestionQueryEngine` — decomposes complex queries |
| 2 | `metadata_filter.py` | `MetadataFilters` — topic-based filtered retrieval |
| 3 | `react_agent.py` | `ReActAgent` — document search + calculator tools |
| 4 | `reranking.py` | `LLMRerank` — two-stage retrieval with reranking |
| 5 | `persistence.py` | `StorageContext` — save/load index to `./storage/` |
| 6 | `router.py` | `RouterQueryEngine` — auto-routes to vector vs summary index |

All demos read from `data/sample.txt`, which uses a pipe-delimited format: `doc_id | title | content`.

## Data Format

`data/sample.txt` lines follow: `<id> | <title> | <content>`. Demo 2 derives topic metadata from title matching against a predefined topic list.
