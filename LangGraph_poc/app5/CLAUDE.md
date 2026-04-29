# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the server (http://localhost:5001)
uv run python main.py
```

No test suite or linter is configured.

## Architecture

Flask web app (`main.py`) exposing three main routes:
- `POST /upload` — ingests an uploaded file into a named ChromaDB collection
- `POST /load-samples` — ingests `data/*.txt` into the `samples` collection
- `POST /chat` — runs the LangGraph RAG pipeline and returns a JSON response

### LangGraph Pipeline (`agent/graph.py`)

Linear graph: `transform → retrieve → rerank → generate`

- **transform**: Applies query transformation based on mode (`none` / `hyde` / `multi_query`). HyDE generates a hypothetical answer and embeds it; multi-query rewrites the question into 3 variants.
- **retrieve**: Queries ChromaDB using transformed queries or embeddings; deduplicates chunks.
- **rerank**: Optionally scores each chunk 0–10 via LLM and re-sorts, keeping top-K.
- **generate**: Calls `gpt-4o-mini` with retrieved context to produce the final answer.

State is typed as `RAGState` (`agent/state.py`), a `TypedDict` that flows through all nodes.

### RAG Modules (`rag/`)

- `ingest.py` — `ingest_file()` (chunk → embed → upsert to ChromaDB), `retrieve()` (text query), `retrieve_by_embedding()` (vector query)
- `chunkers.py` — three strategies: `char` (500-char sliding window), `sentence` (N-sentence groups), `paragraph` (blank-line split with short-paragraph merging)
- `query_transform.py` — `hyde_embedding()` and `multi_query_expand()` using `text-embedding-3-small` + `gpt-4o-mini`
- `rerank.py` — LLM-based relevance scoring

### Config (`.env`)

```
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads
```

ChromaDB persists to `./chroma_db/`. Each ingested collection is named by the user at upload time (default `samples` for built-in docs).
