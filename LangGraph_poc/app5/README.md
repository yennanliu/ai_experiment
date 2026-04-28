# Enterprise Knowledge Base — RAG with LangGraph + OpenAI

A simple enterprise-grade document Q&A system. Upload internal docs, ask questions, get grounded answers with source citations.

---

## Architecture

```
User
 │
 ▼
Flask (main.py)
 ├── POST /upload   ──► RAG Ingest (rag/ingest.py)
 │                        load → chunk → embed → ChromaDB
 │
 └── POST /chat    ──► LangGraph Pipeline (agent/graph.py)
                          │
                    ┌─────▼──────┐
                    │  retrieve  │  embed query → top-5 chunks from ChromaDB
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │  generate  │  gpt-4o-mini answers using only retrieved context
                    └─────┬──────┘
                          │
                    JSON response (answer + sources)
```

**Key components:**

| Layer | Tech | Role |
|---|---|---|
| Web | Flask | HTTP API + static UI |
| Orchestration | LangGraph | RAG pipeline as a state graph |
| LLM | OpenAI `gpt-4o-mini` | Answer generation |
| Embeddings | OpenAI `text-embedding-3-small` | Chunk & query vectorization |
| Vector store | ChromaDB (persistent) | Similarity search |
| Chunking | Sliding window (500 chars, 50 overlap) | Text splitting |

---

## Concepts

### RAG (Retrieval-Augmented Generation)
Instead of relying on an LLM's training data, RAG fetches relevant snippets from *your* documents at query time and feeds them as context. This keeps answers grounded, current, and auditable.

### Chunking
Documents are split into overlapping character windows (500 chars, 50 char overlap). Overlap prevents a sentence from being cut off at a chunk boundary, preserving coherence across retrieval.

```
[  chunk 1  ]
          [  chunk 2  ]
                    [  chunk 3  ]
```

### LangGraph Pipeline
The RAG flow is modeled as a `StateGraph` with two nodes:

```
retrieve → generate → END
```

Each node is a pure function that reads and updates a typed state dict (`RAGState`). This makes the pipeline easy to extend (e.g. add a `rerank` or `guardrails` node) without touching existing logic.

### Collections
Documents are organized into named collections (backed by ChromaDB). Each collection is an isolated vector namespace — useful for separating e.g. "HR policies" from "Engineering specs".

---

## Install

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
cd app5
uv sync
```

Set your OpenAI API key in `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads
```

---

## Run

```bash
uv run python main.py
```

Open [http://localhost:5000](http://localhost:5000).

---

## Usage

1. **Create a collection** — type a name in the sidebar (e.g. `hr-policies`)
2. **Upload a document** — PDF, TXT, or MD; chunks are embedded and stored
3. **Ask a question** — the LangGraph pipeline retrieves relevant chunks and generates a grounded answer
4. **Check sources** — each answer shows which documents were used

---

## Project Structure

```
app5/
├── main.py          # Flask routes: /upload /chat /collections
├── rag/
│   └── ingest.py    # load → chunk → embed → ChromaDB  +  retrieve()
├── agent/
│   ├── state.py     # RAGState TypedDict
│   └── graph.py     # LangGraph graph: retrieve → generate
├── static/
│   └── index.html   # Single-page UI
├── .env             # API keys and config
└── pyproject.toml   # uv dependencies
```
