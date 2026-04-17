# PrivateAssist — Local-First RAG Agent

A privacy-first document Q&A assistant. Upload internal docs, ask questions in natural language, get grounded answers with source citations — **100% on-premise, zero data sent to the cloud**.

Built for teams in finance, healthcare, and legal where sending data to external APIs is not an option.

---

## Installation

### 1. Install Ollama

```bash
brew install ollama        # macOS
# or download from https://ollama.com/download
```

### 2. Pull the models

```bash
ollama pull llama3.2:1b        # ~1.3GB — smallest capable chat model
ollama pull nomic-embed-text   # ~274MB — local embeddings
```

### 3. Clone and install dependencies

```bash
git clone <repo>
cd app4
uv sync
```

> Requires [uv](https://docs.astral.sh/uv/). Install with `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.

### 4. Configure environment

```bash
cp .env.example .env
```

Default `.env` works out of the box. Override models or paths as needed:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:1b
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads
```

---

## Run

```bash
ollama serve                                    # start Ollama in background
uv run uvicorn main:app --reload --port 8000    # start app
```

Open **http://localhost:8000**

---

## Architecture

```
Browser UI
  │
  ├── POST /upload  ──► Ingest Pipeline
  │                       1. Parse  (PDF / TXT / MD)
  │                       2. Chunk  (500 words, 50 overlap)
  │                       3. Embed  (Ollama nomic-embed-text)
  │                       4. Store  (ChromaDB local disk)
  │
  └── POST /chat   ──► LangGraph Agent  (SSE stream)
                          │
                          ├── [retrieve]  embed question → top-4 chunks from ChromaDB
                          └── [generate]  RAG prompt → Ollama llama3.2:1b → stream tokens
```

**Stack**

| Layer | Technology |
|---|---|
| LLM inference | Ollama (`llama3.2:1b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector store | ChromaDB (persisted to disk) |
| Agent orchestration | LangGraph StateGraph |
| Backend | FastAPI + Server-Sent Events |
| Frontend | Vanilla HTML/JS |
| Package manager | uv |

All components run locally. No telemetry, no external HTTP calls during inference.

---

## Features & Ideas

### What's in here

- **Upload panel** — drag-and-drop PDF, TXT, or MD files into named collections
- **Collections** — organize documents by topic (e.g. "HR Docs", "Legal", "Tech Specs")
- **Streaming answers** — tokens stream in real-time via SSE as the model generates
- **Source citations** — every answer shows which file(s) it was drawn from
- **Model selector** — switch between any locally installed Ollama model
- **Privacy badge** — "Local Only" indicator always visible in the header
- **Health check** — `/health` endpoint reports Ollama connection status

### Ideas for next steps

- **Multi-turn chat** — maintain conversation history per session
- **Re-ranking** — add a cross-encoder pass to improve chunk relevance before generation
- **Hallucination check** — LLM-as-judge node to verify the answer is grounded in context
- **Bulk upload** — ingest an entire folder at once
- **Larger models** — swap to `llama3.2:3b` or `mistral` for better reasoning quality
- **Auth layer** — basic login for multi-user team deployments
- **Export** — download chat transcripts as PDF or Markdown
