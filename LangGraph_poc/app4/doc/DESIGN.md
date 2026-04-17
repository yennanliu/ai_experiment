# PrivateAssist — Local-First Private Agent

## Overview

A privacy-first document Q&A assistant that runs **100% on-premise**. No data leaves the machine. Designed for companies in finance, healthcare, and legal where sending data to external APIs is not permitted.

Users upload internal documents, ask questions in natural language, and get grounded answers with source citations — all powered by a local LLM via Ollama.

---

## Problem Statement

Most AI assistants rely on cloud APIs (OpenAI, Anthropic, etc.). This creates:
- **Data privacy risk** — sensitive docs sent to third-party servers
- **Compliance issues** — violates HIPAA, GDPR, SOC2 policies
- **Network dependency** — doesn't work air-gapped

This app solves all three by keeping inference, embeddings, and data storage fully local.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Local LLM | Ollama (`llama3.2`) | Easy local inference, streaming support |
| Embeddings | Ollama (`nomic-embed-text`) | Local embeddings, no OpenAI needed |
| Vector DB | ChromaDB (persistent) | Lightweight, local, no server needed |
| Agent Orchestration | LangGraph StateGraph | Consistent with app1-3 in this series |
| Backend | FastAPI + SSE | Async streaming, same pattern as prior apps |
| Frontend | Vanilla HTML/JS | No build step, consistent with app series |
| Package Manager | uv | Fast, consistent with app series |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Browser UI                       │
│  [Upload Panel]  [Collection Selector]  [Chat Panel] │
└──────────────┬──────────────────────────┬────────────┘
               │ POST /upload             │ POST /chat (SSE)
               ▼                          ▼
┌─────────────────────────────────────────────────────┐
│                    FastAPI Backend                    │
│                                                      │
│  /upload → Ingest Pipeline                           │
│  /chat   → LangGraph Agent                           │
│  /collections, /models, /health                      │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────┐
│  Ingest Pipeline │   │     LangGraph RAG Agent       │
│                  │   │                               │
│  1. Parse doc    │   │  retrieve → grade → generate  │
│  2. Chunk text   │   │                               │
│  3. Embed chunks │   │  State: question, documents,  │
│  4. Store Chroma │   │  context, answer, sources     │
└────────┬─────────┘   └──────────────┬────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐        ┌──────────────────────────┐
│    ChromaDB     │◄───────│       Ollama (local)      │
│  (local disk)   │        │                           │
│                 │        │  - nomic-embed-text        │
│  Collections:   │        │  - llama3.2 (chat)        │
│  per user group │        │  - mistral (optional)     │
└─────────────────┘        └──────────────────────────┘
```

---

## LangGraph Agent Flow

```
[START]
   │
   ▼
[retrieve]  — embed question → query ChromaDB → get top-5 chunks
   │
   ▼
[grade]     — LLM judges each chunk: relevant? yes/no → filter
   │
   ▼
[generate]  — build RAG prompt with context → call Ollama → stream answer
   │
   ▼
[END]
```

### AgentState

```python
class AgentState(TypedDict):
    question: str          # User's question
    collection: str        # Which doc collection to search
    documents: List[Document]  # Retrieved chunks
    context: str           # Joined context string
    answer: str            # Final answer
    sources: List[str]     # Source filenames cited
    step: str              # Current pipeline step (for SSE progress)
    grounded: Optional[bool]   # Hallucination check result
```

---

## File Structure

```
app4/
├── main.py                  # FastAPI app entry point
├── agent/
│   ├── __init__.py
│   ├── state.py             # AgentState TypedDict
│   ├── nodes.py             # retrieve, grade, generate node functions
│   └── graph.py             # LangGraph StateGraph wiring
├── rag/
│   ├── __init__.py
│   ├── ingest.py            # Doc parsing, chunking, embedding, ChromaDB write
│   ├── retriever.py         # ChromaDB query wrapper
│   └── embeddings.py        # Ollama embeddings (LangChain Embeddings interface)
├── models/
│   ├── __init__.py
│   └── ollama_client.py     # Ollama HTTP API: embed, chat_stream, list_models
├── static/
│   └── index.html           # Single-page UI
├── chroma_db/               # Persisted vector store (gitignored)
├── uploads/                 # Temp upload storage (gitignored)
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Ollama connection status |
| `GET` | `/models` | List locally available Ollama models |
| `GET` | `/collections` | List document collections |
| `DELETE` | `/collections/{name}` | Delete a collection + its embeddings |
| `POST` | `/upload` | Upload file, ingest into named collection |
| `POST` | `/chat` | SSE stream: question + collection → answer |

### `/upload` Request
```
multipart/form-data:
  file: <PDF, TXT, or MD file>
  collection: <string, collection name>
```

### `/chat` Request
```json
{
  "question": "What is the refund policy?",
  "collection": "HR Docs",
  "model": "llama3.2"
}
```

### `/chat` SSE Response Stream
```
data: {"step": "retrieved", "message": "Found 5 relevant chunks"}
data: {"step": "graded", "message": "3 chunks passed relevance check"}
data: {"step": "generating", "token": "The refund policy states..."}
data: {"step": "done", "sources": ["policy.pdf"], "answer": "...full answer..."}
```

---

## UI Design

```
┌──────────────────────────────────────────────────────────────┐
│  🔒 PrivateAssist          [Model: llama3.2 ▼]  ● Local Only │
├──────────────────┬───────────────────────────────────────────┤
│                  │                                            │
│  Collections     │   HR Docs ▼                               │
│  ─────────────   │   ┌────────────────────────────────────┐  │
│  + HR Docs    ✓  │   │ What is the vacation policy?       │  │
│    Legal         │   └────────────────────────────────────┘  │
│    Tech Specs    │                              [Ask]         │
│                  │                                            │
│  Upload          │   ─────────────────────────────────────   │
│  ┌────────────┐  │   Assistant (sources: handbook.pdf)        │
│  │ Drop files │  │                                            │
│  │ here  📄   │  │   Employees are entitled to 15 days of    │
│  └────────────┘  │   paid vacation per year...               │
│  [Add to: HR ▼]  │                                            │
│  [Upload]        │                                            │
└──────────────────┴───────────────────────────────────────────┘
```

**Key UI elements:**
- Privacy badge ("● Local Only") — always visible, reassures users
- Collection sidebar — organize docs by department/topic
- Model selector — switch between installed Ollama models
- Source citation — shows which file(s) the answer came from
- Upload panel — drag-and-drop, assign to collection on upload

---

## Prerequisites

Users must install Ollama and pull models before running:

```bash
# Install Ollama
brew install ollama        # macOS
# or: https://ollama.com/download

# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text

# Start Ollama (runs as background service)
ollama serve
```

---

## Privacy Guarantee

- All inference runs via `localhost:11434` (Ollama)
- ChromaDB persists to local disk only
- No telemetry, no external HTTP calls during Q&A
- Upload files stored temporarily in `./uploads/`, can be deleted after ingest
- Can run fully air-gapped after initial model download

---

## Future Enhancements

- Conversation history / multi-turn chat
- Re-ranking retrieved chunks (cross-encoder)
- Hallucination detection node (verify answer grounded in context)
- Multi-file collection bulk upload
- Export chat logs to PDF
- Auth layer for multi-user deployments
