# RAG Examples

Retrieval-Augmented Generation (RAG) implementations in Python.

## What is RAG?

```
Query → Retrieve relevant docs → Generate answer with context
```

RAG grounds LLM responses in your data, reducing hallucinations.

## Project Structure

```
simple-rag/
├── script_1.py      # Basic RAG (~90 lines)
├── script_2.py      # Advanced RAG (~200 lines)
├── script_3.py      # Conversational RAG (~220 lines)
├── script_4.py      # Agentic RAG (~230 lines)
├── pyproject.toml   # Dependencies
└── README.md
```

## Setup

```bash
uv sync
export OPENAI_API_KEY="your-key"
```

## Scripts

### script_1.py - Basic RAG

Minimal implementation demonstrating the core pattern.

```bash
uv run script_1.py
```

**Features:**
- In-memory vector store
- Simple document indexing
- Basic retrieval + generation

---

### script_2.py - Advanced RAG

Production-ready patterns for better retrieval quality.

```bash
uv run script_2.py
```

**Features:**

| Feature | Description |
|---------|-------------|
| **Chunking** | Split documents with overlap for better context |
| **Metadata** | Filter by topic, source, or custom attributes |
| **Persistent storage** | Data survives restarts (`./chroma_db`) |
| **Query expansion** | Generate query variations for better recall |
| **Source attribution** | Citations in responses with `[1]`, `[2]` |

---

### script_3.py - Conversational RAG

Multi-turn conversation with memory and reranking.

```bash
uv run script_3.py
```

**Features:**

| Feature | Description |
|---------|-------------|
| **Conversation memory** | Maintains context across multiple turns |
| **LLM reranking** | Uses LLM to reorder results by relevance |
| **Streaming responses** | Real-time token-by-token output |
| **Follow-up questions** | Understands "it", "that", etc. from history |

**Example conversation:**
```
--- Turn 1 ---
Q: What databases are good for caching?
A: Redis is ideal for caching - it's an in-memory data store...

--- Turn 2 ---
Q: How does it compare to PostgreSQL?
A: Redis and PostgreSQL serve different purposes. Redis excels at
   caching with sub-millisecond latency, while PostgreSQL is a
   full relational database with ACID compliance...
```

---

### script_4.py - Agentic RAG

LLM-driven retrieval decisions using function calling.

```bash
uv run script_4.py
```

**Features:**

| Feature | Description |
|---------|-------------|
| **Function calling** | LLM decides when to search via tool use |
| **Self-reflection** | Evaluates document relevance before answering |
| **Multi-step reasoning** | Can refine searches based on results |
| **Smart routing** | Skips retrieval for simple questions |

**Example behavior:**
```
Q: Hello, how are you?
A: Hello! I'm doing well...
   [Used retrieval: False]

Q: What are Python decorators?
  [Tool] search_knowledge_base({"query": "Python decorators"})
  [Eval] Relevant: True, Confidence: 0.95
A: Python decorators are functions that modify other functions...
   [Used retrieval: True]
```

---

## Comparison

| Aspect | script_1 | script_2 | script_3 | script_4 |
|--------|----------|----------|----------|----------|
| Storage | In-memory | Persistent | Persistent | Persistent |
| Chunking | No | Yes | No | No |
| Metadata filtering | No | Yes | Yes | Yes |
| Query expansion | No | Yes | No | No |
| Reranking | No | No | Yes (LLM) | No |
| Conversation memory | No | No | Yes | No |
| Streaming | No | No | Yes | No |
| Function calling | No | No | No | Yes |
| Self-reflection | No | No | No | Yes |
| Lines of code | ~90 | ~200 | ~220 | ~230 |

## Dependencies

- `chromadb` - Vector store with built-in embeddings
- `openai` - LLM for generation

## Architecture Progression

```
script_1: Query → Retrieve → Generate
script_2: Query → Expand → Retrieve+Filter → Generate+Cite
script_3: Query → Retrieve → Rerank → Generate+Stream (with history)
script_4: Query → LLM decides → [Retrieve? → Evaluate?]* → Generate
```
