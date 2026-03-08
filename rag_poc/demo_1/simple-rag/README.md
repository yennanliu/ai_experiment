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
├── script_5.py      # HyDE RAG (~230 lines)
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

### script_5.py - HyDE RAG

Hypothetical Document Embeddings for better semantic matching.

```bash
uv run script_5.py
```

**Features:**

| Feature | Description |
|---------|-------------|
| **Hypothetical generation** | LLM generates a hypothetical answer first |
| **Better embedding match** | Hypothetical doc is closer to real docs in vector space |
| **Comparison mode** | Compare standard vs HyDE retrieval side-by-side |
| **Abstract query handling** | Excels when user vocabulary differs from documents |

**How HyDE works:**
```
Standard: "How do I keep data fresh?" → embed query → search
HyDE:     "How do I keep data fresh?" → generate hypothetical answer
          → embed hypothetical → search (better match!)
```

**Example comparison:**
```
Q: How do I keep my data fresh?

Standard retrieval: [scale-1, perf-1, cache-2] (avg dist: 1.42)
HyDE retrieval:     [cache-1, cache-2, perf-2] (avg dist: 1.18)
                    ↑ Better! Found cache invalidation docs
```

---

## Comparison

| Aspect | script_1 | script_2 | script_3 | script_4 | script_5 |
|--------|----------|----------|----------|----------|----------|
| Storage | In-memory | Persistent | Persistent | Persistent | Persistent |
| Chunking | No | Yes | No | No | No |
| Metadata filtering | No | Yes | Yes | Yes | No |
| Query expansion | No | Yes | No | No | No |
| Reranking | No | No | Yes (LLM) | No | No |
| Conversation memory | No | No | Yes | No | No |
| Streaming | No | No | Yes | No | No |
| Function calling | No | No | No | Yes | No |
| Self-reflection | No | No | No | Yes | No |
| HyDE | No | No | No | No | Yes |
| Lines of code | ~90 | ~200 | ~220 | ~230 | ~230 |

## Dependencies

- `chromadb` - Vector store with built-in embeddings
- `openai` - LLM for generation

## Architecture Progression

```
script_1: Query → Retrieve → Generate
script_2: Query → Expand → Retrieve+Filter → Generate+Cite
script_3: Query → Retrieve → Rerank → Generate+Stream (with history)
script_4: Query → LLM decides → [Retrieve? → Evaluate?]* → Generate
script_5: Query → Generate hypothetical → Retrieve (with hypo) → Generate
```
