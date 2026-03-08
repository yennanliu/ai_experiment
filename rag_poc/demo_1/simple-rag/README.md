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
├── script_1.py      # Basic RAG (~80 lines)
├── script_2.py      # Advanced RAG (~200 lines)
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

**Example output:**
```
Q: How does Python handle concurrency?
A: Python handles concurrency through asyncio [1] and threading,
   though the GIL limits CPU-bound parallelism [2].
Sources: ['python_async.md', 'python_threading.md']
```

---

## Comparison

| Aspect | script_1 | script_2 |
|--------|----------|----------|
| Storage | In-memory | Persistent |
| Chunking | No | Yes (with overlap) |
| Metadata | No | Yes (filtering) |
| Query expansion | No | Yes |
| Source citations | No | Yes |
| Lines of code | ~80 | ~200 |

## Dependencies

- `chromadb` - Vector store with built-in embeddings
- `openai` - LLM for generation

## Next Steps

- Add reranking (e.g., Cohere rerank, cross-encoder)
- Implement hybrid search (keyword + semantic)
- Add streaming responses
- Use custom embedding models
