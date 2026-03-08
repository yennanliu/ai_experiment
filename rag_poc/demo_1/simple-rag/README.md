# Simple RAG Example

A minimal Retrieval-Augmented Generation (RAG) implementation in Python.

## What is RAG?

RAG combines retrieval and generation to answer questions:

```
Query → Retrieve relevant docs → Generate answer with context
```

This approach grounds LLM responses in your data, reducing hallucinations.

## Project Structure

```
simple-rag/
├── main.py          # RAG implementation
├── pyproject.toml   # Project config
└── README.md
```

## Setup

```bash
# Requires uv (https://docs.astral.sh/uv)
uv sync

# Set OpenAI API key
export OPENAI_API_KEY="your-key"
```

## Run

```bash
uv run main.py
```

## How It Works

### 1. Index Documents
Documents are embedded and stored in ChromaDB (in-memory vector store):

```python
collection.add(documents=["doc1", "doc2"], ids=["id1", "id2"])
```

### 2. Retrieve
Find relevant documents using semantic similarity:

```python
results = collection.query(query_texts=["question"], n_results=2)
```

### 3. Generate
Pass retrieved context to the LLM:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer based on context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ],
)
```

## Dependencies

- `chromadb` - Vector store with built-in embeddings
- `openai` - LLM for answer generation

## Key Concepts

| Component | Purpose |
|-----------|---------|
| Vector Store | Store and search document embeddings |
| Embeddings | Convert text to numerical vectors |
| Retriever | Find relevant documents for a query |
| Generator | LLM that produces the final answer |

## Extending This Example

- **Persistent storage**: Use `chromadb.PersistentClient(path="./db")`
- **Custom embeddings**: Pass `embedding_function` to collection
- **Chunking**: Split large documents before indexing
- **Reranking**: Add a reranker to improve retrieval quality
