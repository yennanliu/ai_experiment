"""
Advanced RAG Example

Demonstrates:
- Document chunking with overlap
- Metadata filtering
- Persistent vector store
- Multi-query retrieval (query expansion)
- Source attribution in responses
"""

import os
import sys
import traceback

import chromadb
from openai import OpenAI

# Sample documents with metadata
DOCUMENTS = [
    {
        "text": """Python's asyncio module provides infrastructure for writing
        single-threaded concurrent code using coroutines. It uses async/await
        syntax introduced in Python 3.5. The event loop is the core of asyncio,
        managing and distributing the execution of different tasks.""",
        "source": "python_async.md",
        "topic": "async",
    },
    {
        "text": """FastAPI is a modern, fast web framework for building APIs with
        Python 3.7+ based on standard Python type hints. It's built on Starlette
        for web parts and Pydantic for data validation. FastAPI automatically
        generates OpenAPI documentation.""",
        "source": "fastapi_intro.md",
        "topic": "web",
    },
    {
        "text": """Python's GIL (Global Interpreter Lock) is a mutex that protects
        access to Python objects, preventing multiple threads from executing
        Python bytecodes at once. This makes threading less effective for
        CPU-bound tasks but fine for I/O-bound operations.""",
        "source": "python_threading.md",
        "topic": "async",
    },
    {
        "text": """Django is a high-level Python web framework that encourages
        rapid development and clean, pragmatic design. It follows the
        model-template-view (MTV) architectural pattern and includes an ORM,
        authentication, and admin interface out of the box.""",
        "source": "django_intro.md",
        "topic": "web",
    },
]


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


def check_api_key():
    """Check if OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        log("ERROR: OPENAI_API_KEY environment variable not set.")
        log("Run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def index_documents(collection, documents: list[dict], chunk_size: int = 100) -> int:
    """Index documents with chunking and metadata (simplified single comprehension)."""
    log("       Preparing chunks...")
    data = [
        (f"doc{di}_chunk{ci}", chunk, {"source": doc["source"], "topic": doc["topic"], "chunk_idx": ci})
        for di, doc in enumerate(documents)
        for ci, chunk in enumerate(chunk_text(doc["text"], chunk_size=chunk_size))
    ]

    if not data:
        return 0

    ids, chunks, metadatas = zip(*data)
    log(f"       Upserting {len(data)} chunks (embedding may download model on first run)...")
    collection.upsert(documents=list(chunks), ids=list(ids), metadatas=list(metadatas))
    log("       Upsert complete.")
    return len(data)


def expand_query(client, question: str) -> list[str]:
    """Generate multiple query variations for better retrieval."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate 2 alternative phrasings. One per line."},
            {"role": "user", "content": question},
        ],
    )
    variations = response.choices[0].message.content.strip().split("\n")
    return [question] + [v.strip() for v in variations if v.strip()]


def retrieve_with_filter(
    collection, queries: list[str], n_results: int = 3, topic_filter: str | None = None
) -> list[dict]:
    """Retrieve documents with optional metadata filtering."""
    where_filter = {"topic": topic_filter} if topic_filter else None
    seen_ids = set()
    all_results = []

    for query in queries:
        results = collection.query(query_texts=[query], n_results=n_results, where=where_filter)
        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

    return sorted(all_results, key=lambda x: x["distance"])[:n_results]


def query_with_sources(
    client, collection, question: str, use_query_expansion: bool = True, topic_filter: str | None = None
) -> dict:
    """RAG query with source attribution."""
    queries = expand_query(client, question) if use_query_expansion else [question]
    results = retrieve_with_filter(collection, queries, n_results=3, topic_filter=topic_filter)

    if not results:
        return {"answer": "No relevant documents found.", "sources": [], "queries_used": queries}

    context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(results))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on context. Cite sources using [1], [2], etc. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [{"source": r["metadata"]["source"], "topic": r["metadata"]["topic"]} for r in results],
        "queries_used": queries,
    }


def main():
    log("=== Advanced RAG Example ===\n")
    check_api_key()

    log("[1/5] Creating OpenAI client...")
    client = OpenAI()

    log("[2/5] Creating ChromaDB persistent client...")
    db = chromadb.PersistentClient(path="./chroma_db")

    log("[3/5] Getting/creating collection...")
    collection = db.get_or_create_collection("advanced_docs")

    log("[4/5] Indexing documents...")
    num_chunks = index_documents(collection, DOCUMENTS, chunk_size=50)
    log(f"       Indexed {len(DOCUMENTS)} documents into {num_chunks} chunks\n")

    # Query 1: General question
    log("--- Query 1: General question ---")
    result = query_with_sources(client, collection, "How does Python handle concurrency?")
    log(f"Q: How does Python handle concurrency?")
    log(f"A: {result['answer']}")
    log(f"Sources: {[s['source'] for s in result['sources']]}")
    log(f"Query expansion: {result['queries_used']}\n")

    # Query 2: Filtered by topic
    log("--- Query 2: Filtered by topic ---")
    result = query_with_sources(client, collection, "What web frameworks are available?", topic_filter="web")
    log(f"Q: What web frameworks are available? (topic=web)")
    log(f"A: {result['answer']}")
    log(f"Sources: {[s['source'] for s in result['sources']]}\n")

    # Query 3: Without query expansion
    log("--- Query 3: Without query expansion ---")
    result = query_with_sources(client, collection, "What is the GIL?", use_query_expansion=False)
    log(f"Q: What is the GIL?")
    log(f"A: {result['answer']}")
    log(f"Sources: {[s['source'] for s in result['sources']]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        log(f"\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
