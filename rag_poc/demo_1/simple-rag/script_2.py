"""
Advanced RAG Example

Demonstrates:
- Document chunking with overlap
- Metadata filtering
- Persistent vector store
- Multi-query retrieval (query expansion)
- Source attribution in responses
"""

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


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


def create_persistent_rag(db_path: str = "./chroma_db"):
    """Create RAG with persistent storage."""
    client = OpenAI()
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_or_create_collection("advanced_docs")
    return client, collection


def index_documents(collection, documents: list[dict], chunk_size: int = 100):
    """Index documents with chunking and metadata."""
    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc["text"], chunk_size=chunk_size)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"doc{doc_idx}_chunk{chunk_idx}")
            all_metadata.append({
                "source": doc["source"],
                "topic": doc["topic"],
                "chunk_idx": chunk_idx,
            })

    # Upsert to handle re-indexing
    collection.upsert(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadata,
    )
    return len(all_chunks)


def expand_query(client, question: str) -> list[str]:
    """Generate multiple query variations for better retrieval."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate 2 alternative phrasings of this question. "
                "Return only the questions, one per line.",
            },
            {"role": "user", "content": question},
        ],
    )
    variations = response.choices[0].message.content.strip().split("\n")
    return [question] + [v.strip() for v in variations if v.strip()]


def retrieve_with_filter(
    collection,
    queries: list[str],
    n_results: int = 3,
    topic_filter: str | None = None,
) -> list[dict]:
    """Retrieve documents with optional metadata filtering."""
    where_filter = {"topic": topic_filter} if topic_filter else None

    all_results = []
    seen_ids = set()

    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

    # Sort by relevance (lower distance = more relevant)
    return sorted(all_results, key=lambda x: x["distance"] or 0)[:n_results]


def query_with_sources(
    client,
    collection,
    question: str,
    use_query_expansion: bool = True,
    topic_filter: str | None = None,
) -> dict:
    """RAG query with source attribution."""
    # Expand query for better recall
    queries = expand_query(client, question) if use_query_expansion else [question]

    # Retrieve with optional filtering
    results = retrieve_with_filter(
        collection, queries, n_results=3, topic_filter=topic_filter
    )

    if not results:
        return {"answer": "No relevant documents found.", "sources": []}

    # Build context with source markers
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(f"[{i+1}] {r['text']}")

    context = "\n\n".join(context_parts)

    # Generate answer with citations
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer based on the provided context. "
                "Cite sources using [1], [2], etc. Be concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    sources = [{"source": r["metadata"]["source"], "topic": r["metadata"]["topic"]} for r in results]

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "queries_used": queries,
    }


def main():
    print("=== Advanced RAG Example ===\n")

    # Initialize with persistent storage
    client, collection = create_persistent_rag()

    # Index documents with chunking
    num_chunks = index_documents(collection, DOCUMENTS, chunk_size=50)
    print(f"Indexed {len(DOCUMENTS)} documents into {num_chunks} chunks\n")

    # Example 1: Basic query with sources
    print("--- Query 1: General question ---")
    result = query_with_sources(client, collection, "How does Python handle concurrency?")
    print(f"Q: How does Python handle concurrency?")
    print(f"A: {result['answer']}")
    print(f"Sources: {[s['source'] for s in result['sources']]}")
    print(f"Query expansion: {result['queries_used']}\n")

    # Example 2: Filtered query (only web topic)
    print("--- Query 2: Filtered by topic ---")
    result = query_with_sources(
        client,
        collection,
        "What web frameworks are available?",
        topic_filter="web",
    )
    print(f"Q: What web frameworks are available? (topic=web)")
    print(f"A: {result['answer']}")
    print(f"Sources: {[s['source'] for s in result['sources']]}\n")

    # Example 3: Without query expansion
    print("--- Query 3: Without query expansion ---")
    result = query_with_sources(
        client,
        collection,
        "What is the GIL?",
        use_query_expansion=False,
    )
    print(f"Q: What is the GIL?")
    print(f"A: {result['answer']}")
    print(f"Sources: {[s['source'] for s in result['sources']]}")


if __name__ == "__main__":
    main()
