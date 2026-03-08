"""
HyDE RAG Example (Hypothetical Document Embeddings)

Demonstrates:
- Generate hypothetical answer before retrieval
- Use hypothetical answer for semantic search (better embedding match)
- Compare standard vs HyDE retrieval quality
- Works especially well for complex or abstract queries
"""

import os
import sys
import traceback

import chromadb
from openai import OpenAI


def log(msg: str):
    print(msg, flush=True)


def check_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        log("ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)


# Knowledge base - technical documentation style
DOCUMENTS = [
    {
        "id": "cache-1",
        "text": "Implementing a cache invalidation strategy requires understanding your data access patterns. Time-based expiration (TTL) works well for data that changes predictably. Event-driven invalidation is better for real-time consistency requirements.",
        "topic": "caching",
    },
    {
        "id": "cache-2",
        "text": "Cache-aside pattern: Application checks cache first, on miss reads from database and populates cache. Write-through pattern: Writes go to cache and database simultaneously. Write-behind pattern: Writes go to cache, async sync to database.",
        "topic": "caching",
    },
    {
        "id": "scale-1",
        "text": "Horizontal scaling adds more machines to handle load. Vertical scaling adds more resources to existing machines. Horizontal scaling provides better fault tolerance but requires stateless application design or distributed state management.",
        "topic": "scaling",
    },
    {
        "id": "scale-2",
        "text": "Database sharding partitions data across multiple databases. Common strategies: range-based (by ID ranges), hash-based (consistent hashing), and directory-based (lookup table). Sharding adds complexity but enables massive scale.",
        "topic": "scaling",
    },
    {
        "id": "auth-1",
        "text": "JWT (JSON Web Tokens) are self-contained tokens with encoded claims. They're stateless - no server-side session storage needed. Include expiration time, sign with secret key, and validate signature on each request.",
        "topic": "auth",
    },
    {
        "id": "auth-2",
        "text": "OAuth 2.0 authorization flow: User redirects to provider, grants permission, receives authorization code, exchanges code for access token. Refresh tokens allow obtaining new access tokens without re-authentication.",
        "topic": "auth",
    },
    {
        "id": "perf-1",
        "text": "N+1 query problem occurs when fetching related data requires N additional queries. Solutions: eager loading (JOIN), batch loading, or GraphQL DataLoader pattern. Always monitor query counts in development.",
        "topic": "performance",
    },
    {
        "id": "perf-2",
        "text": "Connection pooling reuses database connections instead of creating new ones. Configure pool size based on expected concurrency. Too small causes waiting; too large wastes resources and may hit database limits.",
        "topic": "performance",
    },
]


def create_rag(db_path: str = "./chroma_db_v5"):
    """Initialize RAG components."""
    client = OpenAI()
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_or_create_collection("hyde_docs")
    return client, collection


def index_documents(collection, documents: list[dict]) -> int:
    """Index documents into collection."""
    if not documents:
        return 0

    collection.upsert(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[{"topic": d["topic"]} for d in documents],
    )
    return len(documents)


def generate_hypothetical_answer(client, question: str) -> str:
    """Generate a hypothetical answer to use for retrieval."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a technical documentation writer. Given a question, write a
hypothetical paragraph that would answer it. Write as if you're explaining the concept
in documentation. Be specific and technical. This will be used for semantic search.""",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,  # Some creativity for diverse hypothetical answers
    )
    return response.choices[0].message.content


def retrieve_standard(collection, query: str, n_results: int = 3) -> list[dict]:
    """Standard retrieval using query directly."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"id": results["ids"][0][i], "text": results["documents"][0][i], "distance": results["distances"][0][i]}
        for i in range(len(results["ids"][0]))
    ]


def retrieve_hyde(collection, hypothetical_doc: str, n_results: int = 3) -> list[dict]:
    """HyDE retrieval using hypothetical document."""
    results = collection.query(query_texts=[hypothetical_doc], n_results=n_results)
    return [
        {"id": results["ids"][0][i], "text": results["documents"][0][i], "distance": results["distances"][0][i]}
        for i in range(len(results["ids"][0]))
    ]


def generate_answer(client, question: str, context: str) -> str:
    """Generate final answer from retrieved context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer based on the provided context. Be concise and accurate.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


class HyDERAG:
    """RAG with Hypothetical Document Embeddings."""

    def __init__(self, client, collection):
        self.client = client
        self.collection = collection

    def query(self, question: str, use_hyde: bool = True, verbose: bool = True) -> dict:
        """Process query with optional HyDE."""
        if use_hyde:
            # Step 1: Generate hypothetical answer
            hypothetical = generate_hypothetical_answer(self.client, question)
            if verbose:
                log(f"  [HyDE] Generated hypothetical ({len(hypothetical)} chars)")

            # Step 2: Retrieve using hypothetical
            docs = retrieve_hyde(self.collection, hypothetical, n_results=3)
        else:
            # Standard retrieval
            hypothetical = None
            docs = retrieve_standard(self.collection, question, n_results=3)

        # Step 3: Generate answer
        context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
        answer = generate_answer(self.client, question, context)

        return {
            "answer": answer,
            "hypothetical": hypothetical,
            "retrieved_docs": [d["id"] for d in docs],
            "distances": [round(d["distance"], 4) for d in docs],
        }

    def compare(self, question: str) -> dict:
        """Compare standard vs HyDE retrieval for the same question."""
        # Standard retrieval
        standard_docs = retrieve_standard(self.collection, question, n_results=3)

        # HyDE retrieval
        hypothetical = generate_hypothetical_answer(self.client, question)
        hyde_docs = retrieve_hyde(self.collection, hypothetical, n_results=3)

        return {
            "question": question,
            "hypothetical_preview": hypothetical[:150] + "...",
            "standard": {
                "docs": [d["id"] for d in standard_docs],
                "avg_distance": round(sum(d["distance"] for d in standard_docs) / len(standard_docs), 4),
            },
            "hyde": {
                "docs": [d["id"] for d in hyde_docs],
                "avg_distance": round(sum(d["distance"] for d in hyde_docs) / len(hyde_docs), 4),
            },
            "overlap": len(set(d["id"] for d in standard_docs) & set(d["id"] for d in hyde_docs)),
        }


def main():
    log("=== HyDE RAG Example ===\n")
    check_api_key()

    log("[1/3] Initializing...")
    client, collection = create_rag()

    log("[2/3] Indexing documents...")
    num_docs = index_documents(collection, DOCUMENTS)
    log(f"       Indexed {num_docs} documents\n")

    log("[3/3] Testing HyDE queries...\n")
    rag = HyDERAG(client, collection)

    # Test queries - abstract questions work best with HyDE
    queries = [
        "How do I keep my data fresh?",  # Abstract → maps to cache invalidation
        "My app is slow when loading users",  # Vague → maps to N+1 or connection pooling
        "Best way to split a huge database",  # Informal → maps to sharding
    ]

    for i, q in enumerate(queries, 1):
        log(f"--- Query {i}: HyDE vs Standard ---")
        log(f"Q: {q}\n")

        comparison = rag.compare(q)

        log(f"Hypothetical: \"{comparison['hypothetical_preview']}\"")
        log(f"\nStandard retrieval: {comparison['standard']['docs']} (avg dist: {comparison['standard']['avg_distance']})")
        log(f"HyDE retrieval:     {comparison['hyde']['docs']} (avg dist: {comparison['hyde']['avg_distance']})")
        log(f"Overlap: {comparison['overlap']}/3 docs\n")

    # Full HyDE query with answer
    log("--- Full HyDE Query ---")
    question = "Why is my web app creating too many database connections?"
    log(f"Q: {question}")

    result = rag.query(question, use_hyde=True, verbose=True)
    log(f"Retrieved: {result['retrieved_docs']}")
    log(f"A: {result['answer']}\n")

    # When HyDE shines
    log("--- When HyDE Helps Most ---")
    log("HyDE excels when:")
    log("  1. Queries are abstract or conceptual")
    log("  2. User vocabulary differs from document vocabulary")
    log("  3. Questions are vague or informal")
    log("  4. Documents are technical/formal, queries are casual")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        log(f"\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
