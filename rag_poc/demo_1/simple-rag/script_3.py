"""
Conversational RAG Example

Demonstrates:
- Multi-turn conversation with memory
- LLM-based reranking for better relevance
- Streaming responses
- Loading documents from files
"""

import os
import sys
import traceback
from pathlib import Path

import chromadb
from openai import OpenAI


def log(msg: str):
    print(msg, flush=True)


def check_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        log("ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)


# Sample documents (could be loaded from files)
DOCUMENTS = [
    {"id": "arch-1", "text": "Microservices architecture breaks applications into small, independent services that communicate via APIs. Each service can be developed, deployed, and scaled independently.", "category": "architecture"},
    {"id": "arch-2", "text": "Monolithic architecture combines all application components into a single codebase. It's simpler to develop initially but harder to scale and maintain as the application grows.", "category": "architecture"},
    {"id": "arch-3", "text": "Event-driven architecture uses events to trigger communication between decoupled services. It enables real-time processing and loose coupling between components.", "category": "architecture"},
    {"id": "db-1", "text": "PostgreSQL is a powerful relational database with ACID compliance, JSON support, and advanced features like full-text search and geographic queries.", "category": "database"},
    {"id": "db-2", "text": "MongoDB is a document database that stores data in flexible JSON-like documents. It's good for unstructured data and horizontal scaling.", "category": "database"},
    {"id": "db-3", "text": "Redis is an in-memory data store used for caching, session management, and real-time analytics. It supports various data structures like strings, hashes, and sorted sets.", "category": "database"},
    {"id": "api-1", "text": "REST APIs use HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations. They are stateless and use standard HTTP status codes.", "category": "api"},
    {"id": "api-2", "text": "GraphQL allows clients to request exactly the data they need in a single query. It reduces over-fetching and under-fetching compared to REST.", "category": "api"},
]


def load_documents_from_dir(dir_path: str) -> list[dict]:
    """Load .txt files from a directory as documents."""
    docs = []
    path = Path(dir_path)
    if not path.exists():
        return docs

    for file in path.glob("*.txt"):
        docs.append({
            "id": file.stem,
            "text": file.read_text().strip(),
            "category": "file",
        })
    return docs


def create_rag(db_path: str = "./chroma_db_v3"):
    """Initialize RAG components."""
    client = OpenAI()
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_or_create_collection("conversational_docs")
    return client, collection


def index_documents(collection, documents: list[dict]) -> int:
    """Index documents into the collection."""
    if not documents:
        return 0

    collection.upsert(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[{"category": d["category"]} for d in documents],
    )
    return len(documents)


def retrieve(collection, query: str, n_results: int = 5) -> list[dict]:
    """Retrieve relevant documents."""
    results = collection.query(query_texts=[query], n_results=n_results)

    return [
        {"id": results["ids"][0][i], "text": results["documents"][0][i], "score": results["distances"][0][i]}
        for i in range(len(results["ids"][0]))
    ]


def rerank_with_llm(client, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
    """Use LLM to rerank documents by relevance."""
    if len(documents) <= top_k:
        return documents

    doc_list = "\n".join(f"[{i}] {d['text'][:200]}" for i, d in enumerate(documents))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Given the query, rank the top {top_k} most relevant document numbers. Return only comma-separated numbers (e.g., '2,0,4')."},
            {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{doc_list}"},
        ],
        temperature=0,
    )

    try:
        indices = [int(x.strip()) for x in response.choices[0].message.content.split(",")]
        return [documents[i] for i in indices if i < len(documents)][:top_k]
    except (ValueError, IndexError):
        return documents[:top_k]


def generate_response(client, query: str, context: str, history: list[dict], stream: bool = False):
    """Generate response with conversation history."""
    messages = [
        {"role": "system", "content": "You are a helpful technical assistant. Answer based on the provided context. Be concise and accurate. If unsure, say so."},
    ]

    # Add conversation history (last 4 exchanges)
    for h in history[-8:]:
        messages.append(h)

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=stream,
    )

    if stream:
        return response  # Return generator for streaming
    return response.choices[0].message.content


class ConversationalRAG:
    """RAG with conversation memory and reranking."""

    def __init__(self, client, collection):
        self.client = client
        self.collection = collection
        self.history: list[dict] = []

    def query(self, question: str, use_reranking: bool = True, stream: bool = False) -> str:
        """Process a query with context and history."""
        # Retrieve
        docs = retrieve(self.collection, question, n_results=5)

        # Rerank
        if use_reranking and len(docs) > 3:
            docs = rerank_with_llm(self.client, question, docs, top_k=3)

        # Build context
        context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)

        # Generate
        if stream:
            return self._stream_response(question, context)

        answer = generate_response(self.client, question, context, self.history)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def _stream_response(self, question: str, context: str) -> str:
        """Stream response and collect full answer."""
        stream = generate_response(self.client, question, context, self.history, stream=True)

        full_response = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response.append(content)
        print()  # Newline after streaming

        answer = "".join(full_response)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):
        """Clear conversation history."""
        self.history = []


def main():
    log("=== Conversational RAG Example ===\n")
    check_api_key()

    log("[1/3] Initializing...")
    client, collection = create_rag()

    log("[2/3] Indexing documents...")
    num_docs = index_documents(collection, DOCUMENTS)
    log(f"       Indexed {num_docs} documents\n")

    log("[3/3] Starting conversation...\n")
    rag = ConversationalRAG(client, collection)

    # Demo conversation
    questions = [
        ("What databases are good for caching?", False),  # No streaming
        ("How does it compare to PostgreSQL?", False),    # Follow-up (uses history)
        ("What about API design patterns?", True),        # Streaming response
    ]

    for i, (question, stream) in enumerate(questions, 1):
        log(f"--- Turn {i} ---")
        log(f"Q: {question}")

        if stream:
            log("A: ", )
            rag.query(question, stream=True)
        else:
            answer = rag.query(question)
            log(f"A: {answer}")
        log("")

    # Show reranking comparison
    log("--- Reranking Comparison ---")
    rag.clear_history()

    question = "Which architecture is best for large scale systems?"

    log(f"Q: {question}")
    log("\nWithout reranking:")
    docs = retrieve(collection, question, n_results=5)
    for d in docs[:3]:
        log(f"  - [{d['id']}] {d['text'][:60]}...")

    log("\nWith LLM reranking:")
    reranked = rerank_with_llm(client, question, docs, top_k=3)
    for d in reranked:
        log(f"  - [{d['id']}] {d['text'][:60]}...")


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
