"""
Simple RAG (Retrieval-Augmented Generation) Example

This demonstrates the core RAG pattern:
1. Index documents into a vector store
2. Retrieve relevant documents for a query
3. Generate an answer using the retrieved context
"""

import os
import sys

import chromadb
from openai import OpenAI


def check_api_key():
    """Check if OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)


def create_rag():
    """Create and return RAG components."""
    client = OpenAI()
    db = chromadb.Client()
    collection = db.create_collection("docs")
    return client, collection


def add_documents(collection, documents: list[str]):
    """Add documents to the vector store."""
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )


def query(client, collection, question: str, n_results: int = 2) -> str:
    """
    RAG query: retrieve relevant docs and generate answer.

    Args:
        client: OpenAI client
        collection: ChromaDB collection
        question: User's question
        n_results: Number of documents to retrieve

    Returns:
        Generated answer based on retrieved context
    """
    # Retrieve relevant documents
    results = collection.query(query_texts=[question], n_results=n_results)
    context = "\n".join(results["documents"][0])

    # Generate answer with context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer based on the provided context. Be concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def main():
    check_api_key()

    # Sample knowledge base
    documents = [
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python emphasizes code readability with significant indentation.",
        "The Zen of Python includes principles like 'Simple is better than complex'.",
        "Python supports multiple paradigms: procedural, OOP, and functional.",
        "Popular Python frameworks include Django, Flask, and FastAPI.",
    ]

    # Initialize RAG
    client, collection = create_rag()
    add_documents(collection, documents)

    # Example queries
    questions = [
        "Who created Python?",
        "What are Python's design principles?",
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {query(client, collection, q)}\n")


if __name__ == "__main__":
    main()
