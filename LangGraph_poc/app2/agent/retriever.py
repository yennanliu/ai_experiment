"""
ChromaDB vector store for RAG.
Stores historical (inbound_email → reply) pairs, retrieved at query time
to provide few-shot examples for the draft generator.
"""
from __future__ import annotations
import os
from functools import lru_cache
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

COLLECTION_NAME = "email_examples"
TOP_K = 3


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_example(doc_id: str, inbound_email: str, reply: str, email_type: str) -> None:
    """Upsert one email→reply example into the vector store."""
    get_collection().upsert(
        ids=[doc_id],
        documents=[inbound_email],
        metadatas=[{"reply": reply, "email_type": email_type}],
    )


def retrieve(inbound_email: str, email_type: str, k: int = TOP_K) -> list[dict]:
    """Return top-k similar historical examples, preferring same email_type."""
    col = get_collection()
    if col.count() == 0:
        return []

    n = min(k, col.count())

    # Try same-type filter first
    try:
        results = col.query(
            query_texts=[inbound_email],
            n_results=n,
            where={"email_type": email_type},
        )
        if results["ids"][0]:
            return _to_list(results)
    except Exception:
        pass

    # Fall back to global search
    results = col.query(query_texts=[inbound_email], n_results=n)
    return _to_list(results)


def _to_list(results: dict) -> list[dict]:
    return [
        {"email": doc, "reply": meta["reply"], "email_type": meta["email_type"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def count() -> int:
    return get_collection().count()
