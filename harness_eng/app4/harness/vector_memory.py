"""
Project 11: Persistent Vector Memory — ChromaDB semantic store.

Drop-in replacement for the flat store.json K/V used in app1-3.
On first run, chromadb downloads a small local embedding model (~25 MB).
Subsequent runs use the cached model and the persisted chroma store.
"""

from pathlib import Path
import chromadb

_DIR = Path(__file__).parent.parent / "memory" / "chroma"
_COLLECTION = "harness_memory"


def _col():
    client = chromadb.PersistentClient(path=str(_DIR))
    return client.get_or_create_collection(_COLLECTION)


def add(key: str, text: str, metadata: dict | None = None) -> None:
    """Embed and upsert a document. Overwrites if key exists."""
    _col().upsert(ids=[key], documents=[text], metadatas=[metadata or {}])


def search(query: str, n: int = 3) -> list[dict]:
    """Return top-n semantically similar documents for query."""
    col = _col()
    count = col.count()
    if count == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(n, count))
    return [
        {"id": id_, "text": doc, "score": round(1 - dist, 3)}
        for id_, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]


def get(key: str) -> str | None:
    """Exact-key lookup; returns None if not found."""
    try:
        r = _col().get(ids=[key])
        return r["documents"][0] if r["documents"] else None
    except Exception:
        return None


def count() -> int:
    return _col().count()
