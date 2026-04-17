import os
from typing import List, Tuple
import chromadb
from models.ollama_client import embed
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


async def retrieve(question: str, collection: str, top_k: int = 4) -> List[Tuple[str, str]]:
    """Returns list of (text, source_filename) tuples."""
    vec = await embed(question)
    db = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        col = db.get_collection(collection)
    except Exception:
        return []

    n = min(top_k, col.count())
    if n == 0:
        return []

    results = col.query(query_embeddings=[vec], n_results=n, include=["documents", "metadatas"])
    return [
        (doc, meta.get("source", "unknown"))
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
