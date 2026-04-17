import os
from typing import List, Tuple
import chromadb
from models.ollama_client import embed
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


async def retrieve(
    question: str,
    collection_name: str,
    top_k: int = 5,
) -> List[Tuple[str, dict]]:
    """Returns list of (text, metadata) tuples."""
    query_vec = await embed(question)
    client = get_chroma_client()

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append((doc, meta))
    return chunks
