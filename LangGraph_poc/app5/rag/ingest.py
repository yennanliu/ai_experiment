"""Document ingestion: load → chunk → embed → store."""
import os
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

from rag.chunkers import chunk as do_chunk

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    return _embedder


def _load_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def ingest_file(path: str, collection: str, filename: str, strategy: str = "char") -> int:
    text = _load_text(path)
    chunks = do_chunk(text, strategy=strategy)
    embeddings = _get_embedder().embed_documents(chunks)

    col = _client.get_or_create_collection(collection)
    ids = [f"{filename}::{strategy}::{i}" for i in range(len(chunks))]
    col.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": filename, "chunk": i, "strategy": strategy} for i in range(len(chunks))],
    )
    return len(chunks)


def _l2_to_cosine_similarity(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity for unit-norm embeddings.
    OpenAI embeddings are unit-normalised, so: cosine_sim = 1 - L2² / 2.
    Returns a value in [0, 1] where 1 = identical.
    """
    return round(max(0.0, 1.0 - (l2_distance ** 2) / 2), 4)


def retrieve(query: str, collection: str, k: int = 5) -> list[tuple[str, str, float]]:
    """Returns (chunk_text, source, cosine_similarity) triples."""
    col = _client.get_or_create_collection(collection)
    if col.count() == 0:
        return []
    q_emb = _get_embedder().embed_query(query)
    results = col.query(query_embeddings=[q_emb], n_results=min(k, col.count()),
                        include=["documents", "metadatas", "distances"])
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [(doc, meta["source"], _l2_to_cosine_similarity(d))
            for doc, meta, d in zip(docs, metas, dists)]


def retrieve_by_embedding(embedding: list[float], collection: str, k: int = 5) -> list[tuple[str, str, float]]:
    """Returns (chunk_text, source, cosine_similarity) triples."""
    col = _client.get_or_create_collection(collection)
    if col.count() == 0:
        return []
    results = col.query(query_embeddings=[embedding], n_results=min(k, col.count()),
                        include=["documents", "metadatas", "distances"])
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [(doc, meta["source"], _l2_to_cosine_similarity(d))
            for doc, meta, d in zip(docs, metas, dists)]


def list_collections() -> list[str]:
    return [c.name for c in _client.list_collections()]


def delete_collection(name: str):
    _client.delete_collection(name)
