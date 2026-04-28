"""Document ingestion: load → chunk → embed → store."""
import os
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    return _embedder


def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Simple sliding-window character chunker."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def _load_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def ingest_file(path: str, collection: str, filename: str) -> int:
    text = _load_text(path)
    chunks = _chunk_text(text)
    embeddings = _get_embedder().embed_documents(chunks)

    col = _client.get_or_create_collection(collection)
    ids = [f"{filename}::{i}" for i in range(len(chunks))]
    col.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def retrieve(query: str, collection: str, k: int = 5) -> list[tuple[str, str]]:
    col = _client.get_or_create_collection(collection)
    if col.count() == 0:
        return []
    q_emb = _get_embedder().embed_query(query)
    results = col.query(query_embeddings=[q_emb], n_results=min(k, col.count()))
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [(doc, meta["source"]) for doc, meta in zip(docs, metas)]


def list_collections() -> list[str]:
    return [c.name for c in _client.list_collections()]


def delete_collection(name: str):
    _client.delete_collection(name)
