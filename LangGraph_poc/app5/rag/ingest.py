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


def retrieve(query: str, collection: str, k: int = 5) -> list[tuple[str, str]]:
    col = _client.get_or_create_collection(collection)
    if col.count() == 0:
        return []
    q_emb = _get_embedder().embed_query(query)
    results = col.query(query_embeddings=[q_emb], n_results=min(k, col.count()))
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [(doc, meta["source"]) for doc, meta in zip(docs, metas)]


def retrieve_by_embedding(embedding: list[float], collection: str, k: int = 5) -> list[tuple[str, str]]:
    col = _client.get_or_create_collection(collection)
    if col.count() == 0:
        return []
    results = col.query(query_embeddings=[embedding], n_results=min(k, col.count()))
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [(doc, meta["source"]) for doc, meta in zip(docs, metas)]


def list_collections() -> list[str]:
    return [c.name for c in _client.list_collections()]


def delete_collection(name: str):
    _client.delete_collection(name)
