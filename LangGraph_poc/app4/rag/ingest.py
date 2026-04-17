import os
from pathlib import Path
from typing import List
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from models.ollama_client import embed

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _get_db():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _parse(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return Path(file_path).read_text(encoding="utf-8")


def _chunk(text: str) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


async def ingest_file(file_path: str, collection: str, filename: str) -> int:
    text = _parse(file_path)
    chunks = _chunk(text)

    db = _get_db()
    col = db.get_or_create_collection(collection)

    ids = [f"{filename}__{i}" for i in range(len(chunks))]
    vectors = [await embed(c) for c in chunks]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    col.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
    return len(chunks)


def list_collections() -> List[str]:
    return [c.name for c in _get_db().list_collections()]


def delete_collection(name: str):
    _get_db().delete_collection(name)
