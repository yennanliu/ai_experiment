import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
from rag.embeddings import OllamaEmbeddings

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


async def ingest_file(file_path: str, collection_name: str, filename: str) -> int:
    docs = load_document(file_path)
    chunks = chunk_documents(docs)

    embeddings = OllamaEmbeddings()
    client = get_chroma_client()
    collection = client.get_or_create_collection(collection_name)

    texts = [c.page_content for c in chunks]
    metadatas = [{**c.metadata, "source": filename, "chunk": i} for i, c in enumerate(chunks)]
    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    # Embed in batches
    import asyncio
    from models.ollama_client import embed

    vectors = []
    for text in texts:
        vec = await embed(text)
        vectors.append(vec)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )
    return len(chunks)


def list_collections() -> List[str]:
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]


def delete_collection(name: str):
    client = get_chroma_client()
    client.delete_collection(name)
