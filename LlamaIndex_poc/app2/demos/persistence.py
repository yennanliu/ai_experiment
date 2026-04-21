"""Demo 5: Index Persistence
Build an index once, persist it to disk, then reload it on subsequent runs
without re-embedding. Demonstrates the storage/persistence layer.
"""
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

PERSIST_DIR = "./storage"


def run():
    if os.path.exists(PERSIST_DIR):
        print(f"\n[Persistence] Found existing index at '{PERSIST_DIR}', loading ...")
        storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_ctx)
        print("Index loaded from disk — no re-embedding needed!")
    else:
        print(f"\n[Persistence] No index found. Building and saving to '{PERSIST_DIR}' ...")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Index saved! Restart this demo to load from disk instead of re-building.")

    engine = index.as_query_engine(similarity_top_k=3)

    print("\nQuery the persisted index. ('quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print(f"AI: {engine.query(q)}\n")
