"""Demo 4: Two-Stage Retrieval with Reranking
Stage 1 — retrieve top-K candidates via vector similarity.
Stage 2 — rerank with an LLM-based postprocessor (LLMRerank, no Cohere key needed).
Shows how reranking improves result relevance.
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import LLMRerank


def run():
    print("\n[Reranking] Loading documents ...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Stage 1: fetch 10 candidates; Stage 2: LLM reranks to top 3
    reranker = LLMRerank(choice_batch_size=5, top_n=3)

    engine_no_rerank = index.as_query_engine(similarity_top_k=3)
    engine_rerank = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[reranker],
    )

    print("\nCompares results WITH and WITHOUT reranking side-by-side.")
    print("Try: 'What are the key concepts in embedding models and how do they relate to semantic search?'")
    print("('quit' to exit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        print("\n--- Without reranking (top-3 by cosine similarity) ---")
        r1 = engine_no_rerank.query(q)
        print(r1)

        print("\n--- With LLM reranking (top-10 → reranked to top-3) ---")
        r2 = engine_rerank.query(q)
        print(r2)
        print()
