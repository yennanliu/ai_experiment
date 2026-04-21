"""Demo 1: Sub-Question Query Engine
Breaks a complex question into sub-questions, queries each independently,
then synthesizes a final answer. Great for multi-hop reasoning.
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def run():
    print("\n[Sub-Question Query] Loading documents ...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    base_engine = index.as_query_engine(similarity_top_k=3)

    tool = QueryEngineTool(
        query_engine=base_engine,
        metadata=ToolMetadata(
            name="tech_docs",
            description="AI/ML technology documents covering machine learning, LLMs, "
                        "vector databases, RAG systems, semantic search, embedding models, "
                        "AI agents, NLP pipelines, and knowledge graphs.",
        ),
    )

    engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[tool],
        verbose=True,
    )

    print("\nThis engine decomposes your question into sub-questions automatically.")
    print("Try: 'Compare RAG systems and vector databases. Which is more important for semantic search?'")
    print("('quit' to exit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            response = engine.query(q)
            print(f"\nFinal Answer: {response}\n")
