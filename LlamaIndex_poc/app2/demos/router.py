"""Demo 6: Router Query Engine
Maintains two indexes — a vector index (semantic) and a summary index (holistic).
The LLM router automatically picks the best engine for each query:
- Factual / specific questions → vector index
- Broad / summarization questions → summary index
"""
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def run():
    print("\n[Router Query Engine] Loading documents and building two indexes ...")
    documents = SimpleDirectoryReader("data").load_data()

    vector_index = VectorStoreIndex.from_documents(documents)
    summary_index = SummaryIndex.from_documents(documents)

    vector_tool = QueryEngineTool(
        query_engine=vector_index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="vector_search",
            description="Good for specific factual questions about a particular AI/ML "
                        "topic such as 'what is a knowledge graph?' or 'explain RAG'.",
        ),
    )

    summary_tool = QueryEngineTool(
        query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
        metadata=ToolMetadata(
            name="summary",
            description="Good for broad or summarization questions like 'give me an "
                        "overview of all topics' or 'what subjects are covered?'.",
        ),
    )

    engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[vector_tool, summary_tool],
        verbose=True,
    )

    print("\nThe router picks vector search OR summary index based on your question.")
    print("Try specific: 'What is semantic search?'")
    print("Try broad:    'Give me a high-level overview of all topics covered.'")
    print("('quit' to exit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            response = engine.query(q)
            print(f"\nAI: {response}\n")
