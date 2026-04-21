"""Demo 3: ReAct Agent
An LLM agent that can choose between a document search tool and a
calculator tool to answer questions requiring reasoning + retrieval.
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata


def _calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression and return the result."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic is supported."
    try:
        return str(eval(expression))  # noqa: S307 — restricted to arithmetic chars
    except Exception as e:
        return f"Error: {e}"


def run():
    print("\n[ReAct Agent] Loading documents ...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    search_tool = QueryEngineTool(
        query_engine=index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="document_search",
            description="Search AI/ML documents about machine learning, LLMs, "
                        "vector databases, RAG, semantic search, embedding models, "
                        "AI agents, NLP pipelines, and knowledge graphs.",
        ),
    )

    calc_tool = FunctionTool.from_defaults(
        fn=_calculator,
        name="calculator",
        description="Evaluate arithmetic expressions. Input must be a valid math "
                    "expression like '(3 + 5) * 12'. Returns the numeric result.",
    )

    agent = ReActAgent.from_tools(
        [search_tool, calc_tool],
        verbose=True,
        max_iterations=10,
    )

    print("\nReAct agent with document search + calculator tools.")
    print("Try: 'How many documents are about AI agents? Multiply that count by 7.'")
    print("('quit' to exit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            response = agent.chat(q)
            print(f"\nAgent: {response}\n")
