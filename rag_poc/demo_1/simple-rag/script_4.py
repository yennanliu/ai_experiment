"""
Agentic RAG Example

Demonstrates:
- Function calling to decide WHEN to retrieve
- Multi-step reasoning with tool use
- Self-reflection on retrieved documents
- Fallback to direct answer when retrieval not needed
"""

import json
import os
import sys
import traceback

import chromadb
from openai import OpenAI


def log(msg: str):
    print(msg, flush=True)


def check_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        log("ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)


# Knowledge base documents
DOCUMENTS = [
    {"id": "py-1", "text": "Python's list comprehensions provide a concise way to create lists. The syntax is [expression for item in iterable if condition]. They are faster than equivalent for loops.", "category": "python"},
    {"id": "py-2", "text": "Python decorators are functions that modify other functions. Use @decorator syntax above a function definition. Common uses include logging, timing, and access control.", "category": "python"},
    {"id": "py-3", "text": "Python's asyncio uses async/await for concurrent I/O operations. It's single-threaded but can handle many connections efficiently through cooperative multitasking.", "category": "python"},
    {"id": "db-1", "text": "Database indexing improves query performance by creating data structures that allow faster lookups. B-tree indexes are common for range queries, while hash indexes excel at equality checks.", "category": "database"},
    {"id": "db-2", "text": "ACID properties ensure database reliability: Atomicity (all or nothing), Consistency (valid state), Isolation (concurrent safety), Durability (permanent writes).", "category": "database"},
    {"id": "api-1", "text": "REST API best practices: use nouns for resources (/users), HTTP methods for actions (GET/POST/PUT/DELETE), proper status codes (200, 404, 500), and versioning (/v1/users).", "category": "api"},
    {"id": "api-2", "text": "API rate limiting protects servers from overload. Common strategies: token bucket, sliding window, and fixed window. Return 429 Too Many Requests when limits exceeded.", "category": "api"},
]

# Tools available to the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for information about Python, databases, or APIs. Use this when you need specific technical information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["python", "database", "api", "all"],
                        "description": "Category to filter results (optional, default: all)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_relevance",
            "description": "Evaluate if retrieved documents are relevant enough to answer the question. Use after searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The original question"},
                    "documents": {"type": "string", "description": "The retrieved documents"},
                },
                "required": ["question", "documents"],
            },
        },
    },
]


def create_rag(db_path: str = "./chroma_db_v4"):
    """Initialize RAG components."""
    client = OpenAI()
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_or_create_collection("agentic_docs")
    return client, collection


def index_documents(collection, documents: list[dict]) -> int:
    """Index documents into collection."""
    if not documents:
        return 0

    collection.upsert(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[{"category": d["category"]} for d in documents],
    )
    return len(documents)


def search_knowledge_base(collection, query: str, category: str = "all", n_results: int = 3) -> list[dict]:
    """Search the knowledge base."""
    where_filter = {"category": category} if category != "all" else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where_filter)

    return [
        {"id": results["ids"][0][i], "text": results["documents"][0][i], "score": results["distances"][0][i]}
        for i in range(len(results["ids"][0]))
    ]


def evaluate_relevance(client, question: str, documents: str) -> dict:
    """Use LLM to evaluate document relevance."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Evaluate if the documents are relevant to answer the question.
Return JSON: {"relevant": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}""",
            },
            {"role": "user", "content": f"Question: {question}\n\nDocuments:\n{documents}"},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"relevant": True, "confidence": 0.5, "reason": "Could not parse evaluation"}


class AgenticRAG:
    """RAG agent that decides when and how to retrieve."""

    def __init__(self, client, collection):
        self.client = client
        self.collection = collection
        self.max_iterations = 5

    def query(self, question: str, verbose: bool = True) -> dict:
        """Process query with agentic reasoning."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful technical assistant with access to a knowledge base.

Guidelines:
1. For factual technical questions, use search_knowledge_base to find information
2. For simple greetings or general questions, answer directly without searching
3. After retrieving documents, use evaluate_relevance to check if they help
4. If documents aren't relevant, try a different search query
5. Provide concise, accurate answers based on retrieved information""",
            },
            {"role": "user", "content": question},
        ]

        iterations = 0
        tool_calls_made = []

        while iterations < self.max_iterations:
            iterations += 1

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # No tool calls - agent decided to answer directly
            if not message.tool_calls:
                return {
                    "answer": message.content,
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                    "used_retrieval": len(tool_calls_made) > 0,
                }

            # Process tool calls
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if verbose:
                    log(f"  [Tool] {func_name}({json.dumps(args, ensure_ascii=False)})")

                # Execute the tool
                if func_name == "search_knowledge_base":
                    results = search_knowledge_base(
                        self.collection, args["query"], args.get("category", "all")
                    )
                    tool_result = "\n\n".join(f"[{r['id']}] {r['text']}" for r in results)
                    tool_calls_made.append({"function": func_name, "args": args, "results_count": len(results)})

                elif func_name == "evaluate_relevance":
                    eval_result = evaluate_relevance(self.client, args["question"], args["documents"])
                    tool_result = json.dumps(eval_result)
                    tool_calls_made.append({"function": func_name, "args": args, "evaluation": eval_result})

                    if verbose:
                        log(f"  [Eval] Relevant: {eval_result.get('relevant')}, Confidence: {eval_result.get('confidence')}")

                else:
                    tool_result = f"Unknown function: {func_name}"

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

        return {
            "answer": "Max iterations reached without final answer.",
            "tool_calls": tool_calls_made,
            "iterations": iterations,
            "used_retrieval": len(tool_calls_made) > 0,
        }


def main():
    log("=== Agentic RAG Example ===\n")
    check_api_key()

    log("[1/3] Initializing...")
    client, collection = create_rag()

    log("[2/3] Indexing documents...")
    num_docs = index_documents(collection, DOCUMENTS)
    log(f"       Indexed {num_docs} documents\n")

    log("[3/3] Testing agentic queries...\n")
    agent = AgenticRAG(client, collection)

    # Test queries - mix of retrieval-needed and direct-answer
    queries = [
        "Hello, how are you?",  # Should NOT use retrieval
        "What are Python decorators?",  # Should use retrieval
        "Explain ACID properties in databases",  # Should use retrieval
        "What is 2 + 2?",  # Should NOT use retrieval
        "How do I implement rate limiting for an API?",  # Should use retrieval
    ]

    for i, q in enumerate(queries, 1):
        log(f"--- Query {i} ---")
        log(f"Q: {q}")

        result = agent.query(q, verbose=True)

        log(f"A: {result['answer']}")
        log(f"   [Used retrieval: {result['used_retrieval']}, Iterations: {result['iterations']}]")
        log("")

    # Demonstrate multi-step reasoning
    log("--- Multi-step Reasoning Demo ---")
    complex_query = "I need to build a fast API endpoint. What Python features and API best practices should I use?"
    log(f"Q: {complex_query}")

    result = agent.query(complex_query, verbose=True)
    log(f"A: {result['answer']}")
    log(f"   [Tool calls: {len(result['tool_calls'])}, Iterations: {result['iterations']}]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        log(f"\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
