"""LangGraph RAG pipeline: retrieve → generate."""
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.state import RAGState
from rag.ingest import retrieve

LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _llm


def retrieve_node(state: RAGState) -> RAGState:
    chunks = retrieve(state["question"], state["collection"], k=state.get("k", 5))
    return {**state, "context": chunks}


def generate_node(state: RAGState) -> RAGState:
    context = state["context"]
    if not context:
        return {**state, "answer": "No relevant documents found in this knowledge base."}

    context_text = "\n\n---\n\n".join(text for text, _ in context)
    messages = [
        SystemMessage(content=(
            "You are an enterprise knowledge assistant. "
            "Answer using ONLY the provided context. "
            "Be concise and professional. "
            "If the answer is not in the context, say so clearly."
        )),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {state['question']}"),
    ]
    response = _get_llm().invoke(messages)
    return {**state, "answer": response.content}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = None


def run(question: str, collection: str, k: int = 5) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    result = _graph.invoke({"question": question, "collection": collection, "k": k, "context": [], "answer": ""})
    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(src for _, src in result["context"])),
        "chunks_used": len(result["context"]),
        "chunks": [{"text": text, "source": src} for text, src in result["context"]],
    }
