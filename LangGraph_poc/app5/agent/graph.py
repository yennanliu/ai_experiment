"""LangGraph RAG pipeline with advanced retrieval techniques."""
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.state import RAGState
from rag.ingest import retrieve, retrieve_by_embedding
from rag.query_transform import hyde_embedding, multi_query_expand
from rag.rerank import rerank as do_rerank

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"), temperature=0)
    return _llm


# ── Node: Query Transform ────────────────────────────────────────────────────

def transform_node(state: RAGState) -> RAGState:
    mode = state.get("query_transform", "none")
    question = state["question"]

    if mode == "hyde":
        emb, hypo_doc = hyde_embedding(question)
        return {**state, "transformed_queries": [question], "hyde_doc": hypo_doc, "_hyde_emb": emb}

    if mode == "multi_query":
        variants = multi_query_expand(question)
        return {**state, "transformed_queries": variants, "hyde_doc": ""}

    return {**state, "transformed_queries": [question], "hyde_doc": ""}


# ── Node: Retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> RAGState:
    mode = state.get("query_transform", "none")
    collection = state["collection"]
    k = state.get("k", 5)
    seen, chunks = set(), []

    if mode == "hyde" and "_hyde_emb" in state:
        raw = retrieve_by_embedding(state["_hyde_emb"], collection, k=k * 2)
    elif mode == "multi_query":
        raw = []
        for q in state["transformed_queries"]:
            raw.extend(retrieve(q, collection, k=k))
    else:
        raw = retrieve(state["question"], collection, k=k * 2)

    # deduplicate by chunk text
    for text, source in raw:
        if text not in seen:
            seen.add(text)
            chunks.append((text, source))

    return {**state, "context": chunks}


# ── Node: Rerank ─────────────────────────────────────────────────────────────

def rerank_node(state: RAGState) -> RAGState:
    if not state.get("rerank") or not state["context"]:
        return state
    reranked = do_rerank(state["question"], state["context"], top_k=state.get("k", 5))
    return {**state, "context": reranked}


# ── Node: Generate ───────────────────────────────────────────────────────────

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


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("transform", transform_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("transform")
    g.add_edge("transform", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = None


def run(question: str, collection: str, k: int = 5,
        query_transform: str = "none", rerank: bool = False) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()

    result = _graph.invoke({
        "question": question,
        "collection": collection,
        "k": k,
        "query_transform": query_transform,
        "rerank": rerank,
        "transformed_queries": [],
        "hyde_doc": "",
        "context": [],
        "answer": "",
    })

    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(src for _, src in result["context"])),
        "chunks_used": len(result["context"]),
        "chunks": [{"text": text, "source": src} for text, src in result["context"]],
        "transformed_queries": result.get("transformed_queries", []),
        "hyde_doc": result.get("hyde_doc", ""),
        "pipeline": {
            "query_transform": query_transform,
            "rerank": rerank,
            "k": k,
        },
    }
