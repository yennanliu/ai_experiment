"""LangGraph RAG pipeline with advanced retrieval techniques."""
import os
import time

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
    t0 = time.perf_counter()
    mode = state.get("query_transform", "none")
    question = state["question"]

    if mode == "hyde":
        emb, hypo_doc = hyde_embedding(question)
        result = {**state, "transformed_queries": [question], "hyde_doc": hypo_doc, "_hyde_emb": emb}
    elif mode == "multi_query":
        variants = multi_query_expand(question)
        result = {**state, "transformed_queries": variants, "hyde_doc": ""}
    else:
        result = {**state, "transformed_queries": [question], "hyde_doc": ""}

    timings = {**state.get("timings", {}), "transform": round(time.perf_counter() - t0, 3)}
    return {**result, "timings": timings}


# ── Node: Retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> RAGState:
    t0 = time.perf_counter()
    mode = state.get("query_transform", "none")
    collection = state["collection"]
    k = state.get("k", 5)
    seen, chunks, scores = set(), [], []

    if mode == "hyde" and "_hyde_emb" in state:
        raw = retrieve_by_embedding(state["_hyde_emb"], collection, k=k * 2)
    elif mode == "multi_query":
        raw = []
        for q in state["transformed_queries"]:
            raw.extend(retrieve(q, collection, k=k))
    else:
        raw = retrieve(state["question"], collection, k=k * 2)

    # deduplicate by chunk text; keep best (highest) score per chunk
    best_score: dict[str, float] = {}
    ordered: list[tuple[str, str]] = []
    for text, source, score in raw:
        if text not in seen:
            seen.add(text)
            ordered.append((text, source))
        if text not in best_score or score > best_score[text]:
            best_score[text] = score

    chunks = ordered
    scores = [best_score[text] for text, _ in chunks]

    timings = {**state.get("timings", {}), "retrieve": round(time.perf_counter() - t0, 3)}
    return {**state, "context": chunks, "retrieval_scores": scores, "timings": timings}


# ── Node: Rerank ─────────────────────────────────────────────────────────────

def rerank_node(state: RAGState) -> RAGState:
    if not state.get("rerank") or not state["context"]:
        return {**state, "rerank_scores": []}
    t0 = time.perf_counter()
    reranked = do_rerank(state["question"], state["context"], top_k=state.get("k", 5))
    # reranked is now list of (text, source, score)
    context = [(text, source) for text, source, _ in reranked]
    rscores = [score for _, _, score in reranked]
    timings = {**state.get("timings", {}), "rerank": round(time.perf_counter() - t0, 3)}
    return {**state, "context": context, "rerank_scores": rscores, "timings": timings}


# ── Node: Generate ───────────────────────────────────────────────────────────

def generate_node(state: RAGState) -> RAGState:
    t0 = time.perf_counter()
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
    timings = {**state.get("timings", {}), "generate": round(time.perf_counter() - t0, 3)}
    return {**state, "answer": response.content, "timings": timings}


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
        "retrieval_scores": [],
        "rerank_scores": [],
        "timings": {},
    })

    retrieval_scores = result.get("retrieval_scores", [])
    rerank_scores = result.get("rerank_scores", [])
    context = result["context"]

    chunks = []
    for i, (text, src) in enumerate(context):
        entry = {"text": text, "source": src}
        if i < len(retrieval_scores):
            entry["similarity"] = retrieval_scores[i]
        if rerank_scores and i < len(rerank_scores):
            entry["rerank_score"] = rerank_scores[i]
        chunks.append(entry)

    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(src for _, src in context)),
        "chunks_used": len(context),
        "chunks": chunks,
        "transformed_queries": result.get("transformed_queries", []),
        "hyde_doc": result.get("hyde_doc", ""),
        "pipeline": {
            "query_transform": query_transform,
            "rerank": rerank,
            "k": k,
        },
        "timings": result.get("timings", {}),
    }
