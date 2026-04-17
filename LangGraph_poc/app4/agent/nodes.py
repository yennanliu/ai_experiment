import os
from langchain_core.documents import Document
from agent.state import AgentState
from rag.retriever import retrieve
from models.ollama_client import chat, OLLAMA_LLM_MODEL
import asyncio


async def retrieve_node(state: AgentState) -> AgentState:
    chunks = await retrieve(state["question"], state["collection"])
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in chunks
    ]
    return {**state, "documents": documents, "step": "retrieved"}


async def grade_node(state: AgentState) -> AgentState:
    """Filter chunks not relevant to the question."""
    question = state["question"]
    relevant = []

    for doc in state["documents"]:
        prompt = (
            f"Is the following document relevant to answering this question?\n\n"
            f"Question: {question}\n\n"
            f"Document: {doc.page_content[:400]}\n\n"
            f"Reply only 'yes' or 'no'."
        )
        answer = await chat(prompt)
        if "yes" in answer.lower():
            relevant.append(doc)

    # Fall back to all docs if none pass grading
    if not relevant:
        relevant = state["documents"]

    return {**state, "documents": relevant, "step": "graded"}


async def generate_node(state: AgentState) -> AgentState:
    model = os.getenv("OLLAMA_LLM_MODEL", OLLAMA_LLM_MODEL)
    docs = state["documents"]

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source", "unknown") for d in docs})

    system = (
        "You are a helpful assistant. Answer the user's question based ONLY on the "
        "provided context. If the answer is not in the context, say so clearly."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"

    answer = await chat(prompt, model=model, system=system)

    return {**state, "context": context, "answer": answer, "sources": sources, "step": "generated"}
