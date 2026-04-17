from agent.state import AgentState
from rag.retriever import retrieve


async def retrieve_node(state: AgentState) -> AgentState:
    chunks = await retrieve(state["question"], state["collection"])
    return {**state, "chunks": chunks}


async def generate_node(state: AgentState) -> AgentState:
    # Build context and deduplicate sources
    context = "\n\n---\n\n".join(text for text, _ in state["chunks"])
    sources = list(dict.fromkeys(src for _, src in state["chunks"]))

    # Import here to keep node pure/testable
    from models.ollama_client import chat_stream
    tokens = []
    async for token in chat_stream(
        prompt=f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:",
        system=(
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "If the answer is not in the context, say so."
        ),
    ):
        tokens.append(token)

    return {**state, "answer": "".join(tokens), "sources": sources}
