"""LLM-based reranker: score each chunk for relevance then re-sort."""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"), temperature=0)
    return _llm


def rerank(question: str, chunks: list[tuple[str, str]], top_k: int = 5) -> list[tuple[str, str]]:
    """
    Score each (text, source) chunk for relevance to the question.
    Returns top_k chunks sorted by descending relevance score.
    """
    if not chunks:
        return chunks

    scored = []
    for text, source in chunks:
        messages = [
            SystemMessage(content=(
                "Rate how relevant the following document chunk is for answering the question. "
                "Reply with a single integer from 0 (irrelevant) to 10 (perfectly relevant). "
                "No explanation, just the number."
            )),
            HumanMessage(content=f"Question: {question}\n\nChunk:\n{text}"),
        ]
        try:
            score_str = _get_llm().invoke(messages).content.strip()
            score = int("".join(c for c in score_str if c.isdigit()) or "0")
        except Exception:
            score = 0
        scored.append((score, text, source))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(text, source) for _, text, source in scored[:top_k]]
