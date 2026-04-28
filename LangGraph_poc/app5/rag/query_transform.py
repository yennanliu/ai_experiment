"""Query transformation techniques: HyDE and Multi-Query."""
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

_llm = None
_embedder = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"), temperature=0.3)
    return _llm


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    return _embedder


def hyde_embedding(question: str) -> list[float]:
    """
    HyDE: Hypothetical Document Embeddings.
    Generate a plausible answer to the question, then embed that answer
    instead of the raw question. Closes the query-document distribution gap.
    """
    messages = [
        SystemMessage(content=(
            "Write a concise, factual paragraph that would answer the following question. "
            "Do not say you don't know — write a plausible, specific answer."
        )),
        HumanMessage(content=question),
    ]
    hypothetical_doc = _get_llm().invoke(messages).content
    return _get_embedder().embed_query(hypothetical_doc), hypothetical_doc


def multi_query_expand(question: str) -> list[str]:
    """
    Multi-Query: rewrite the question into N alternative phrasings
    to improve recall across different surface forms in the documents.
    """
    messages = [
        SystemMessage(content=(
            "Generate 3 different ways to ask the following question. "
            "Each rephrasing should capture a slightly different angle or vocabulary. "
            "Return only the 3 questions, one per line, no numbering or extra text."
        )),
        HumanMessage(content=question),
    ]
    result = _get_llm().invoke(messages).content
    variants = [q.strip() for q in result.strip().splitlines() if q.strip()]
    # always include the original
    all_queries = [question] + variants[:3]
    return list(dict.fromkeys(all_queries))  # deduplicate while preserving order
