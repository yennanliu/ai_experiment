from typing import TypedDict


class RAGState(TypedDict):
    question: str
    collection: str
    context: list[tuple[str, str]]   # (chunk_text, source)
    answer: str
