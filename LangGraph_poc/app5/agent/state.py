from typing import TypedDict


class RAGState(TypedDict):
    question: str
    collection: str
    k: int
    # advanced options
    query_transform: str   # "none" | "hyde" | "multi_query"
    rerank: bool
    # pipeline outputs
    transformed_queries: list[str]   # queries actually used for retrieval
    hyde_doc: str                    # hypothetical doc (HyDE only)
    context: list[tuple[str, str]]   # (chunk_text, source)
    answer: str
