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
    # learning-by-doing: scores & timing
    retrieval_scores: list[float]    # cosine similarity per chunk (before rerank)
    rerank_scores: list[int]         # LLM relevance score per chunk (0-10, after rerank)
    timings: dict                    # node_name -> elapsed seconds
    # evaluation
    evaluate: bool                   # whether to run the evaluator node
    reference_answer: str            # optional ground-truth answer for correctness scoring
    evaluation: dict                 # output of rag.evaluator.evaluate()
