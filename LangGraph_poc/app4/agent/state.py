from typing import TypedDict, List


class AgentState(TypedDict):
    question: str
    collection: str
    chunks: List[tuple]   # (text, source)
    answer: str
    sources: List[str]
