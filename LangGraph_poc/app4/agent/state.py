from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    collection: str
    documents: List[Document]
    context: str
    answer: str
    sources: List[str]
    step: str
    grounded: Optional[bool]
