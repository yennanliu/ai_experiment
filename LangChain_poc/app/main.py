"""FastAPI app backed by LangChain / LangGraph."""

import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from core import make_chain, get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

app = FastAPI(title="LangChain API")

# --- Chains / Graphs ---

chat_chain = make_chain()

llm = get_llm()


# Research → Summary graph
class ResearchState(TypedDict):
    topic: str
    research: str
    summary: str


def _research(state: ResearchState) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. List 5 key facts about the topic. Be concise."),
        ("human", "Topic: {topic}"),
    ]) | llm | StrOutputParser()
    return {"research": chain.invoke({"topic": state["topic"]})}


def _summarize(state: ResearchState) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "Produce a structured summary: 1) Overview, 2) Bullet points, 3) Why it matters."),
        ("human", "Research notes:\n{research}"),
    ]) | llm | StrOutputParser()
    return {"summary": chain.invoke({"research": state["research"]})}


_g = StateGraph(ResearchState)
_g.add_node("research", _research)
_g.add_node("summarize", _summarize)
_g.add_edge(START, "research")
_g.add_edge("research", "summarize")
_g.add_edge("summarize", END)
research_graph = _g.compile()


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class ResearchRequest(BaseModel):
    topic: str

class ResearchResponse(BaseModel):
    research: str
    summary: str


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response = chat_chain.invoke({"input": req.message})
    return ChatResponse(reply=response.content)


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    result = research_graph.invoke({"topic": req.topic})
    return ResearchResponse(research=result["research"], summary=result["summary"])
