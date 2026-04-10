"""Multi-step graph — research a topic, then generate a structured summary."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core import get_llm, print_graph

llm = get_llm()


class State(TypedDict):
    topic: str
    research: str
    summary: str


def research(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Given a topic, list 5 key facts. Be factual and concise."),
        ("human", "Topic: {topic}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"research": chain.invoke({"topic": state["topic"]})}


def summarize(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a technical writer. Given research notes, produce a structured summary with: "
         "1) One-line overview, 2) Key points as bullet list, 3) A 'Why it matters' conclusion."),
        ("human", "Research notes:\n{research}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"summary": chain.invoke({"research": state["research"]})}


graph = StateGraph(State)
graph.add_node("research", research)
graph.add_node("summarize", summarize)
graph.add_edge(START, "research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", END)
app = graph.compile()

if __name__ == "__main__":
    print_graph(app, "Research → Summary")
    print("Multi-Step Research → Summary (type 'quit' to exit)\n")
    while True:
        topic = input("Topic: ").strip()
        if not topic or topic.lower() == "quit":
            break
        print("\nResearching & summarizing...\n")
        result = app.invoke({"topic": topic})
        print(result["summary"])
        print("\n" + "=" * 60 + "\n")
