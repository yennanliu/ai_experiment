"""Critique loop — generate an answer, critique it, then refine. Demonstrates cycles in LangGraph."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core import get_llm, print_graph

llm = get_llm()
MAX_REVISIONS = 2


class State(TypedDict):
    question: str
    draft: str
    critique: str
    revision_count: int
    final: str


def generate(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question clearly and concisely."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"draft": chain.invoke({"question": state["question"]}), "revision_count": 0}


def critique(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a critical reviewer. Evaluate the draft answer for accuracy, clarity, and completeness. "
         "List specific improvements needed. If the answer is already excellent, reply with exactly: APPROVED"),
        ("human", "Question: {question}\n\nDraft answer:\n{draft}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"critique": chain.invoke({"question": state["question"], "draft": state["draft"]})}


def should_revise(state: State) -> str:
    if "APPROVED" in state["critique"] or state["revision_count"] >= MAX_REVISIONS:
        return "finalize"
    return "revise"


def revise(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Improve the draft based on the critique. Keep the answer concise."),
        ("human", "Question: {question}\n\nDraft:\n{draft}\n\nCritique:\n{critique}"),
    ])
    chain = prompt | llm | StrOutputParser()
    new_draft = chain.invoke({
        "question": state["question"],
        "draft": state["draft"],
        "critique": state["critique"],
    })
    return {"draft": new_draft, "revision_count": state["revision_count"] + 1}


def finalize(state: State) -> dict:
    return {"final": state["draft"]}


graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("critique", critique)
graph.add_node("revise", revise)
graph.add_node("finalize", finalize)

graph.add_edge(START, "generate")
graph.add_edge("generate", "critique")
graph.add_conditional_edges("critique", should_revise, {"revise": "revise", "finalize": "finalize"})
graph.add_edge("revise", "critique")
graph.add_edge("finalize", END)

app = graph.compile()

if __name__ == "__main__":
    print_graph(app, "Generate → Critique → Revise Loop")
    print("Critique Loop (type 'quit' to exit)\n")
    while True:
        question = input("Question: ").strip()
        if not question or question.lower() == "quit":
            break
        print("\nGenerating → critiquing → refining...\n")
        result = app.invoke({"question": question})
        print(f"Revisions: {result['revision_count']}")
        print(f"\n{result['final']}\n")
        print("=" * 60 + "\n")
