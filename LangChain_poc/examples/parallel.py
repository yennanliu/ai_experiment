"""Parallel graph — analyze text from multiple angles simultaneously, then merge."""

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
    text: str
    sentiment: str
    keywords: str
    summary: str
    report: str


def analyze_sentiment(state: State) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "Analyze the sentiment of the text. Reply with: Positive, Negative, or Neutral, followed by a one-line explanation."),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()
    return {"sentiment": chain.invoke({"text": state["text"]})}


def extract_keywords(state: State) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "Extract 5-8 key topics/keywords from the text. List them comma-separated."),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()
    return {"keywords": chain.invoke({"text": state["text"]})}


def summarize(state: State) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "Summarize the text in 2-3 sentences."),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()
    return {"summary": chain.invoke({"text": state["text"]})}


def merge_report(state: State) -> dict:
    report = f"""--- Analysis Report ---

SUMMARY: {state['summary']}

SENTIMENT: {state['sentiment']}

KEYWORDS: {state['keywords']}
"""
    return {"report": report}


graph = StateGraph(State)
graph.add_node("sentiment", analyze_sentiment)
graph.add_node("keywords", extract_keywords)
graph.add_node("summarize", summarize)
graph.add_node("merge", merge_report)

graph.add_edge(START, "sentiment")
graph.add_edge(START, "keywords")
graph.add_edge(START, "summarize")
graph.add_edge("sentiment", "merge")
graph.add_edge("keywords", "merge")
graph.add_edge("summarize", "merge")
graph.add_edge("merge", END)

app = graph.compile()

if __name__ == "__main__":
    print_graph(app, "Parallel Analysis → Merge")
    print("Parallel Text Analyzer (type 'quit' to exit)\n")
    while True:
        print("Paste text (end with empty line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if not text or text.lower() == "quit":
            break
        print("\nAnalyzing in parallel...\n")
        result = app.invoke({"text": text})
        print(result["report"])
        print("=" * 60 + "\n")
