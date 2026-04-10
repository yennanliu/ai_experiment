"""Router graph — classify user intent and route to specialized handlers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core import get_llm, print_graph

llm = get_llm()


class State(TypedDict):
    input: str
    intent: str
    response: str


def classify(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the user message into exactly one category: code, math, or general. "
         "Reply with ONLY the category word, nothing else."),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"input": state["input"]}).strip().lower()
    return {"intent": intent}


def route(state: State) -> Literal["code_expert", "math_expert", "general_expert"]:
    if "code" in state["intent"]:
        return "code_expert"
    elif "math" in state["intent"]:
        return "math_expert"
    return "general_expert"


def code_expert(state: State) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "You are an expert programmer. Give concise, practical code answers."),
        ("human", "{input}"),
    ]) | llm | StrOutputParser()
    return {"response": chain.invoke({"input": state["input"]})}


def math_expert(state: State) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "You are a math tutor. Show step-by-step solutions clearly."),
        ("human", "{input}"),
    ]) | llm | StrOutputParser()
    return {"response": chain.invoke({"input": state["input"]})}


def general_expert(state: State) -> dict:
    chain = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise and friendly."),
        ("human", "{input}"),
    ]) | llm | StrOutputParser()
    return {"response": chain.invoke({"input": state["input"]})}


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("code_expert", code_expert)
graph.add_node("math_expert", math_expert)
graph.add_node("general_expert", general_expert)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("code_expert", END)
graph.add_edge("math_expert", END)
graph.add_edge("general_expert", END)

app = graph.compile()

if __name__ == "__main__":
    print_graph(app, "Intent Router")
    print("Router Chat — auto-routes to code/math/general expert (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break
        result = app.invoke({"input": user_input})
        print(f"[{result['intent']}] {result['response']}\n")
