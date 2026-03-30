"""
Basic LangGraph Demo - A simple chatbot with state management.
"""

import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")


# Define the chatbot node
def chatbot(state: State) -> State:
    """Process messages and generate a response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build the graph
def create_graph():
    """Create and compile the LangGraph."""
    graph_builder = StateGraph(State)

    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot)

    # Define edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # Compile the graph
    return graph_builder.compile()


def main():
    """Run the chatbot in a loop."""
    graph = create_graph()

    print("LangGraph Chatbot Demo")
    print("Type 'quit' to exit\n")

    messages = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message and invoke graph
        messages.append({"role": "user", "content": user_input})
        result = graph.invoke({"messages": messages})

        # Get the assistant's response
        assistant_message = result["messages"][-1]
        messages = result["messages"]

        print(f"Assistant: {assistant_message.content}\n")


if __name__ == "__main__":
    main()
