"""
Advanced LangChain Demo - Agents with Tools
"""

import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


# Define custom tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python syntax."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (mock data)."""
    weather_data = {
        "tokyo": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "new york": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def search_knowledge(query: str) -> str:
    """Search internal knowledge base (mock data)."""
    knowledge = {
        "langchain": "LangChain is a framework for building LLM applications.",
        "python": "Python is a high-level programming language.",
        "ai": "AI refers to artificial intelligence systems.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No relevant information found."


def demo_tool_calling():
    """Agent that uses tools to answer questions"""
    print("=== Agent with Tools Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [calculate, get_weather, search_knowledge]
    llm_with_tools = llm.bind_tools(tools)

    queries = [
        "What's 25 * 4 + 10?",
        "What's the weather in Tokyo?",
        "Tell me about LangChain",
    ]

    for query in queries:
        print(f"Q: {query}")

        # Get LLM response with tool calls
        response = llm_with_tools.invoke([HumanMessage(content=query)])

        if response.tool_calls:
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Find and execute the tool
                tool_map = {t.name: t for t in tools}
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    print(f"A: [{tool_name}] {result}\n")
        else:
            print(f"A: {response.content}\n")


def demo_react_agent():
    """ReAct pattern: Reason + Act iteratively"""
    print("=== ReAct Agent Demo ===\n")

    from langgraph.prebuilt import create_react_agent

    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [calculate, get_weather]

    # Create ReAct agent
    agent = create_react_agent(llm, tools)

    # Complex query requiring multiple steps
    query = "What's the weather in London, and what's 100 divided by 5?"
    print(f"Q: {query}")

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # Get final response
    final_message = result["messages"][-1]
    print(f"A: {final_message.content}\n")


if __name__ == "__main__":
    demo_tool_calling()
    demo_react_agent()
