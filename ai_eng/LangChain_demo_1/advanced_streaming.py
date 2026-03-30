"""
Advanced LangChain Demo - Streaming
"""

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def demo_sync_streaming():
    """Synchronous token streaming"""
    print("=== Sync Streaming Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

    prompt = ChatPromptTemplate.from_template(
        "Write a short poem about {topic} (4 lines max)."
    )

    chain = prompt | llm | StrOutputParser()

    print("Streaming response:")
    for chunk in chain.stream({"topic": "coding"}):
        print(chunk, end="", flush=True)
    print("\n")


async def demo_async_streaming():
    """Asynchronous token streaming"""
    print("=== Async Streaming Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

    prompt = ChatPromptTemplate.from_template(
        "Explain {concept} in one sentence."
    )

    chain = prompt | llm | StrOutputParser()

    print("Async streaming response:")
    async for chunk in chain.astream({"concept": "machine learning"}):
        print(chunk, end="", flush=True)
    print("\n")


def demo_stream_events():
    """Stream events from chain execution"""
    print("=== Stream Events Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template("What is {topic}?")
    chain = prompt | llm | StrOutputParser()

    print("Events during execution:")
    for event in chain.stream_events({"topic": "Python"}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                print(chunk, end="", flush=True)
    print("\n")


async def demo_parallel_streaming():
    """Stream multiple chains in parallel"""
    print("=== Parallel Streaming Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

    prompt = ChatPromptTemplate.from_template(
        "In 10 words or less, what is {topic}?"
    )

    chain = prompt | llm | StrOutputParser()

    topics = ["Python", "JavaScript", "Rust"]

    async def stream_topic(topic):
        result = []
        async for chunk in chain.astream({"topic": topic}):
            result.append(chunk)
        return topic, "".join(result)

    # Run all streams concurrently
    results = await asyncio.gather(*[stream_topic(t) for t in topics])

    for topic, response in results:
        print(f"{topic}: {response}")


if __name__ == "__main__":
    demo_sync_streaming()

    print("-" * 50 + "\n")
    asyncio.run(demo_async_streaming())

    print("-" * 50 + "\n")
    demo_stream_events()

    print("-" * 50 + "\n")
    asyncio.run(demo_parallel_streaming())
