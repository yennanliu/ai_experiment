"""
LangChain Demo - Core Concepts
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def demo_basic_chat():
    """Demo 1: Basic Chat Model"""
    print("=== Demo 1: Basic Chat ===")

    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("What is LangChain in one sentence?")
    print(response.content)
    print()


def demo_prompt_template():
    """Demo 2: Prompt Templates"""
    print("=== Demo 2: Prompt Template ===")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains concepts simply."),
        ("user", "Explain {topic} in {num_sentences} sentences.")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "machine learning", "num_sentences": 2})
    print(result)
    print()


def demo_chain():
    """Demo 3: LCEL Chain (LangChain Expression Language)"""
    print("=== Demo 3: LCEL Chain ===")

    # Chain 1: Generate a topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Give me one interesting topic about {subject}. Just the topic, nothing else."
    )

    # Chain 2: Write about that topic
    write_prompt = ChatPromptTemplate.from_template(
        "Write a brief haiku about: {topic}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()

    # Compose chains using LCEL pipe operator
    chain = (
        topic_prompt
        | llm
        | parser
        | (lambda topic: {"topic": topic})
        | write_prompt
        | llm
        | parser
    )

    result = chain.invoke({"subject": "nature"})
    print(result)
    print()


if __name__ == "__main__":
    demo_basic_chat()
    demo_prompt_template()
    demo_chain()
