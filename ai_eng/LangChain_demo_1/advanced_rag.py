"""
Advanced LangChain Demo - RAG (Retrieval Augmented Generation)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def demo_simple_rag():
    """RAG: Answer questions using custom knowledge base"""
    print("=== RAG Demo ===\n")

    # Sample documents (your knowledge base)
    documents = [
        "LangChain was created by Harrison Chase in October 2022.",
        "LangChain supports Python and JavaScript/TypeScript.",
        "LCEL (LangChain Expression Language) uses the pipe operator for chaining.",
        "LangGraph is used for building stateful, multi-actor applications.",
        "LangSmith provides observability and debugging for LLM applications.",
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAG prompt
    prompt = ChatPromptTemplate.from_template("""
Answer based only on the context below. If unsure, say "I don't know."

Context: {context}

Question: {question}
""")

    llm = ChatOpenAI(model="gpt-4o-mini")

    # RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test queries
    questions = [
        "Who created LangChain?",
        "What languages does LangChain support?",
        "What is LangGraph used for?",
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {chain.invoke(q)}\n")


if __name__ == "__main__":
    demo_simple_rag()
