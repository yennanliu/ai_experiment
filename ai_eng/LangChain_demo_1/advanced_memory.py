"""
Advanced LangChain Demo - Conversation Memory
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()


def demo_manual_memory():
    """Manually manage conversation history"""
    print("=== Manual Memory Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])

    chain = prompt | llm

    # Simulate conversation with manual history
    history = []

    conversations = [
        "My name is Alice.",
        "I work as a data scientist.",
        "What's my name and job?",
    ]

    for user_input in conversations:
        print(f"User: {user_input}")

        response = chain.invoke({"history": history, "input": user_input})
        print(f"AI: {response.content}\n")

        # Update history
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response.content))


def demo_auto_memory():
    """Automatic memory management with RunnableWithMessageHistory"""
    print("=== Auto Memory Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])

    chain = prompt | llm

    # Store for multiple sessions
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Wrap chain with memory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Conversation in session "user-1"
    config = {"configurable": {"session_id": "user-1"}}

    conversations = [
        "I'm learning Python.",
        "What framework should I use for web development?",
        "What was I learning again?",
    ]

    for user_input in conversations:
        print(f"User: {user_input}")
        response = chain_with_memory.invoke({"input": user_input}, config=config)
        print(f"AI: {response.content}\n")


if __name__ == "__main__":
    demo_manual_memory()
    print("-" * 50 + "\n")
    demo_auto_memory()
