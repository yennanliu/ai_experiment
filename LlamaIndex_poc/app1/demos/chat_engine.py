"""Demo 3: Chat Engine — conversational Q&A with memory over your documents."""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


def run():
    print("\n[Chat Engine] Loading documents from ./data ...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=False)

    print("Chat with your docs! Conversation history is maintained. ('quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            response = engine.chat(q)
            print(f"AI: {response}\n")
