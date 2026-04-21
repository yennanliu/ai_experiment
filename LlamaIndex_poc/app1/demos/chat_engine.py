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

if __name__ == "__main__":
    from dotenv import load_dotenv
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    load_dotenv()
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    run()
