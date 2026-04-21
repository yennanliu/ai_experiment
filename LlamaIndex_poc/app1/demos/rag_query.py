"""Demo 1: RAG Query — ask questions over your documents."""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


def run():
    print("\n[RAG Query] Loading documents from ./data ...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    engine = index.as_query_engine()

    print("Ready! Ask questions about your documents. ('quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print(f"AI: {engine.query(q)}\n")

if __name__ == "__main__":
    from dotenv import load_dotenv
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    load_dotenv()
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    run()
