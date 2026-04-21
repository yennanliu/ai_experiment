from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def main():
    print("Loading documents from ./data ...")
    documents = SimpleDirectoryReader("data").load_data()

    print(f"Indexing {len(documents)} document(s)...")
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    print("\nReady! Type your question (or 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        response = query_engine.query(question)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    main()
