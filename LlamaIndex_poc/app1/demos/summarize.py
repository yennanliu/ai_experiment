"""Demo 2: Summarization — generate a summary of all loaded documents."""
from llama_index.core import SummaryIndex, SimpleDirectoryReader


def run():
    print("\n[Summarize] Loading documents from ./data ...")
    documents = SimpleDirectoryReader("data").load_data()
    index = SummaryIndex.from_documents(documents)
    engine = index.as_query_engine()

    print("Generating summary...\n")
    response = engine.query("Provide a concise summary of all the documents.")
    print(f"Summary:\n{response}\n")

if __name__ == "__main__":
    from dotenv import load_dotenv
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    load_dotenv()
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    run()
