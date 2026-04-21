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
