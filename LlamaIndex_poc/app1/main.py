from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

DEMOS = {
    "1": ("RAG Query          — Q&A over your documents", "demos.rag_query"),
    "2": ("Summarize          — summarize all documents", "demos.summarize"),
    "3": ("Chat Engine        — conversational Q&A with memory", "demos.chat_engine"),
    "4": ("Keyword Search     — retrieval without embeddings", "demos.keyword_search"),
}


def main():
    print("\nLlamaIndex Demo Playground")
    print("=" * 40)
    for key, (label, _) in DEMOS.items():
        print(f"  {key}. {label}")
    print("  q. Quit")
    print()

    choice = input("Select a demo: ").strip().lower()
    if choice == "q":
        return

    if choice not in DEMOS:
        print("Invalid choice.")
        return

    _, module_path = DEMOS[choice]
    import importlib
    demo = importlib.import_module(module_path)
    demo.run()


if __name__ == "__main__":
    main()
