from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

DEMOS = {
    "1": ("Sub-Question Query  — decompose complex questions across docs",  "demos.sub_question"),
    "2": ("Metadata Filter     — retrieve by topic/category filter",        "demos.metadata_filter"),
    "3": ("ReAct Agent         — LLM agent with search + calculator tools", "demos.react_agent"),
    "4": ("Reranking           — two-stage retrieval with reranker",        "demos.reranking"),
    "5": ("Index Persistence   — save & reload index from disk",            "demos.persistence"),
    "6": ("Router Query Engine — auto-route to best index strategy",        "demos.router"),
    "7": ("OCR / Document Parse — parse PDF/DOCX/images via GPT-4o vision", "demos.ocr_parse"),
    "8": ("Multi-Agent Workflow — Researcher → Analyst → Writer pipeline",  "demos.agent_workflow"),
}


def main():
    print("\nLlamaIndex Advanced Demo Playground")
    print("=" * 50)
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

    import importlib
    _, module_path = DEMOS[choice]
    demo = importlib.import_module(module_path)
    demo.run()


if __name__ == "__main__":
    main()
