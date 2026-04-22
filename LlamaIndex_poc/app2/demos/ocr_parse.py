"""Demo 7: OCR / Document Parsing with LlamaParse
Upload any file (PDF, DOCX, PPTX, images, TXT, …) and parse it
using LlamaParse. The parsed markdown is then handed to GPT-4o-mini
for a quick Q&A session over the document contents.
"""
import os
from llama_cloud import LlamaCloud
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document


def _parse_file(path: str) -> str:
    """Upload a file to LlamaParse and return the full markdown text."""
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "LLAMA_CLOUD_API_KEY is not set. "
            "Get a free key at https://cloud.llamaindex.ai/"
        )

    client = LlamaCloud(token=api_key)

    print(f"  Uploading '{path}' ...")
    with open(path, "rb") as f:
        file_obj = client.files.upload_file(upload_file=f)

    print("  Parsing (this may take a moment) ...")
    result = client.parsing.parse(
        file_id=file_obj.id,
        tier="fast",           # 'fast' is cheapest; change to 'agentic' for complex PDFs
        expand=["markdown"],
    )

    pages = result.markdown.pages
    return "\n\n".join(p.markdown for p in pages if p.markdown)


def run():
    print("\n[OCR / Document Parse] Powered by LlamaParse + GPT-4o-mini")
    print("Supported: PDF, DOCX, PPTX, XLSX, images (PNG/JPG), TXT, and 130+ more\n")

    path = input("Enter path to file: ").strip().strip("'\"")
    if not path:
        print("No path provided.")
        return

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        markdown_text = _parse_file(path)
    except Exception as e:
        print(f"Parsing failed: {e}")
        return

    print(f"\n--- Parsed content preview (first 500 chars) ---\n{markdown_text[:500]}\n---\n")

    # Build an in-memory index over the parsed content for Q&A
    doc = Document(text=markdown_text, metadata={"source": os.path.basename(path)})
    index = VectorStoreIndex.from_documents([doc])
    engine = index.as_query_engine(similarity_top_k=3)

    print("Index built. Ask questions about the document. ('quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print(f"AI: {engine.query(q)}\n")
