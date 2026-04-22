"""Demo 7: OCR / Document Parsing via GPT-4o Vision
Parse scanned PDFs and images using GPT-4o's vision capability — no third-party
OCR API required. Each page/image is sent to GPT-4o as an ImageBlock, which
extracts the text. The extracted text is then indexed for Q&A.

Supported:
  - Scanned / image-based PDFs  →  requires: pymupdf  (pip install pymupdf)
  - Images (PNG, JPG, JPEG, WEBP, GIF)
  - Text-layer PDFs, DOCX, TXT  →  handled by SimpleDirectoryReader (no vision needed)
"""
import os
import tempfile
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_EXTS  = {".txt", ".md", ".csv", ".html", ".docx", ".pptx", ".xlsx"}

OCR_PROMPT = (
    "You are an OCR engine. Extract ALL text from this image exactly as it appears. "
    "Preserve layout, headings, bullet points, and table structure using markdown. "
    "Return only the extracted text — no commentary."
)


def _ocr_image(llm: OpenAI, image_path: str) -> str:
    """Send a single image to GPT-4o and return the extracted text."""
    msg = ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path=image_path),
            TextBlock(text=OCR_PROMPT),
        ],
    )
    return llm.chat([msg]).message.content


def _parse_pdf_via_vision(llm: OpenAI, pdf_path: str) -> list[Document]:
    """Convert each PDF page to a PNG, OCR it with GPT-4o, return Documents."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF OCR.\n"
            "Install it with:  uv add pymupdf"
        )

    docs = []
    pdf = fitz.open(pdf_path)

    with tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(pdf, start=1):
            print(f"  OCR page {i}/{len(pdf)} ...")
            img_path = os.path.join(tmp, f"page_{i}.png")
            # Render at 2× resolution for better OCR accuracy
            pix = page.get_pixmap(dpi=200)
            pix.save(img_path)

            text = _ocr_image(llm, img_path)
            if text.strip():
                docs.append(Document(
                    text=text,
                    metadata={"source": os.path.basename(pdf_path), "page": i},
                ))

    pdf.close()
    return docs


def _parse_image(llm: OpenAI, path: str) -> list[Document]:
    """OCR a single image file."""
    print("  OCR image ...")
    text = _ocr_image(llm, path)
    return [Document(text=text, metadata={"source": os.path.basename(path)})]


def _parse_text_file(path: str) -> list[Document]:
    """Use SimpleDirectoryReader for files that already have a text layer."""
    print("  Extracting text via SimpleDirectoryReader ...")
    return SimpleDirectoryReader(input_files=[path]).load_data()


def run():
    print("\n[OCR / Document Parse] Powered by GPT-4o Vision — no extra API key needed")
    print("  Scanned PDFs / images  → GPT-4o vision OCR")
    print("  Text PDFs / DOCX / TXT → direct text extraction\n")

    path = input("Enter path to file: ").strip().strip("'\"")
    if not path:
        print("No path provided.")
        return
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    ext = Path(path).suffix.lower()
    # Use gpt-4o (vision) for OCR; the global Settings LLM (gpt-4o-mini) is used for Q&A
    vision_llm = OpenAI(model="gpt-4o")

    try:
        if ext == ".pdf":
            docs = _parse_pdf_via_vision(vision_llm, path)
        elif ext in IMAGE_EXTS:
            docs = _parse_image(vision_llm, path)
        elif ext in TEXT_EXTS:
            docs = _parse_text_file(path)
        else:
            print(f"Unsupported extension '{ext}'. Trying SimpleDirectoryReader anyway ...")
            docs = _parse_text_file(path)
    except Exception as e:
        print(f"Parsing failed: {e}")
        return

    if not docs:
        print("No text extracted from the file.")
        return

    full_text = "\n\n".join(d.get_content() for d in docs)
    print(f"\n--- Extracted text preview (first 500 chars) ---\n{full_text[:500]}\n---\n")

    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine(similarity_top_k=3)

    print("Index built. Ask questions about the document. ('quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print(f"AI: {engine.query(q)}\n")
