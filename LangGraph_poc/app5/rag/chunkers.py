"""Pluggable chunking strategies."""
import re


def chunk_char(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Sliding window over raw characters."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def chunk_sentence(text: str, window: int = 5, overlap: int = 1) -> list[str]:
    """Group sentences into overlapping windows."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks = []
    for i in range(0, len(sentences), window - overlap):
        group = sentences[i:i + window]
        if group:
            chunks.append(" ".join(group))
    return chunks


def chunk_paragraph(text: str, max_size: int = 800) -> list[str]:
    """Split on blank lines; merge short paragraphs, split oversized ones."""
    raw = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks, buf = [], ""
    for para in raw:
        if len(buf) + len(para) < max_size:
            buf = (buf + "\n\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    # split any oversized single paragraphs with char chunker as fallback
    final = []
    for c in chunks:
        if len(c) > max_size * 1.5:
            final.extend(chunk_char(c, size=max_size, overlap=80))
        else:
            final.append(c)
    return final


STRATEGIES = {
    "char": chunk_char,
    "sentence": chunk_sentence,
    "paragraph": chunk_paragraph,
}


def chunk(text: str, strategy: str = "char") -> list[str]:
    fn = STRATEGIES.get(strategy, chunk_char)
    return fn(text)
