import os
from crewai.tools import tool


@tool("Read Text File")
def read_text_file(filepath: str) -> str:
    """Reads and returns the full content of a text file at the given path."""
    try:
        with open(filepath) as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"


@tool("Count Words")
def count_words(text: str) -> str:
    """Returns word count and sentence count for the given text."""
    words = len(text.split())
    sentences = sum(text.count(c) for c in ".!?")
    return f"Words: {words}, Sentences: {sentences}"


@tool("Save Report")
def save_report(content: str) -> str:
    """Saves report markdown content to output/report.md. Returns the saved path."""
    os.makedirs("output", exist_ok=True)
    path = "output/report.md"
    with open(path, "w") as f:
        f.write(content)
    return f"Report saved to {path}"
