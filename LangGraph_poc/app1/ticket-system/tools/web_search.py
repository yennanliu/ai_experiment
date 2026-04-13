"""Web search tool using DuckDuckGo"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 3) -> str:
    """Search the web and return a summary of top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"[{r['title']}] {r['body']}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Search failed: {exc}"
