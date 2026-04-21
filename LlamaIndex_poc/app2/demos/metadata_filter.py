"""Demo 2: Metadata Filtering
Attach topic metadata to each document during ingestion, then
filter retrieval to only documents matching a chosen topic.
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Topics that appear in the sample data
TOPICS = [
    "machine learning",
    "llm",
    "vector databases",
    "rag systems",
    "semantic search",
    "data indexing",
    "embedding models",
    "ai agents",
    "nlp pipelines",
    "knowledge graphs",
]


def _load_nodes_with_metadata() -> list[TextNode]:
    """Parse sample.txt and attach topic metadata to each node."""
    nodes = []
    with open("data/sample.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            doc_id, title, content = parts[0].strip(), parts[1].strip(), parts[2].strip()

            # Derive topic from title (first two words)
            topic = "unknown"
            for t in TOPICS:
                if t in title.lower():
                    topic = t
                    break

            nodes.append(TextNode(
                text=f"{title}: {content}",
                metadata={"topic": topic, "doc_id": doc_id},
            ))
    return nodes


def run():
    print("\n[Metadata Filter] Building index with topic metadata ...")
    nodes = _load_nodes_with_metadata()
    index = VectorStoreIndex(nodes)

    print("\nAvailable topics:")
    for i, t in enumerate(TOPICS, 1):
        print(f"  {i:2}. {t}")
    print()

    while True:
        choice = input("Enter topic number (or 'q' to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(TOPICS)):
            print("Invalid choice.")
            continue

        topic = TOPICS[int(choice) - 1]
        filters = MetadataFilters(filters=[ExactMatchFilter(key="topic", value=topic)])
        engine = index.as_query_engine(filters=filters, similarity_top_k=5)

        query = input(f"Ask something about '{topic}': ").strip()
        if query:
            response = engine.query(query)
            print(f"\nAI (filtered to '{topic}'): {response}\n")
