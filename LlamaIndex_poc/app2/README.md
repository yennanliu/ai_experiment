# LlamaIndex Advanced Demo Playground

An interactive CLI playground exploring advanced LlamaIndex retrieval and agent patterns, each implemented as a standalone, runnable demo.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key

## Setup

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
uv sync
```

## Running

```bash
uv run python main.py
```

Select a demo by number from the menu.

## Demos

| # | Demo | Description |
|---|------|-------------|
| 1 | Sub-Question Query | Decomposes a complex question into sub-questions, queries each independently, then synthesizes a final answer |
| 2 | Metadata Filter | Attaches topic metadata at ingestion time and filters retrieval to documents matching a chosen topic |
| 3 | ReAct Agent | LLM agent that reasons across a document search tool and a calculator tool |
| 4 | Reranking | Two-stage retrieval: vector similarity fetches 10 candidates, then `LLMRerank` reranks to top 3 |
| 5 | Index Persistence | Builds a vector index once, persists it to `./storage/`, and reloads it on subsequent runs without re-embedding |
| 6 | Router Query Engine | Maintains a vector index and a summary index; the LLM router picks the best one per query |

All demos use `gpt-4o-mini` (LLM) and `text-embedding-3-small` (embeddings), configured globally in `main.py`.

## Data

`data/sample.txt` contains 100 synthetic AI/ML documents in pipe-delimited format:

```
<id> | <title> | <content>
```

Topics covered: machine learning, LLMs, vector databases, RAG systems, semantic search, data indexing, embedding models, AI agents, NLP pipelines, knowledge graphs.
