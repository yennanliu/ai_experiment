# LlamaIndex + OpenAI Demo Playground

A collection of LlamaIndex demos using OpenAI. Drop your documents into `./data` and explore different indexing/query strategies via an interactive CLI menu.

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Configure API key**
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

**3. Add your documents**

Place `.txt`, `.pdf`, or other supported files into `./data`. A sample file is included.

## Run

```bash
uv run main.py
```

## Demos

| # | Name | Description |
|---|------|-------------|
| 1 | RAG Query | Ask questions over your docs using vector similarity search |
| 2 | Summarize | Generate a concise summary of all loaded documents |
| 3 | Chat Engine | Conversational Q&A with message history |
| 4 | Keyword Search | BM25-style retrieval — no embeddings required |

## Project Structure

```
app1/
├── main.py           # menu entry point
├── demos/
│   ├── rag_query.py      # demo 1
│   ├── summarize.py      # demo 2
│   ├── chat_engine.py    # demo 3
│   └── keyword_search.py # demo 4
├── data/             # put your documents here
├── .env              # OPENAI_API_KEY (not committed)
└── pyproject.toml
```
