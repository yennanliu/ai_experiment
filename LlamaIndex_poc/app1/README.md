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

**Interactive menu** (pick a demo at runtime):
```bash
uv run main.py
```

**Run a specific demo directly:**
```bash
uv run demos/rag_query.py       # RAG Q&A
uv run demos/summarize.py       # Summarization
uv run demos/chat_engine.py     # Chat with memory
uv run demos/keyword_search.py  # Keyword search
```

## Demos

| # | Name | Command | Description |
|---|------|---------|-------------|
| 1 | RAG Query | `uv run demos/rag_query.py` | Ask questions over your docs using vector similarity search |
| 2 | Summarize | `uv run demos/summarize.py` | Generate a concise summary of all loaded documents |
| 3 | Chat Engine | `uv run demos/chat_engine.py` | Conversational Q&A with message history |
| 4 | Keyword Search | `uv run demos/keyword_search.py` | BM25-style retrieval — no embeddings required |

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
