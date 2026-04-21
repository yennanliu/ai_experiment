# LlamaIndex + OpenAI App

A minimal RAG (Retrieval-Augmented Generation) app using [LlamaIndex](https://www.llamaindex.ai/) and OpenAI. Drop your documents into `./data`, then ask questions about them in an interactive CLI.

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

Place `.txt`, `.pdf`, or other supported files into the `./data` folder. A sample file is already included.

## Run

```bash
uv run main.py
```

Then type your questions at the prompt. Enter `quit` to exit.

## How it works

1. Documents in `./data` are loaded and chunked
2. Chunks are embedded with `text-embedding-3-small` and stored in an in-memory vector index
3. Your question is embedded and the most relevant chunks are retrieved
4. `gpt-4o-mini` generates an answer grounded in those chunks
