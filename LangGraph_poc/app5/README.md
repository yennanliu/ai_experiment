# Enterprise Knowledge Base — RAG with LangGraph + OpenAI

A simple enterprise-grade document Q&A system with pluggable advanced RAG techniques. Upload internal docs, ask questions, get grounded answers with source citations.

---

## Architecture

```
User
 │
 ▼
Flask (main.py)
 ├── POST /upload        ──► Chunker → Embed → ChromaDB
 ├── POST /load-samples  ──► Ingest data/*.txt into "samples" collection
 │
 └── POST /chat         ──► LangGraph Pipeline
                               │
                         ┌─────▼──────┐
                         │ transform  │  none / HyDE / multi-query
                         └─────┬──────┘
                               │
                         ┌─────▼──────┐
                         │  retrieve  │  embed query → ChromaDB top-K
                         └─────┬──────┘
                               │
                         ┌─────▼──────┐
                         │   rerank   │  LLM scores each chunk 0–10
                         └─────┬──────┘
                               │
                         ┌─────▼──────┐
                         │  generate  │  gpt-4o-mini, grounded answer
                         └─────┬──────┘
                               │
                         JSON response
```

**Components:**

| Layer | Tech | Role |
|---|---|---|
| Web | Flask | HTTP API + single-page UI |
| Orchestration | LangGraph | RAG pipeline as a typed state graph |
| LLM | OpenAI `gpt-4o-mini` | Answer generation, HyDE, multi-query, reranking |
| Embeddings | OpenAI `text-embedding-3-small` | Chunk & query vectorization |
| Vector store | ChromaDB (persistent) | Similarity search |
| Chunking | Char / Sentence / Paragraph | Pluggable text splitting |

---

## RAG Concepts

### Chunking Strategies

Documents are split before embedding. Strategy is chosen at ingest time and affects retrieval quality.

| Strategy | How it works | Best for |
|---|---|---|
| **Char** | Sliding window (500 chars, 50 overlap) | Quick baseline |
| **Sentence** | Group N sentences with 1-sentence overlap | Conversational/prose docs |
| **Paragraph** | Split on blank lines, merge short paragraphs | Structured docs, policies |

```
Char:       [  chunk 1  ]
                      [  chunk 2  ]   ← overlap preserves context at boundaries
                                [  chunk 3  ]

Paragraph:  [ full paragraph A ]  [ paragraph B + C (merged if short) ]
```

### HyDE — Hypothetical Document Embeddings

Standard RAG embeds the *question* and finds similar chunks. But questions and answers look very different in embedding space.

HyDE closes this gap:
1. Ask the LLM to write a *hypothetical answer* to the question
2. Embed that hypothetical answer (not the question)
3. Use that embedding for retrieval

```
Question ──► LLM ──► Hypothetical Answer ──► Embed ──► ChromaDB query
```

The hypothetical answer lives in the same embedding space as real document chunks, so retrieval is more accurate.

### Multi-Query

A single question phrased one way may miss relevant chunks phrased differently. Multi-Query:
1. Rewrites the question into 3 alternative phrasings
2. Retrieves separately for each variant
3. Deduplicates and merges results

```
"How many vacation days?"
  ├── "What is the annual leave entitlement?"
  ├── "How much PTO do employees receive?"
  └── "Leave policy days off per year"
        └── each → retrieve → union → deduplicate
```

### Reranking

After retrieval, chunks are sorted by embedding similarity — but similarity ≠ relevance. Reranking:
1. Sends each chunk to the LLM with the original question
2. Asks for a relevance score (0–10)
3. Re-sorts chunks by score, keeps top-K

This is slower (1 LLM call per chunk) but significantly improves answer quality when retrieved chunks are noisy.

---

## Install

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
cd app5
uv sync
```

Set your OpenAI API key in `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads
```

---

## Run

```bash
uv run python main.py
```

Open [http://localhost:5001](http://localhost:5001).

---

## Usage

1. **Load samples** — click **⚡ Load sample docs** to ingest the 3 built-in docs into the `samples` collection
2. **Or upload** — drag & drop a PDF, TXT, or MD file
3. **Pick chunking strategy** — applied on next upload/load
4. **Configure retrieval** — set Top-K, choose query transform, toggle reranking
5. **Ask a question** — answer includes source tags and chunk count
6. **Inspect** — click **Inspect ↗** or open the **Debug panel** to see every pipeline step: query variants, hypothetical doc, retrieved chunks with char counts

---

## Sample Documents (`data/`)

| File | Content |
|---|---|
| `hr_policy.txt` | Leave policy, remote work, performance reviews, expenses |
| `engineering_handbook.txt` | Git workflow, deployment process, on-call, tech stack, API standards |
| `product_faq.txt` | Billing, SSO, SLA, rate limits, data export, free trial |

**Example queries to try:**
- *"How many days of annual leave do employees get?"* (HR)
- *"What's the deployment rollout process?"* (Eng)
- *"Does the product support SSO, and which plans?"* (FAQ)
- *"What happens to my data if I cancel my account?"* (FAQ)

---

## Project Structure

```
app5/
├── main.py                  # Flask routes: /upload /chat /collections /load-samples
├── rag/
│   ├── ingest.py            # load → chunk → embed → ChromaDB  +  retrieve()
│   ├── chunkers.py          # char / sentence / paragraph strategies
│   ├── query_transform.py   # HyDE + multi-query expansion
│   └── rerank.py            # LLM-based relevance scoring & resorting
├── agent/
│   ├── state.py             # RAGState TypedDict
│   └── graph.py             # LangGraph: transform → retrieve → rerank → generate
├── data/
│   ├── hr_policy.txt
│   ├── engineering_handbook.txt
│   └── product_faq.txt
├── static/
│   └── index.html           # Single-page UI with debug panel
├── .env                     # API keys and config
└── pyproject.toml           # uv dependencies
```
