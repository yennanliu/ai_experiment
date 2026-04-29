# Enterprise Knowledge Base — RAG with LangGraph + OpenAI

A simple enterprise-grade document Q&A system with pluggable advanced RAG techniques. Upload internal docs, ask questions, get grounded answers with source citations.

---

## Architecture

```
User
 │
 ▼
Flask (main.py)
 ├── POST /upload             ──► Chunker → Embed → ChromaDB
 ├── POST /load-samples       ──► Ingest data/*.txt into "samples" collection
 ├── POST /api/chunk-preview  ──► Preview chunk boundaries (no ingestion)
 │
 └── POST /chat              ──► LangGraph Pipeline
                                   │
                             ┌─────▼──────┐
                             │ transform  │  none / HyDE / multi-query
                             └─────┬──────┘  ⏱ timed
                                   │
                             ┌─────▼──────┐
                             │  retrieve  │  embed query → ChromaDB top-K
                             └─────┬──────┘  returns cosine similarity scores
                                   │
                             ┌─────▼──────┐
                             │   rerank   │  LLM scores each chunk 0–10
                             └─────┬──────┘  returns per-chunk scores
                                   │
                             ┌─────▼──────┐
                             │  generate  │  gpt-4o-mini, grounded answer
                             └─────┬──────┘  ⏱ timed
                                   │
                             JSON response (answer + scores + timings)
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

**Try it:** Use the "Compare chunking strategies" tool in the sidebar. Paste any text to see all three strategies side-by-side — chunk count, average length, and a sample of the first chunk.

### Cosine Similarity (Retrieval Score)

After embedding the query and retrieving chunks from ChromaDB, each chunk gets a **cosine similarity** score in [0, 1]. This measures how geometrically close the chunk embedding is to the query embedding.

```
score = 1 - L2_distance² / 2   (valid for unit-normalised embeddings like OpenAI's)
```

- Score near **1.0** → chunk embedding is nearly identical to the query embedding
- Score near **0.5** → moderately related
- Score near **0.0** → unrelated

**Why it matters:** Similarity is purely geometric — it picks up on surface vocabulary and phrasing. A chunk with high similarity may still be off-topic if the question is phrased differently from the document.

### HyDE — Hypothetical Document Embeddings

Standard RAG embeds the *question* and finds similar chunks. But questions and answers look very different in embedding space.

HyDE closes this gap:
1. Ask the LLM to write a *hypothetical answer* to the question
2. Embed that hypothetical answer (not the question)
3. Use that embedding for retrieval

```
Question ──► LLM ──► Hypothetical Answer ──► Embed ──► ChromaDB query
```

The hypothetical answer lives in the same embedding space as real document chunks, so retrieval is more accurate. **Cost:** 1 extra LLM call (visible in timings).

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

**Cost:** 1 extra LLM call for expansion + N retrieval calls instead of 1.

### Reranking (LLM Relevance Score)

After retrieval, chunks are sorted by cosine similarity — but similarity ≠ relevance. Reranking:
1. Sends each chunk to the LLM with the original question
2. Asks for a relevance score (0–10)
3. Re-sorts chunks by score, keeps top-K

```
Before rerank:  chunk A (sim=0.82) · chunk B (sim=0.79) · chunk C (sim=0.77)
LLM scores:     chunk A → 4        · chunk B → 9        · chunk C → 7
After rerank:   chunk B (9/10)     · chunk C (7/10)     · chunk A (4/10)
```

This is slower (1 LLM call per chunk) but significantly improves answer quality when retrieved chunks are noisy. **Cost:** K extra LLM calls — visible in the `rerank` timing.

---

## What the Debug Panel Shows

Open the **Debug** panel (top-right button) or click **Inspect ↗** on any answer to see the full pipeline trace.

### Pipeline Nodes + Timings
Each node shows its elapsed time. The **Latency Breakdown** bar chart makes the relative cost of each stage immediately visible:

```
transform  ████░░░░░░  0.85s   ← HyDE/multi-query LLM call
retrieve   █░░░░░░░░░  0.18s   ← ChromaDB query
rerank     ██████████  2.40s   ← K LLM calls (1 per chunk)
generate   ███░░░░░░░  0.62s   ← final answer
```

### Similarity Scores (before rerank)
Each chunk shows a colour-coded **similarity bar** and numeric score:
- Green bar → high cosine similarity (≥ 0.75)
- Amber bar → moderate similarity (0.5–0.75)
- Red bar  → low similarity (< 0.5)

### Rerank Scores (after rerank)
When reranking is on, each chunk also shows a **purple `N/10` badge** — the LLM's relevance judgment. Compare this against the similarity bar to see where the two rankings diverge: that divergence is exactly where reranking adds value.

### Query Variants (Multi-Query)
When Multi-Query is active, all generated phrasings are shown. Each was used as a separate retrieval query; results were merged and deduplicated.

### Hypothetical Document (HyDE)
When HyDE is active, the full hypothetical answer written by the LLM is shown — this is what was actually embedded for retrieval instead of the raw question.

---

## `/api/chunk-preview` — Chunking Strategy Comparison

```
POST /api/chunk-preview
Content-Type: application/json

{ "text": "...", "strategies": ["char", "sentence", "paragraph"] }
```

Response:
```json
{
  "char":      { "chunks": [...], "count": 12, "avg_chars": 480 },
  "sentence":  { "chunks": [...], "count": 8,  "avg_chars": 310 },
  "paragraph": { "chunks": [...], "count": 5,  "avg_chars": 720 }
}
```

No document is ingested. Use this to understand how your text will be split before committing to a strategy.

---

## `/chat` Response Shape

```json
{
  "answer": "...",
  "sources": ["hr_policy.txt"],
  "chunks_used": 3,
  "chunks": [
    {
      "text": "...",
      "source": "hr_policy.txt",
      "similarity": 0.812,      ← cosine similarity to query (always present)
      "rerank_score": 9         ← LLM relevance score 0-10 (only when rerank=true)
    }
  ],
  "transformed_queries": ["..."],
  "hyde_doc": "...",
  "pipeline": { "query_transform": "hyde", "rerank": true, "k": 5 },
  "timings": {
    "transform": 0.85,          ← seconds
    "retrieve":  0.18,
    "rerank":    2.40,
    "generate":  0.62
  }
}
```

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
3. **Pick chunking strategy** — applied on next upload/load. Click **Compare chunking strategies** to preview first.
4. **Configure retrieval** — set Top-K, choose query transform, toggle reranking
5. **Ask a question** — answer shows source tags, chunk count, and total latency (`⏱ Ns`)
6. **Inspect** — click **Inspect ↗** or open the **Debug** panel to see:
   - Per-node timing bars
   - Similarity scores and rerank scores side-by-side per chunk
   - Query variants (Multi-Query) or hypothetical doc (HyDE)

---

## Learning Experiments

| Question | What to do |
|---|---|
| Does reranking change the order? | Enable rerank, ask a question, open Debug. Compare the similarity bar vs the purple rerank badge for each chunk. |
| How expensive is reranking? | Check the Latency Breakdown. `rerank` time ≈ K × one LLM call. |
| HyDE vs raw query — does it retrieve different chunks? | Run the same question with None and then HyDE. Compare chunks in Debug. |
| Which chunking strategy fits my doc? | Use **Compare chunking strategies** in the sidebar before ingesting. |
| Where does my question sit in embedding space? | Open `/viz`, Query Probe tab, plot the question with and without HyDE. |

---

## Sample Documents (`data/`)

| File | Content |
|---|---|
| `hr_policy.txt` | Leave policy, remote work, performance reviews, expenses |
| `engineering_handbook.txt` | Git workflow, deployment process, on-call, tech stack, API standards |
| `product_faq.txt` | Billing, SSO, SLA, rate limits, data export, free trial |

**Example queries:**
- *"How many days of annual leave do employees get?"* (HR)
- *"What's the deployment rollout process?"* (Eng)
- *"Does the product support SSO, and which plans?"* (FAQ)
- *"What happens to my data if I cancel my account?"* (FAQ)

---

## Project Structure

```
app5/
├── main.py                  # Flask routes: /upload /chat /collections /load-samples /api/chunk-preview
├── rag/
│   ├── ingest.py            # load → chunk → embed → ChromaDB  +  retrieve() (returns similarity scores)
│   ├── chunkers.py          # char / sentence / paragraph strategies
│   ├── query_transform.py   # HyDE + multi-query expansion
│   └── rerank.py            # LLM-based relevance scoring (returns per-chunk scores)
├── agent/
│   ├── state.py             # RAGState TypedDict (includes retrieval_scores, rerank_scores, timings)
│   └── graph.py             # LangGraph: transform → retrieve → rerank → generate (all nodes timed)
├── data/
│   ├── hr_policy.txt
│   ├── engineering_handbook.txt
│   └── product_faq.txt
├── static/
│   ├── index.html           # Single-page UI with debug panel (scores, timings, chunk preview)
│   └── viz.html             # Embedding space visualizer (PCA 2D)
├── .env                     # API keys and config
└── pyproject.toml           # uv dependencies
```
