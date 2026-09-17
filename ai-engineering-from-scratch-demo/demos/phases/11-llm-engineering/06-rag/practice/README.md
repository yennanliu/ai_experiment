<!-- generated:start -->
# 11-llm-engineering / 06-rag

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/06-rag/) · upstream spec
`phases/11-llm-engineering/06-rag/docs/en.md`

```bash
uv run demo practice run 06-rag --ex 1
uv run demo explain 06-rag --ex 1
uv run pytest demos/phases/11-llm-engineering/06-rag
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace the TF-IDF embeddings with a simple bag-of-words approach (binary: 1 if word present,… | code | T0 | `ex01_tf_idf_and_binary_agree_on_every_top_hit.py` |
| 2 | Experiment with chunk sizes: try 50, 100, 200, and 500 words on the same document set. For ea… | code | T0 | `ex02_the_sweet_spot_is_flat_and_the_first_size_hangs.py` |
| 3 | Add metadata to each chunk (source document name, chunk position). Modify the prompt template… | code | T0 | `ex03_the_source_names_exist_and_never_reach_the_prompt.py` |
| 4 | Implement a simple evaluation: given 10 question-answer pairs, run each question through the… | code | T0 | `ex04_the_metric_it_names_and_the_metric_it_defines_disagree.py` |
| 5 | Build a conversation-aware RAG pipeline: maintain a history of the last 3 exchanges and inclu… | code | T0 | `ex05_the_follow_up_fails_on_a_question_mark.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib — `math` and `collections` — so all five exercises are
**T0** and run in CI. What they find is that the retriever is the part that
works: recall at 1 is 10 of 10 across a hand-labelled evaluation set. Every
exercise that looks like it is about retrieval quality turns out to be about
something else — the corpus being five documents, `top_k` equalling the corpus
size, `str.split()` being the tokeniser, or the metric measuring `k`.

### 1 — TF-IDF and binary bag-of-words agree on every top hit

| query | TF-IDF top-3 | binary top-3 | same rank 1 |
|---|---|---|---|
| refund window for enterprise customers | `[0, 4, 3]` | `[0, 4, 1]` | ✅ |
| how do I reset my password | `[0, 1, 2]` | `[0, 1, 2]` | ✅ |
| what is the api rate limit | `[3, 0, 4]` | `[3, 1, 4]` | ✅ |
| how long is data retained | `[2, 1, 3]` | `[2, 1, 3]` | ✅ |
| what is the uptime SLA | `[4, 1, 3]` | `[4, 1, 3]` | ✅ |

**ANSWER: identical rank-1 on all five, and both score 3 of 5 on rank-1
relevance.** The exercise's prediction is not wrong — it is unmeasurable here,
because the two embeddings only disagree about chunks that are already
irrelevant.

**MECHANISM: five documents cannot spread the IDF.**

```text
compute_idf = log((n+1)/(doc_count+1)) + 1,  n = 5
→ the entire 270-word vocabulary lies in [1.000, 2.099]
```

Rare words *are* weighted higher, and the weight is at most doubled.

**FINDING: TF is the other half, and it cancels.** `compute_tf` divides by the
chunk's word count, and at the default chunk size every chunk is a whole
document: 91, 92, 91, 95, 102 words — a spread of 1.12×.

**CONTROL: split to 11 chunks.** The IDF band widens to [1.087, 2.792], rank-1
agreement drops to 3 of 5, and rank-1 relevance becomes **TF-IDF 3, binary 1**.
The claim is correct; five whole-document chunks cannot show it.

### 2 — the sweet spot is flat, and the first size hangs

```python
start += chunk_size - overlap      #  50 - 50  ==  0
```

**ANSWER: chunk size 50 does not terminate.** The pipeline's default overlap is
50, so `RAGPipeline(chunk_size=50)` never returns from `index()`. A bounded clone
emits 1,000 identical chunks with `start` still at 0.

| chunk size | chunks | relevant in top-3 |
|---:|---:|---:|
| 50 | — | **hangs** |
| 100 | 11 | 5 / 5 |
| 200 | 5 | 5 / 5 |
| 500 | 5 | 5 / 5 |

**ANSWER: flat at the ceiling.** The metric the exercise defines is saturated
over its own range. And 200 and 500 hold byte-identical text, because the longest
document is 102 words.

**FINDING: at the pipeline's own `top_k` the question is empty.** `RAGPipeline`
defaults to `top_k=5` against a 5-chunk index — it retrieves the whole corpus for
every query.

**FINDING: what this pipeline is sensitive to is tokenisation.**
`build_vocabulary` is `doc.lower().split()`, so 12 of the 270 entries are a
trailing-punctuation form of another:

```text
enterprise.  plans.  sla.  customers.  hours.  minutes.  month.  credits:  …
```

### 3 — the source names already exist, and never reach the prompt

```python
self.sources = ['doc_0', 'doc_1', 'doc_2', 'doc_3', 'doc_4']    # built by index()
query(...)  ->  {'chunk': …, 'source': 'doc_2', 'score': …}     # returned per hit
build_rag_prompt(query, chunk_texts)                            # texts only
```

**ANSWER: half of it ships, and that half is unused.** The document names appear
in **0 of 5** prompts; the template labels sources `[Source 1]`, `[Source 2]`,
`[Source 3]` — positionally.

**FINDING: the labels are per-query.** Chunk 0 is labelled **1, 2 and 3** across
the five queries. "As stated in Source 1" identifies a rank, not a document.

**FINDING: chunk position is not stored, and is one pass away.** The pipeline
keeps `chunks, embeddings, vocab, idf, sources` — no position. At chunk size 100
the global `index` coincides with the in-document position for only **2 of 11**
chunks; `sources` makes it derivable.

**ANSWER: attribution reaches the prompt, and changes nothing.** A template
carrying `[doc_2, chunk 1]` puts the name in 5 of 5 prompts and leaves the answer
byte-identical in 5 of 5 — because `simple_generate` re-derives the query by
splitting the prompt on `"question:"` and then scans `retrieved_chunks` directly.
Prompt text is never read.

**FINDING: so the citation is a lookup, not a request.** `simple_generate`
returns a verbatim sentence, which resolves to exactly one chunk for 5 of 5
queries — and `query` already knows that chunk's source.

### 4 — the metric it names and the metric it defines disagree

| k | recall (gold chunk retrieved) | precision (the exercise's definition) | ceiling |
|---:|---:|---:|---:|
| 1 | **10 / 10** | **1.000** | 1.000 |
| 3 | **10 / 10** | 0.367 | 0.367 |
| 5 | **10 / 10** | 0.220 | 0.220 |

**ANSWER: the sentence defines precision@k and calls it recall@k.** Retrieval is
perfect and constant; the number the exercise asks for falls by a factor of five
over the same runs.

**MECHANISM: each answer lives in one chunk.** 9 of the 10 answer strings appear
in exactly one of the 5 chunks, so precision@k is capped at ~1/k — and the
measured values *are* the ceiling. "Percentage of retrieved chunks that contain
the answer" is a measurement of k.

**FINDING: at k = 5 it is vacuous in the other direction too.** `top_k` equals
the chunk count, so recall is 1.0 because nothing was left out.

**FINDING: string containment cannot localise.** `"30 days"` occurs in chunks 0
and 4 — the refund window and the service-credit deadline — so a question about
either is scored correct by the other.

**ANSWER: the honest number is recall at 1: 10 of 10.** Nothing is wrong at
retrieval; whatever is wrong is `simple_generate`'s.

### 5 — the follow-up fails on a question mark

```text
"What about enterprise?"   ->  scores [0.0, 0.0, 0.0]   ranking [0, 1, 2] = insertion order
"What about enterprise"    ->  scores [0.099, 0.093, 0.045]   ranking [1, 2, 3]
```

**ANSWER: history cannot reach retrieval.** `RAGPipeline.query` embeds the
question and calls `search` *before* `build_rag_prompt` exists. Three turns of
history reach the prompt on 2 of 2 runs and leave retrieval identical on both.

**MECHANISM: one character.**

| token | in the vocabulary? |
|---|---|
| `enterprise` | ✅ |
| `enterprise.` | ✅ |
| `enterprise?` | ❌ |

**FINDING: alive, its only content word is the one the corpus weighs least.**
`what`, `about` and `tiers` are not in the vocabulary at all, and `enterprise`
occurs in **all five** chunks, so `compute_idf` gives it **1.000** — the floor.
The query embeds to a single non-zero dimension and the ranking is pure term
frequency.

**FINDING: the prompt is inert for a second, independent reason.**
`simple_generate` scans `retrieved_chunks`, not the prompt, so the answer is
byte-identical with and without history on both follow-ups.

**CONTROL: fix the tokeniser, not the prompt.** Stripping trailing punctuation
on both sides takes the vocabulary 270 → 258 and the follow-up *as written*,
question mark included, from 0.0 everywhere to the pricing chunk at rank 1. That
is the intervention the exercise's own test case needed, and it is nowhere near
the prompt.
