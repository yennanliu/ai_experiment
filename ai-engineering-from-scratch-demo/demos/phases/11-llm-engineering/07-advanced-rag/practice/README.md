<!-- generated:start -->
# 11-llm-engineering / 07-advanced-rag

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/07-advanced-rag/) · upstream spec
`phases/11-llm-engineering/07-advanced-rag/docs/en.md`

```bash
uv run demo practice run 07-advanced-rag --ex 1
uv run demo explain 07-advanced-rag --ex 1
uv run pytest demos/phases/11-llm-engineering/07-advanced-rag
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compare BM25 vs vector search vs hybrid search on the sample documents. For each of the 5 tes… | code | T0 | `ex01_all_three_retrievers_return_the_same_first_chunk.py` |
| 2 | Implement a metadata filter. Add a "category" field to each document (security, billing, api,… | code | T0 | `ex02_the_four_categories_do_not_cover_the_corpus.py` |
| 3 | Build a full HyDE pipeline using the simple generate function from Lesson 06. Compare retriev… | code | T0 | `ex03_the_hypothesis_adds_five_stopwords_and_loses_a_hit.py` |
| 4 | Implement the parent-child chunking strategy on the sample documents. Use child_size=30 and p… | code | T0 | `ex04_the_baseline_chunk_size_the_exercise_names_hangs.py` |
| 5 | Create an evaluation dataset: 10 questions with known answer chunks. Measure Recall@3, Recall… | code | T0 | `ex05_twelve_points_one_value_and_only_reranking_ever_moves.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib, so all five exercises are **T0** and run in CI. Six
documents, one chunk each at the default chunk size, a retrieval pool of 15 and
a fusion constant of 60: the sophistication in this lesson has nowhere to act.
Vector, BM25 and hybrid return the identical first chunk on every query, and
they keep doing so until the corpus is five times its shipped size.

### 1 — all three retrievers return the same first chunk

| query | gold | vector | BM25 | hybrid |
|---|---:|---:|---:|---:|
| what encryption is used at rest | 2 | 2 | 2 | 2 |
| what is the enterprise refund window | 0 | 0 | 0 | 0 |
| what is the starter rate limit | 3 | 3 | 3 | 3 |
| what uptime is guaranteed | 5 | 5 | 5 | 5 |
| how much does professional cost | **1** | **2** | **2** | **2** |

**ANSWER: hybrid wins 0 of 5.** The exercise asks for at least 3.

**MECHANISM: the pool is larger than the corpus.** `retrieval_pool=15` against a
6-chunk index means both arms return every chunk and the fusion is a rank-sum
over identical candidate sets.

**MECHANISM: `reciprocal_rank_fusion` discards the scores.**

```python
reciprocal_rank_fusion([[(0, 99000.0), (1, 0.0)]])
reciprocal_rank_fusion([[(0,     0.001), (1, 0.0)]])   # identical output
```

And with `k=60` the fused spread across the whole six-chunk ranking is
**0.00248** — 7% of the top score.

**FINDING: the query all three get wrong has one content word.** "how much does
professional cost" contributes `does` and `professional` and nothing else; the
corpus never writes "cost" or "much".

**CONTROL: 31 chunks before the arms disagree at all.**

| re-chunked without overlap | chunks | queries where the arms differ |
|---|---:|---:|
| 50 words | 13 | 0 |
| 30 words | 23 | 0 |
| 20 words | **31** | **1** |

Only past the 15-item pool — five times the shipped corpus — is there a
comparison to record.

### 2 — the four categories do not cover the corpus

| document | category |
|---|---|
| 0 Refund Policy | billing |
| 1 Product Overview | product |
| 2 Security Practices | security |
| 3 API Documentation | api |
| 4 **Q3 Earnings Report** | **none of the four** |
| 5 **Uptime and Reliability** | **none of the four** |

**ANSWER: the filter does exactly what is asked, and the test case it names is
unmoved.** "What encryption is used?" searches **1** chunk instead of 6 and
returns the same chunk in the same position. Four of five queries unchanged.

**FINDING: after filtering, the category *is* the result.** Each named category
holds exactly 1 chunk, so a top-3 over a filtered set returns the whole category
and the ranking inside it is moot.

**MECHANISM: where the filter is applied decides whether the scores move.**

```text
filter the embedding list   ->  IDF computed over 6 chunks, 6 distinct values, untouched
re-index the filtered set   ->  IDF over 1 document, collapses to 1 value, score meaningless
```

The exercise says "before running vector search", which is the first of the two.

**CONTROL: one query of six.** "what encryption does the enterprise plan
include" goes to the **product** document unfiltered — `enterprise` appears in
all six chunks and is the query's only in-vocabulary content word. Filtered to
security it returns the security document.

### 3 — the hypothesis adds five stopwords and loses a hit

**ANSWER: direct 5 of 5, HyDE 4 of 5.** HyDE loses the encryption query and
gains nowhere.

```text
query        (6 words)  what encryption is used at rest
hypothesis  (30 words)  The answer to 'what encryption is used at rest' is as
                        follows: Based on our documentation, encryption used rest
                        involves specific policies and procedures that define the
                        process and requirements.
```

**MECHANISM: the in-vocabulary words the hypothesis adds are**
`['and', 'on', 'that', 'the', 'to']`. Every content word it invents —
documentation, policies, procedures, requirements — is absent from the corpus,
so it retrieves nothing, and the five stopwords re-weight the ranking.

**FINDING: the query it breaks had no content words to begin with.** "what
encryption is used at rest" contributes `['at', 'is', 'rest']`: the corpus writes
**encrypted**, not *encryption*, and **using**, not *used*. Direct search finds
the security document on the strength of `rest` alone.

**FINDING: the topic extraction drops words it should keep.** The filler list
includes `is`, `at` and `the`, so the topic is `'encryption used rest'` — which
then appears alongside the quoted query, doubling three terms.

**CONTROL: query + topic, no template prose** → 5 of 5, matching direct search.
HyDE needs a generator that knows the corpus vocabulary; a template cannot
supply one.

### 4 — the baseline chunk size the exercise names hangs

```python
chunk_text(doc, 50)          # default overlap=50  ->  step 0  ->  never returns
chunk_text(doc, 50, 0)       # the only overlap that keeps 50-word chunks 50 words
```

| | recall@3 | prompt words (5 queries) |
|---|---:|---:|
| parent-child (100 / 30) | **5 / 5** | **931** |
| standard (50 / 0) | **5 / 5** | 693 |

**ANSWER: a tie on recall, at 1.34× the prompt.**

**FINDING: the generated answers differ on 2 of the 5** — `simple_generate`
scores every sentence it is given, and a 100-word parent offers more of them.

**FINDING: 5 of the 24 children are under 10 words.** Children tile with
`child_start += child_size` and stop at the parent boundary, so a 91-word
document under one 100-word parent yields children of **30, 30, 30, 1**. Those
fragments are embedded and searchable like any other child.

**MECHANISM: the child index and the parent index are different corpora** — 24
against 7 — so the score attached to a returned parent was never computed for
that text. That is the point of the strategy, and it is worth saying out loud.

### 5 — twelve points, one value, and only reranking ever moves

At the shipped chunk size (6 chunks):

| | @3 | @5 | @10 |
|---|---:|---:|---:|
| vector | 1.000 | 1.000 | 1.000 |
| BM25 | 1.000 | 1.000 | 1.000 |
| hybrid | 1.000 | 1.000 | 1.000 |
| hybrid + rerank | 1.000 | 1.000 | 1.000 |

Re-chunked to 31:

| | @3 | @5 | @10 |
|---|---:|---:|---:|
| vector | 0.778 | 0.889 | 1.000 |
| BM25 | 0.778 | 0.889 | 1.000 |
| hybrid | 0.778 | 0.889 | 1.000 |
| **hybrid + rerank** | **1.000** | **1.000** | 1.000 |

**ANSWER: reranking helps most at k=3, and it is the only thing that helps** —
+0.222, +0.111, 0.000. It can only reorder within the pool, so its value is
exactly the recall the smaller k was throwing away. And three of the four
requested curves are one line.

**MECHANISM: `rerank`'s `initial_score * 5.0` weighs its input 66× differently
by retriever.**

```text
hybrid candidates (RRF)     0.133 .. 0.164     range 0.031   vs whole-number overlaps
vector candidates (cosine)  0.000 .. 2.043     range 2.043
```

The same function treats its input ranking as a constant in one case and a real
signal in the other.

**FINDING: the labels move with the chunking.** At 20-word chunks
`"$99 per month"` is split across a boundary and one question becomes
unlabellable — the pair count goes 10 → 9. A derived-label evaluation set is a
function of the chunk size it was derived at.
