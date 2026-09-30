<!-- generated:start -->
# 19-capstone-projects / 67-query-rewriting-hyde

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/67-query-rewriting-hyde/) · upstream spec
`phases/19-capstone-projects/67-query-rewriting-hyde/docs/en.md`

```bash
uv run demo practice run 67-query-rewriting-hyde --ex 1
uv run demo explain 67-query-rewriting-hyde --ex 1
uv run pytest demos/phases/19-capstone-projects/67-query-rewriting-hyde
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement RAG-Fusion (a 2024 variant of multi-query) where the rewriter's paraphrases are int… | code | T1 | `ex01_the_lesson_66_reranker_puts_cancelling_jobs_first_on_all_3_queries_and_drops_gold_from_rank_1_to_6.py` |
| 2 | Add a fourth strategy: step-back prompting (ask the LLM for the more general question, retrie… | code | T0 | `ex02_step_back_takes_first_place_on_1_of_3_queries_and_decomposition_never_wins_its_own_query.py` |
| 3 | Train the decomposer to recognize atomic queries by adding a "is the question atomic" head. M… | code | T0 | `ex03_an_atomic_head_cuts_the_over_split_rate_from_16_of_24_to_1_of_24.py` |
| 4 | Replace the mock LLM with a real model call. Measure the latency-per-strategy on your stack. | code | T1 | `ex04_the_model_call_is_over_99pct_of_every_strategys_latency_and_the_multi_query_prompt_omits_the_question.py` |
| 5 | Add a confidence score per rewrite. Drop rewrites below the threshold. Measure the impact on… | code | T0 | `ex05_a_similarity_confidence_gate_only_lowers_recall_because_rewrites_that_find_gold_score_lower.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`. That file holds a hybrid
BM25 + hashed-embedding retriever over 8 documents, a `MockLLM` with lookup
tables, and three `GOLD` queries: "transfer breaks halfway", "merge two
retrievers" and the multi-topic "upload fails and the retry budget is
exhausted". Exercise 1 also loads lesson 66's cross-encoder, and exercise 4
loads lesson 35's GPT. Papers were read on 2026-09-29:
[RAG-Fusion, arXiv:2402.03367](https://arxiv.org/abs/2402.03367) and
[Take a Step Back, arXiv:2310.06117](https://arxiv.org/html/2310.06117v2).

### 1 — lesson 66's reranker puts "Cancelling jobs" first on all 3 queries and drops gold from rank 1 to 6

**RAG-Fusion is diverse rewrites, then RRF, then a rerank. The diversity
works; the lesson 66 rerank makes the list worse.** Rewrites come from the
mock's paraphrases and decomposition. One is kept only if its token Jaccard
with the query and with every kept rewrite is below 0.5. That keeps 3 / 3 / 2
rewrites. On the multi-topic query it drops all three of the lesson's
paraphrases, which have pairwise Jaccard 0.60 to 1.00, and keeps the two
sub-questions instead.

| gold rank (transfer / merge / multi-topic) | |
|---|---|
| lesson multi-query | 1 / 2 / 1 |
| RAG-Fusion, RRF only | 1 / 2 / 1 |
| + lesson 66 rerank by the query | 6 / 2 / 6 |
| + rerank by best score over all queries | 6 / 1 / 4 |

The published RAG-Fusion uses RRF itself as the rerank step. That is the
row that holds up here. Lesson 66's cross-encoder memorised its 14 training
triples, down to their wording. Lesson 67 rewords the same eight documents,
and on that wording the reranker ranks d8 "Cancelling jobs" first for all
three fixture queries, and even for its own training query. The
lesson's paraphrase fallback also shows the "rewrites all converge" failure
in its own demo. The third paraphrase of the multi-topic query repeats the
first, and both read "an transfer".

### 2 — step-back takes first place on 1 of 3 queries, and decomposition never wins its own query

**Step-back ties HyDE on "merge two retrievers" and loses ground on the
multi-topic query.** Following the paper, the step-back question and the
original query are both retrieved and merged by RRF.

| strategy | transfer | merge | multi-topic |
|---|---:|---:|---:|
| no-rewrite | 3 | 5 | 1 |
| HyDE | 2 | 1 | 2 |
| multi-query | 1 | 2 | 1 |
| decompose | 3 | 5 | 1 |
| step-back + original | 3 | 1 | 2 |
| step-back alone | 3 | 1 | 3 |

The table does not back the lesson's claim that "decomposition wins on the
multi-topic query". No-rewrite and multi-query rank d1 first there too. The
query's second fact, d3, is at rank 2 for both no-rewrite and decompose.
HyDE and multi-query each take first place alone once; decompose never does.

### 3 — an atomic head cuts the over-split rate from 16 of 24 to 1 of 24

**Over-split rate: 66.7% before, 4.2% after.** The fixture has 36 labelled
questions: 24 atomic (16 containing "and") and 12 multi-topic. Off its
table, the lesson's decomposer splits on every " and ". The head is a
logistic regression over four features of the split. Does the right half
start with a question word? Does it have its own verb? Do the two halves
retrieve different top-1 documents? How long is the shorter half?

| | atomic split | multi-topic split |
|---|---:|---:|
| lesson decomposer | 16 / 24 | 12 / 12 |
| with the head (2-fold CV) | 1 / 24 | 12 / 12 |

The one miss is "how much memory do vectors and indexes need". The lesson's
warning about over-splitting holds. Unsplit, all 16 atomic "and" questions
put gold first. Split, the mean rank is 1.44: 3 of them lose first place,
two fall to rank 4, and the head's miss is one of those two.

### 4 — the model call is over 99% of every strategy's latency, and the multi-query prompt omits the question

**The model call is the whole cost, and decomposition is the cheapest
strategy.** The "real model" is lesson 35's `GPTModel` (4 layers, d_model
128), run on CPU with real forward passes. Its prompts are the lesson's own
templates, and its output lengths match the mock's answers. Totals over the
three queries (the milliseconds come from one local run and are not
asserted):

| | HyDE | multi-query | decompose |
|---|---:|---:|---:|
| tokens generated | 78 | 88 | 33 |
| positions processed (no KV cache) | 4,899 | 3,408 | 1,676 |
| model ms | ~113 | ~100 | ~41 |
| retrieval ms | 0.32 | 0.52 | 0.18 |
| hybrid retrievals per query | 2 | 4 | 1 / 1 / 2 |

The retrieval counts differ from the lesson's accounting. HyDE retrieves
twice, because the original query is searched again beside `search_vec`.
Multi-query retrieves N + 1 times. All of these run in a sequential loop,
though the lesson says they "run in parallel". The lesson's multi-query
template has only `{N}` as a placeholder, so a real model would never see
the question it is asked to rewrite.

### 5 — a similarity confidence gate only lowers recall, because the rewrites that find gold score lower

**No threshold raises recall.** Confidence here is the cosine between the
rewrite's embedding and the query's. Recall@3 is measured over 4 relevant
pairs: each query's gold document, plus d3 for the multi-topic query.

| threshold | 0 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| multi-query | 1.00 | 1.00 | 1.00 | 0.75 | 0.75 |
| decompose | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |

The score runs the wrong way. The five paraphrases that on their own put
gold first average 0.501, and the four that miss average 0.658. The
duplicated "an transfer" paraphrase scores 0.907 and misses. A rewrite earns
its place by leaving the query's vocabulary, which is exactly what a
similarity-to-query score penalises.
