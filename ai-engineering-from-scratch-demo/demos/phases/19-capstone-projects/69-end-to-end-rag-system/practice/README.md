<!-- generated:start -->
# 19-capstone-projects / 69-end-to-end-rag-system

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/69-end-to-end-rag-system/) · upstream spec
`phases/19-capstone-projects/69-end-to-end-rag-system/docs/en.md`

```bash
uv run demo practice run 69-end-to-end-rag-system --ex 1
uv run demo explain 69-end-to-end-rag-system --ex 1
uv run pytest demos/phases/19-capstone-projects/69-end-to-end-rag-system
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a per-query strategy selector inside the rewriter: heuristics from lesson 67 (length, con… | code | T1 | `ex01_a_length_conjunction_and_jargon_selector_picks_13_of_14_queries_right_and_no_strategy_moves_an_eval_metric.py` |
| 2 | Add a real LLM call for the generator behind an env flag. Default to the mock. Measure the la… | code | T1 | `ex02_citing_in_parentheses_as_the_prompt_asks_drops_faithfulness_to_0_667_and_off_topic_questions_are_never_refused.py` |
| 3 | Extend the demo to take a `--corpus path` flag that loads a real corpus. Re-run the eval and… | code | T1 | `ex03_the_six_lesson_pages_fail_only_answer_relevance_at_0_625_and_a_corpus_without_qrels_is_scored_on_the_fixtures.py` |
| 4 | Add a `--strategy` flag to the chunker. Measure each strategy's contribution to end-to-end re… | code | T1 | `ex04_recall_at_5_is_1_under_all_6_chunkers_on_both_corpora_and_5_of_6_build_the_same_fixture_index.py` |
| 5 | Add a streaming generator interface and feed it into the eval. Confirm that faithfulness is c… | code | T1 | `ex05_the_streamed_eval_equals_the_batch_eval_and_faithfulness_matches_the_final_score_on_72_of_73_prefixes.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a recursive chunker, a
BM25 + hashed-embedding hybrid index fused by RRF, a rule-based rewriter, a
small cross-encoder trained on 17 triples, and a mock generator with
`[doc:chunk]` citations. It is scored over 12 fixture documents and 4 eval
queries. As shipped, the demo passes with recall@5 1.000, precision@1 0.500
(exactly at its threshold), mrr 0.708, faithfulness 1.000 and
answer_relevance 1.000. Exercise 2 also loads lesson 35's GPT and
exercise 4 loads lesson 64's chunkers. Exercises 3 and 4 use the English
pages of lessons 64 to 69 as the real corpus.

### 1 — a length, conjunction and jargon selector picks right on 13 of 14 queries, and no strategy moves an eval metric

**The selector gets 13 of 14 labelled queries right; the lesson's
`pick_strategy` gets 7.** It follows lesson 67's rule. A query is split
when a conjunction (and / or / versus) is followed by a question word or
joins two clauses of 3 or more content tokens. It goes to HyDE when 25% or
more of its content tokens are corpus identifiers. Everything else gets
multi-query. The lesson's version splits on every " and ", never on "or" or
"versus", and knows only four jargon words.

The rewriter cannot move the eval, though. Forcing HyDE, multi-query,
decomposition or no rewrite gives identical metrics. The candidate pool is
8 to 10 of the 12 chunks and always holds the gold documents, so the
reranker sees the same pool every time. `_REWRITE_HYDE` has exactly the 4
eval queries as keys, and 2 of its hypotheticals are gold documents word
for word (d3, d7). Every other query gets no hypothetical.

### 2 — citing in parentheses as the prompt asks drops faithfulness to 0.667, and off-topic questions are never refused

**`RAG_REAL_LLM=1` swaps in a real forward pass, and generation becomes the
whole cost.** The model is lesson 35's `GPTModel` (4 layers, d_model 128)
on CPU. It is fed the lesson's own prompt template: prompts of 127 to 135
tokens, answers of 16 to 30. On one local run the generate stage went from
about 0.03 ms to 100-230 ms, about 97% of each query. The check holds only
20x and over half.

| (anchor) citations, as the template asks | e1 | e2 | e3 | e4 | mean |
|---|---:|---:|---:|---:|---:|
| faithfulness | 0.667 | 0.500 | 1.000 | 0.500 | 0.667 < 0.75 |

`faithfulness_score` strips only `[...]`, so each parenthesised anchor is
left behind as a claim of its own that no context supports. The refusal
path never fires. 6 off-topic questions ("who wrote hamlet", ...) all get a
cited answer with faithfulness 1.0, and the lowest top-1 score is over 10
times `REFUSE_THRESHOLD` = 0.05. A refusal on an eval query would score
faithfulness 0.0.

### 3 — the six lesson pages fail only answer relevance at 0.625, and a corpus without qrels is scored on the fixture's

**`--corpus` over lessons 64 to 69 (62 KB, 208 chunks, 8 qrels) exits 1 on
answer_relevance.**

| | recall@5 | precision@1 | mrr | faithfulness | answer_relevance |
|---|---:|---:|---:|---:|---:|
| fixture | 1.000 | 0.500 | 0.708 | 1.000 | 1.000 |
| lesson pages | 1.000 | 0.875 | 0.917 | 1.000 | 0.625 |
| lesson pages, no qrels.json | 0.000 | 0.000 | 0.000 | 1.000 | 0.250 |

3 of the 8 answers open with a sentence that shares under 30% of the
question's words, such as "Read the demo output side by side." `run_eval`
reads the module-level `EVAL_QUERIES`. So a new corpus without its own
qrels is scored against d1-d12, and the failure message names four metrics,
not the missing qrels. `gold_answer_substring` is declared once and never
read: setting all four to "zzz" changes nothing.

### 4 — recall@5 is 1 under all 6 chunkers on both corpora, and 5 of 6 build the same fixture index

**No chunking strategy contributes to recall@5. It is 1.000 for all six on
both corpora.** Differences show only at the top:

| strategy | fixture chunks | fixture recall@1 | page chunks | page recall@1 |
|---|---:|---:|---:|---:|
| lesson69 `Chunker` | 12 | 0.375 | 208 | 0.875 |
| fixed | 12 | 0.375 | 195 | 0.875 |
| sentence | 12 | 0.375 | 124 | 1.000 |
| recursive (lesson 64) | 12 | 0.375 | 172 | 0.500 |
| semantic | 18 | 0.875 | 371 | 0.875 |
| structural | 12 | 0.375 | 97 | 1.000 |

The longest fixture document is 137 characters, so on the fixture the
chunker is a no-op. 5 of 6 strategies build the identical index, and the
demo cannot see a chunker regression. Only `semantic` differs: it separates
the distractors' "this is unrelated" sentences, which lifts precision@1
from 0.500 to 1.000. On the pages only `structural` passes all five
thresholds. The lesson's recursive `Chunker` and lesson 64's
`recursive_split` do not agree: precision@1 0.875 against 0.500.

### 5 — the streamed eval equals the batch eval, and faithfulness matches the final score on 72 of 73 prefixes

**Scoring the joined stream reproduces `run_eval` exactly, on all five
metrics and on all 4 answer strings.** The stream yields 73 word pieces.
Faithfulness hardly separates a prefix from the answer: 72 of 73 prefixes
already score the final 1.0. An eval that stops at the first piece still
gets faithfulness 0.750 and passes its threshold. answer_relevance is the
metric that notices. It differs from the final value on 22 of 73 prefixes,
and the first-piece eval scores 0.250.
