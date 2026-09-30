<!-- generated:start -->
# 19-capstone-projects / 65-hybrid-retrieval-bm25-dense

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/65-hybrid-retrieval-bm25-dense/) · upstream spec
`phases/19-capstone-projects/65-hybrid-retrieval-bm25-dense/docs/en.md`

```bash
uv run demo practice run 65-hybrid-retrieval-bm25-dense --ex 1
uv run demo explain 65-hybrid-retrieval-bm25-dense --ex 1
uv run pytest demos/phases/19-capstone-projects/65-hybrid-retrieval-bm25-dense
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace `mock_embed` with a real model from your provider. Re-run the demo and report how the… | code | T1 | `ex01_a_real_encoder_moves_the_upload_doc_from_dense_rank_2_to_1_but_bm25_already_had_it_first.py` |
| 2 | Add a third modality: chunk summaries indexed separately and fused as a third ranked list. Me… | code | T0 | `ex02_a_summary_list_lowers_mrr_from_0_850_to_0_820_and_the_lessons_fusion_already_trails_bm25_alone.py` |
| 3 | Sweep RRF k across 10, 30, 60, 100, 200. Plot the recall@k curve from lesson 68. Report the v… | code | T0 | `ex03_the_recall_curve_is_flat_from_k_9_to_1000_so_it_peaks_at_k_2_below_the_whole_sweep.py` |
| 4 | Implement BM25F properly (per-field length normalization rather than the multiplier trick) an… | code | T0 | `ex04_bm25f_lifts_symbol_lookups_from_375_to_383_of_400_and_the_multiplier_trick_misses_long_bodies.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a from-scratch BM25 with
title tokens repeated 3x, a 96-dimension hashed "embedding", and
reciprocal rank fusion at k = 60. Exercise 3 also runs lesson 68's
`evaluate_pipeline` on its corpus and qrels. External sources were read on
2026-09-29: the TREC-13 Microsoft Cambridge paper (BM25F, section 4.1) and
Wikipedia's Okapi BM25 page.

### 1 — a real encoder moves the upload doc from dense rank 2 to 1, but BM25 already had it first

**On "how do we handle cancelled uploads" the dense-only ranking goes from
d3, d2, d6, d1, d4 with the mock to d2, d1, d3, d5, d4 with
`paraphrase-multilingual-MiniLM-L12-v2`.** The model runs offline from the
Hugging Face cache through a from-scratch BERT forward and Unigram tokenizer.
While authoring, this was checked against sentence-transformers: identical
tokens, unit-normalised embeddings within 1.4e-7. The upload doc d2 goes from 2nd to 1st
(cosine 0.6247, against 0.4517 for the runner-up). The abort function d1
goes from 4th to 2nd. The fused top-2 stays d2, d3. Under the mock those two
tied exactly at 1/61 + 1/62, and d2 won only on insertion order.

**The query is not really paraphrased.** It shares `cancelled`, `do` and
`uploads` with d2's body, so BM25 ranks d2 first (4.85 against 1.69) before
any embedding is involved. The Problem section says the right answer is
"the abort function whose summary mentions cancellation", but d1 contains no
`cancel*` token.

| query | doc's claimed (BM25, dense, RRF) rank | measured |
|---|---|---|
| literal `AbortMultipartOnFail` | 1, 4, 1 | 1, 1, 1 |
| "cancelled uploads" | 6, 1, 1 | 1, 2, 1 |
| "centralized authorization" | 3, 3, 1 | 1, 1, 1 |

A rank of 6 cannot occur at the demo's k_each = 5, and BM25 returns only 2
hits for the paraphrase.

### 2 — a summary list lowers MRR from 0.850 to 0.820, and the lesson's fusion already trails BM25 alone

**The gain is negative.** Each summary is the body's first sentence (no LLM
is available). The summaries get their own index and are fused as a third
RRF list. The test set is 14 labelled queries over the lesson's 7 docs: each
doc once by name and once paraphrased.

| system | MRR | hits@1 | paraphrase MRR |
|---|---:|---:|---:|
| BM25 alone | 0.8673 | 12 | 0.7347 |
| dense alone | 0.8060 | 10 | 0.6119 |
| BM25 + dense (the lesson) | 0.8495 | 11 | 0.6990 |
| + summaries (dense index) | 0.8197 | 10 | 0.6395 |
| + summaries (BM25 index) | 0.8245 | 11 | 0.6490 |

Literal queries score 1.0 in every fused system, so the whole loss is on
paraphrases. A first sentence is a subset of text the other lists already
index, which makes the third list a weaker, correlated vote. **The lesson's
own two-list fusion already scores below BM25 alone** (0.8495 against
0.8673), against the doc's claim that the vote "wins on every query class".

### 3 — the recall curve is flat from k = 9 to 1000, so it peaks at k ≤ 2, below the whole sweep

**All five requested values give the same point: recall@1 0.375, recall@3
0.75, recall@5 1.0, MRR 0.675.** This is lesson 65's retriever on lesson
68's 12 docs and 4 qrels. No fused order changes for any k from 9 to 1000.
The curve peaks at k = 1 and 2 (recall@1 0.625, recall@3 1.0), for one
reason: in the fusion query, gold d6 sits at (BM25 1, dense 4) and d12 at
(2, 2). RRF prefers d6 only while 1/(k+1) + 1/(k+4) > 2/(k+2), which holds
for k < 2. At k = 2 they tie exactly. With two lists, a large k just ranks
by rank sum.

**BM25 alone beats the fusion at every k** (recall@1 0.625, MRR 0.875;
dense alone 0.125 and 0.4375). **Lesson 68 grades a stand-in, not this
retriever.** Its `hybrid_pipeline`, labelled a "stand-in for the lesson 65
retriever", is a synonym-expanded bag of words that scores recall@1 0.875
and MRR 1.0. The real retriever scores 0.375 and 0.675.

### 4 — BM25F lifts symbol lookups from 375 to 383 of 400, and the multiplier trick misses docs with long bodies

**On a generated code index, per-field BM25F puts the defining doc first for
383 of 400 symbol queries, and the lesson's multiplier trick for 375.** The
index is 10 seeds x 40 docs, each titled with a camelCase symbol, with a
body that calls other symbols. MRR is 0.9788 against 0.9658. BM25F does
better on 18 queries and worse on 8. Both use the lesson's weights (title 3,
body 1, b = 0.75). BM25F follows TREC-13 section 4.1: normalise each field
by its own length, sum with weights, saturate once. With a body field alone
it reproduces the lesson's BM25 to 8.9e-16.

**The multiplier trick charges a title match for the body's length.** Its 25
misses have bodies averaging 130.2 words, against 83.3 for the corpus.
BM25F's 17 misses average 82.1. The multiplier trick is Robertson, Zaragoza
and Taylor's CIKM 2004 scheme. It keeps BM25's formula, as the lesson says,
but it does not normalise per field.
