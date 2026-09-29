<!-- generated:start -->
# 19-capstone-projects / 66-reranker-cross-encoder

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/66-reranker-cross-encoder/) · upstream spec
`phases/19-capstone-projects/66-reranker-cross-encoder/docs/en.md`

```bash
uv run demo practice run 66-reranker-cross-encoder --ex 1
uv run demo explain 66-reranker-cross-encoder --ex 1
uv run pytest demos/phases/19-capstone-projects/66-reranker-cross-encoder
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Sweep N from 5 to 50 and plot recall@1 of the reranked output. Find the knee on this fixture. | code | T1 | `ex01_there_is_no_knee_reranking_lowers_held_out_recall_at_1_from_4_of_8_to_3_of_8.py` |
| 2 | Train the cross-encoder for ten epochs instead of one. Measure the score-margin between posit… | code | T1 | `ex02_after_ten_epochs_the_margin_is_0_113_and_held_out_queries_still_score_below_zero.py` |
| 3 | Replace mean-pooling with a CLS-token head. Compare convergence on this fixture. | code | T1 | `ex03_the_cls_head_needs_50_epochs_to_the_mean_pools_33_but_38_8_to_35_over_five_seeds.py` |
| 4 | Add a second cross-encoder head that predicts a binary "is this answer in the document" label… | code | T1 | `ex04_the_answer_head_keeps_the_gold_doc_for_2_of_8_new_queries_and_answers_all_4_off_topic_ones.py` |
| 5 | Replace the deterministic mock bi-encoder with the one from lesson 65 and chain the two stage… | code | T1 | `ex05_lesson_65s_hybrid_gets_7_of_8_held_out_queries_right_and_the_reranker_drops_it_to_3.py` |
<!-- generated:end -->
## Answers

Every exercise imports and runs the lesson's `code/main.py`: an 8-document
corpus, a hashed-bag-of-words mock `BiEncoder`, and a one-block `CrossEncoder`
that `train_tiny` fits to 14 hand-labelled triples. Exercise 5 also loads
lesson 65's `code/main.py`. The lesson labels only its 5 training queries,
so every exercise also scores 8 held-out paraphrases, one per document
(`HELD_OUT`, written for these solutions). Everything runs on CPU from the
lesson's own seed. All five need torch, so they are T1.

### 1 — there is no knee; reranking lowers held-out recall@1 from 4/8 to 3/8

**The curve falls with N and goes flat at N = 8, so there is no knee.** The
best N is 1, which means not reranking at all.

| queries | N=5 | 10 | 20 | 30 | 40 | 50 | bi-encoder alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 training | 5 | 3 | 3 | 3 | 3 | 3 | 4 |
| 8 held-out | 2 | 3 | 3 | 3 | 3 | 3 | 4 |

The corpus has 8 documents, so N = 10 to 50 all rerank the same full corpus,
and five of the six sweep points are one run. At N = 8 the model even gets 2
of its own 5 training queries wrong. It ranks d6 above d3 and d8 above d4, and
d6 and d8 are the only documents the training data never labels below 1.0.
All 3 of `main()`'s demo queries are training queries, which is the leak the
doc itself warns about. Even so, the authorization query reranks d8 to top-1.

### 2 — after ten epochs the margin is 0.113, and held-out queries still score below zero

**Over ten epochs the mean positive-minus-negative margin grows from 0.011
to 0.113, but positives and negatives are still not separated.**

| epoch | 0 | 1 | 2 | 5 | 10 | 60 |
|---|---:|---:|---:|---:|---:|---:|
| mean margin | .031 | .011 | .039 | .068 | .113 | 1.002 |
| worst pos-neg gap | -.164 | -.192 | -.111 | -.010 | -.015 | .971 |
| held-out margin | -.000 | -.011 | -.005 | -.002 | -.010 | .153 |

The worst gap first turns positive at epoch 15. The doc calls `train_tiny`
"one pass of supervised training", but the function defaults to 60 epochs
and `main()` passes 60. After a true single pass the model's margin is
smaller than it was untrained. For all ten epochs the model does not prefer
the gold document on unseen queries.

### 3 — the CLS head needs 50 epochs to the mean-pool's 33, but 38.8 to 35.0 over five seeds

**On the lesson's seed the CLS head converges more slowly; averaged over five
inits the gap is small.** A hook copies the CLS vector over every position, so
the lesson's own pooling reads CLS alone (exact to 1.8e-7). Here is the epoch
where the loss first drops below 0.01:

| init | SEED | +1 | +2 | +3 | +4 | mean |
|---|---:|---:|---:|---:|---:|---:|
| mean-pool | 33 | 32 | 31 | 44 | 35 | 35.0 |
| CLS | 50 | 36 | 31 | 34 | 43 | 38.8 |

The seed the lesson ships shows the biggest gap of the five. Neither head
ranks held-out queries well: 8/40 for mean-pool and 6/40 for CLS, against
20/40 for the bi-encoder alone.

### 4 — the answer head keeps the gold doc for 2 of 8 new queries and answers all 4 off-topic ones

**The rank + threshold pair works only on training queries.** The answer head
is a second linear layer on the pooled vector, trained jointly with BCE on
"label = 1.0".

| queries | gold kept | gold at rank 1 | wrong docs kept | empty |
|---|---:|---:|---:|---:|
| 5 training | 5/5 | 2/5 | 6/35 | 0 |
| 8 held-out | 2/8 | 2/8 | 19/56 | 1 |
| 4 off-topic | -- | -- | 9/32 | 0/4 |

"banana bread recipe" is answered with d3 (retry budgets). The doc's other
suggestion, to treat a low rank-1 score as out of domain, fails as well.
Off-topic top-1 scores are 0.679 to 1.270, and held-out scores are 0.695 to
1.167. No threshold rejects all four off-topic queries without also
rejecting all eight real ones.

### 5 — lesson 65's hybrid gets 7 of 8 held-out queries right, and the reranker drops it to 3

**Chaining makes the better retriever worse.** With K = 3 on held-out queries:

| first stage | N | r@1 alone | r@1 chained | r@3 alone | r@3 chained |
|---|---:|---:|---:|---:|---:|
| hybrid (lesson 65) | 5 | 7 | 3 | 7 | 6 |
| hybrid (lesson 65) | 8 | 7 | 3 | 7 | 3 |
| mock (lesson 66) | 5 | 4 | 2 | 6 | 4 |
| mock (lesson 66) | 8 | 4 | 3 | 6 | 3 |

Top-1 moves on 6 of the 8 queries. At N = 8 every pool is the whole corpus,
so the reranked top-3 is identical, query by query, whichever retriever
feeds it. The first stage matters only when N is smaller than the corpus.
