<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 22-embedding-models-deep-dive

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/) · upstream spec
`phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/docs/en.md`

```bash
uv run demo practice run 22-embedding-models-deep-dive --ex 1
uv run demo explain 22-embedding-models-deep-dive --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Encode 100 sentences with `bge-small-en-v1.5` at full dim (384), then at Matryoshka 128… | code | T0 | `ex01_truncating_a_hash_is_not_matryoshka.py` |
| 2 | Medium. Compare BGE-M3 dense, sparse, and colbert on 500 passages from your domain. Which win… | code | T0 | `ex02_three_modes_are_one_mode_and_rrf_measures_consensus.py` |
| 3 | Hard. Run MTEB on three candidate models across your top-2 domain tasks. Report MTEB score, p… | code | T1 | `ex03_pareto_optimal_names_a_set_or_the_cheapest_model.py` |
<!-- generated:end -->

## Answers

`sentence_transformers`, `FlagEmbedding`, `mteb` and `torch` are all absent, so
all three exercises run against the lesson's own hash-trick encoder over a
40-passage corpus on 10 topics. That substitution is the finding in exercise 1
and the reason the comparisons in 2 and 3 come out the way they do: every
"model" here reads the same lexical feature, and every metric is measured on a
corpus small enough that its own grid is visible.

Exercises 1 and 2 run at **T0**; exercise 3 is **T1** because it times a batch.

### 1 — Truncating a hash is not Matryoshka

40 sentences, 10 topic queries, `hash_embed` at 256 dimensions:

| width | MRR | recall@10 | vocab surviving (of 236) | empty documents |
|---:|---:|---:|---:|---:|
| **256** | **0.9500** | 0.8000 | 236 | 0 |
| 192 | 0.9000 | 0.6250 | 170 | 0 |
| **128** | **0.6663** | 0.4500 | 108 | **2** |
| 96 | 0.4228 | 0.3250 | 74 | 4 |
| 64 | 0.3755 | 0.3000 | 56 | 6 |
| 32 | 0.2515 | 0.2500 | 30 | **14** |

**ANSWER: MRR drops 0.2837, about 30% relative.**

**MECHANISM: but truncation here deletes vocabulary rather than compressing it.**
Matryoshka is a *training* property — a nested loss makes the first `k`
coordinates a usable embedding. `hash_embed` has interchangeable dimensions, so
cutting at `k` drops every token whose hash index is at or above `k`. The curve
above is a bucket-loss curve.

**FINDING: below 128 the deletion removes whole documents.** They become the zero
vector, which `truncate_matryoshka` returns un-normalised, and they score cosine
0.0 against every query.

**MECHANISM: and their order is the corpus's, read backwards.** `rank` sorts
`(score, index)` tuples in reverse, so ties break by *descending index* — at 32
dimensions a seventh of the ranking is insertion order.

**FINDING: the full-width baseline is already lossy.** 236 vocabulary types
occupy **151** of the 256 buckets; **61** buckets hold more than one token and
**38** of those hold opposite signs, which cancel in the sum. The drop is
measured from a floor.

### 2 — Three modes are one mode, and RRF measures consensus

| mode | recall@10 |
|---|---:|
| dense (`hash_embed` + `cosine`) | 0.8000 |
| sparse (`sparse_embed` + `sparse_score`) | 0.8000 |
| **ColBERT (per-token max-sim)** | **0.8250** |
| RRF, dense + sparse | 0.8000 |
| RRF, all three | 0.8000 |
| RRF over `main()`'s top-5 lists | **0.7500** |

**ANSWER: ColBERT wins by 0.0250, and RRF does not beat it.**

**MECHANISM: the win is one document.** 40 passages with 4 relevant per query
means recall@10 moves in steps of 0.025. The comparison has one unit of
resolution and the answer sits inside it.

**FINDING: the three modes read one feature.** `hash_embed` of a single token is
a unit vector on one coordinate, so ColBERT's max-similarity reduces to *does the
passage hold a token in the same bucket*. It agrees with sparse on the sign of
the score for **353 of 400** query-passage pairs.

**MECHANISM: RRF cannot repair that, because it discards the scores.** At `k=60`
over 40 documents the weights run 0.01639 down to 0.01000 — a **1.64x** range end
to end — so a document ranked first by one mode and last by the other scores
0.02639 and *loses* to one ranked fifth by both at 0.03077. RRF is a consensus
vote.

**FINDING: fusing the truncated lists `main()` passes is worse than not fusing.**
A document missing from both heads gets no score at all.

**CONTROL: `rrf_fuse` is not composable with itself.** It consumes
`(score, index)` and returns `(index, score)`, so cascading fusions reads each
score as a document id — silently.

### 3 — Pareto-optimal names a set, or the cheapest model

Three candidates, two domain tasks, latency and cost measured on this machine:

| candidate | MRR | recall@10 | p99 (µs) | $/1M queries |
|---|---:|---:|---:|---:|
| 256-dim | 0.9500 | 0.8000 | ~271 | ~0.070 |
| 128-dim | 0.9200 | 0.7750 | ~150 | ~0.038 |
| **64-dim** | **0.9500** | 0.7000 | **~85** | **~0.021** |

**ANSWER: under recall@10 every candidate is Pareto-optimal.** Quality falls
monotonically with width and so do latency and cost, so nothing dominates
anything and the frontier is all three.

**FINDING: under MRR the frontier collapses to the cheapest model.** 64
dimensions ties 256 at 0.9500 at a third of the latency, so it dominates both
others. Same models, same run, opposite conclusion.

**MECHANISM: the two tasks order the candidates differently** — `256 = 64 > 128`
by MRR, `256 > 128 > 64` by recall — because MRR reads only the first relevant
hit and recall@10 reads all four. A mean over "your top-2 tasks" is decided by
which two.

**MECHANISM: two of the three axes are the same axis.** Dollars per million
queries is seconds per query times a price — a positive scalar multiple, constant
across all three candidates — so it reproduces the latency ordering exactly and
can never change the Pareto set.

**FINDING: p99 on a 100-query batch is a single order statistic.** It is the 99th
of 100 sorted samples, the second-largest in the batch, and it moves ~9–11%
across repeats. That is well inside the 200%+ gap between candidates here, so the
axis separates them because this workload has no tail — not because one
observation is a percentile.

**CONTROL: the quality axis is real.** Recall@10 falls strictly with width, so
the frontier under recall is a genuine trade. It is MRR, the metric that
saturates, that makes the cheapest model look free.
