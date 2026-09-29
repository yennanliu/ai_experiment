<!-- generated:start -->
# 19-capstone-projects / 68-rag-eval-precision-recall

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/68-rag-eval-precision-recall/) · upstream spec
`phases/19-capstone-projects/68-rag-eval-precision-recall/docs/en.md`

```bash
uv run demo practice run 68-rag-eval-precision-recall --ex 1
uv run demo explain 68-rag-eval-precision-recall --ex 1
uv run pytest demos/phases/19-capstone-projects/68-rag-eval-precision-recall
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a fifth retrieval metric: hit-rate@k. Compare it against recall@k. Explain when they differ. | code | T0 | `ex01_hit_rate_parts_from_recall_only_on_the_one_two_gold_query_and_the_reranker_returns_hybrids_lists_on_4_of_4.py` |
| 2 | Implement a graded faithfulness: 0 (unsupported), 1 (partially supported), 2 (fully supported… | code | T0 | `ex02_binary_faithfulness_passes_7_of_8_one_fact_edits_and_every_fixture_claim_is_verbatim_context.py` |
| 3 | Replace the mock judge with a real model call. Measure the disagreement between the mock and… | code | T1 | `ex03_a_real_cross_encoder_judge_disagrees_on_1_of_45_calls_and_accepts_the_same_7_of_8_wrong_facts.py` |
| 4 | Add a query-class slice ("literal", "paraphrased", "multi-topic"). Report per-slice metrics. | code | T0 | `ex04_the_lessons_4_qrels_are_all_paraphrased_and_hybrids_whole_gain_lives_in_that_slice.py` |
| 5 | Add an "answer length" metric and correlate it with faithfulness. Plot the curve. | code | T0 | `ex05_faithfulness_is_constant_on_the_fixture_so_r_is_undefined_and_a_verbosity_sweep_gives_minus_0_872.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: 12 short docs, 4 qrels, three
retrieval pipelines (a bag-of-words baseline, a synonym "hybrid", and hybrid
plus a title-boost "rerank"), and a `MockJudge` that counts shared content
words. The lesson's "generator" pastes the bodies of the top docs, and that
shapes almost every answer below.

### 1 — hit-rate@k and recall@k differ only on the one query with two gold docs, and the reranker changes no list

**They differ only at k = 1, and only on q1.**

| pipeline | hit-rate@1 | recall@1 | k = 3 and 5 |
|---|---:|---:|---:|
| baseline | 0.5 | 0.375 | both 1.0 |
| hybrid | 1.0 | 0.875 | both 1.0 |
| hybrid+rerank | 1.0 | 0.875 | both 1.0 |

q1 has two gold docs, and its top 1 holds one of them: that is a hit (1) but
half the recall (0.5). The other three qrels have one gold doc each, so the two
metrics give the same number. A check over every case (every ranking of 5
docs, every gold set of 1 to 3 of them, k from 1 to 5, 15,000 cases) confirms
the rule. Hit-rate is never below recall. It equals recall whenever there is
one gold doc, and is above it exactly when the top k holds some, but not all,
of two or more gold docs (6,000 of the 15,000 cases). Hit-rate asks whether
the generator saw anything useful; recall asks whether it saw everything.

**The rerank row cannot beat hybrid on MRR, though the doc says it does.**
`hybrid_plus_rerank_pipeline` returns hybrid's exact list on all 4 queries,
so both have MRR 1.0. "Hybrid beats baseline on recall" holds only at k = 1;
at k = 3 and 5 every row is 1.0. At k = 5 the pipelines return only 3, 2, 2
and 2 docs.

### 2 — binary faithfulness passes 7 of 8 one-fact edits, and every fixture claim is verbatim context

**Graded faithfulness is the mean of a 0/1/2 grade per claim, divided by 2.**
The grade uses the mock judge's own overlap signal: 2 means every content word
of the claim is in the context, 1 means at least the 0.4 threshold, 0 means
less. Replace one of q1's four claims with "... at five failed parts" and
binary faithfulness stays at 1.0 while graded falls to 0.875.

| probe | binary (`MockJudge`) | grade |
|---|---|---|
| 8 claims with one fact changed | 7 of 8 supported | 1, 1, 1, 1, 0, 1, 1, 1 |
| "The upload threshold is configured per stale records." (false, all context words) | supported | 2 |
| the 33 claims the three pipelines produce | 33 of 33 | all 2 |

On the lesson's fixture, the update changes nothing. All 33 claims are
verbatim substrings of the retrieved context, so both metrics are 1.0 on all
12 answers. The top grade is still a bag-of-words verdict: rearranging context
words into a false sentence earns a 2.

### 3 — a real cross-encoder judge disagrees with the mock on 1 of 45 calls and accepts the same 7 of 8 wrong facts

The real judge is `cross-encoder/ms-marco-MiniLM-L-6-v2` from the local
Hugging Face cache. It runs through a from-scratch BERT forward, which was
checked on 2026-09-29 against transformers 5.17.0 on all 53 pairs: identical
token ids, logits within 3.8e-6. It says yes when the logit is above 0. The
lesson's `evaluate_pipeline` calls it in place of `MockJudge`.

**Over the three pipelines the judges disagree on 1 of 45 calls (2.2%).** On
the baseline's answer to "how do you stop a long-running worker", the answer
is the worker-pool-sizing doc. The mock calls it relevant because they share
words; the model gives a logit of -3.799. Baseline answer relevance falls from
0.75 to 0.5. Every other number is unchanged.

| calls | agree | note |
|---|---:|---|
| 33 faithfulness | 33 | every claim is pasted context; real logits 8.2 to 10.3 |
| 12 relevance | 11 | baseline q4 split |
| 8 one-fact edits (not in the fixture) | 8 | both accept the same 7 |

The disagreement is small because faithfulness on this fixture cannot fail.
Swapping in a real relevance model does not fix that: it scores topic match,
not entailment, so it accepts "five failed parts" and `k = 10` just as the
mock does.

### 4 — the lesson's 4 qrels are all paraphrased, and hybrid's whole gain lives in that slice

Each of q1-q4 has a query word that no gold doc contains ("dropped",
"central gate", "fuse", "stop"), so all four are paraphrased. The solution
adds 4 literal queries (every word in the gold doc) and 4 multi-topic ones
(two unrelated topics, one gold doc each). Per slice, as recall@1 / MRR /
nDCG@3 / answer relevance:

| slice | baseline | hybrid | hybrid+rerank |
|---|---|---|---|
| literal | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 0.996 / 1.0 | same as hybrid |
| paraphrased | 0.375 / 0.75 / 0.724 / 0.75 | 0.875 / 1.0 / 0.882 / 1.0 | same as hybrid |
| multi-topic | 0.5 / 1.0 / 0.99 / 1.0 | same | same |

recall@3 and faithfulness are 1.0 in every cell. Pooled over 12 queries,
recall@1 is 0.625 for baseline and 0.792 for hybrid, and the paraphrased slice
accounts for the whole gap. On literal queries hybrid is slightly worse
(nDCG@3 0.996). Multi-topic recall@1 cannot exceed 0.5, because two gold docs
compete for one slot. The reranker reorders 1 list of 12 (m3) and changes no
metric in any slice.

### 5 — faithfulness is constant on the fixture, so r is undefined, and a verbosity sweep gives -0.872

**On the fixture there is nothing to correlate.** The 12 answers run 15 to 37
words, and all of them score faithfulness 1.0, so Pearson's r is undefined
(`statistics.correlation`: "at least one of the inputs is constant").

To get a curve, the solution adds a verbosity dial. The hybrid pipeline's
answer is cut off after its first m sentences, reading the retrieved docs
first and then running on into the rest of the corpus. The context stays the
retrieved top 5. This gives 72 answers and r = -0.872:

```
  14 words |########################################| 1.00
  32 words |########################################| 1.00
  44 words |################################        | 0.81
  61 words |#######################                 | 0.58
  95 words |#################                       | 0.42
 141 words |############                            | 0.31
 197 words |########                                | 0.21
```

The shape is a hyperbola, not a line. Once the retrieved sentences run out
(5 for q1, 3 for the others), the count of supported claims stays fixed, and
faithfulness is that count divided by the claim count. The mock also passes
one sentence that q1 never retrieved: d2's "... after three failed parts" has
4 of its 9 content words in q1's context. That clears the 0.4 threshold, so q1
ends at 6 supported claims, not 5.
