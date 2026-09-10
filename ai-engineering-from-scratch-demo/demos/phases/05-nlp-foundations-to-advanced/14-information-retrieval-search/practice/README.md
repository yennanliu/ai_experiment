<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 14-information-retrieval-search

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/14-information-retrieval-search/) · upstream spec
`phases/05-nlp-foundations-to-advanced/14-information-retrieval-search/docs/en.md`

```bash
uv run demo practice run 14-information-retrieval-search --ex 1
uv run demo explain 14-information-retrieval-search --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/14-information-retrieval-search
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement `hybrid_search` above on a 500-document corpus. Test 20 queries. Compare reca… | code | T1 | `ex01_half_the_queries_decide_nothing.py` |
| 2 | Medium. Add MRR calculation. For each test query with a known correct document, find the rank… | code | T1 | `ex02_fusion_costs_three_times_more_mrr_than_recall.py` |
| 3 | Hard. Fine-tune a dense encoder on your domain using MultipleNegativesRankingLoss (Sentence T… | code | T1 | `ex03_the_retriever_has_no_parameters_to_tune.py` |
<!-- generated:end -->

## Answers

Three exercises about hybrid retrieval where the hybrid never wins, the dense arm
is not dense, and the encoder to be fine-tuned has no parameters. All three are
**T1** (scikit-learn, for a real dense arm to compare against the lesson's toy).

The query set is the instrument. Twenty queries: ten that share content words
with their answer, ten written as plain-language paraphrases with none.

### 1 — Half the queries decide nothing

`hybrid_search` is not in the lesson. `code/main.py` ships `BM25`,
`fake_dense_rank` and `reciprocal_rank_fusion`, so the function to implement is
the composition of the last two.

**ANSWER: recall@5.**

| arm | lexical (10) | paraphrase (10) | all 20 |
|---|---:|---:|---:|
| BM25 | 1.00 | 0.30 | 0.65 |
| fake-dense | 1.00 | **0.50** | **0.75** |
| RRF hybrid | 1.00 | 0.40 | 0.70 |
| real dense (TF-IDF + SVD) | 1.00 | 0.40 | 0.70 |
| BM25 + real dense | 1.00 | 0.30 | 0.65 |

**The lexical half is a three-way tie.** Every number that separates the arms
comes from the half the exercise does not ask for.

**FINDING: the hybrid lands below its better arm.** Fusing a 0.50 ranker with a
0.30 one gives 0.40 — reciprocal rank fusion averages ranks, so a weaker arm
pulls a stronger one down.

**MECHANISM: the arm the exercise calls dense is lexical.** `fake_dense_rank` is
Jaccard overlap plus 0.15 for every pair of four-plus-character tokens where one
contains the other. On `sunlight` against `light energy` — no shared token — it
scores **0.15**. That is stemming, not semantics, and it is why the toy beats
latent semantic indexing on this corpus.

**CONTROL: no fusion beats the best single arm.** The answer to "compare
BM25-only, dense-only, and hybrid" is that hybrid is third.

### 2 — Fusion costs three times more MRR than recall

**ANSWER: on the paraphrase half, MRR beside recall@5.**

| arm | recall@5 | MRR |
|---|---:|---:|
| fake-dense | **0.50** | **0.4649** |
| RRF hybrid | 0.40 | **0.1633** |
| real dense | 0.40 | 0.3033 |
| BM25 | 0.30 | 0.2891 |
| BM25 + real dense | 0.30 | 0.2800 |

The hybrid gives up **0.1000 of recall and 0.3016 of MRR** against the better of
its two arms — three times as much. It is second by recall and **last** by MRR,
below both arms it is made of.

**MECHANISM: recall asks whether the answer is in the window, MRR asks where.**
Fusion moves the answer down inside the window without always moving it out.

**FINDING: the direction is not fixed.** Fusing BM25 with the real dense arm
costs 0.1000 of recall and only **0.0233** of MRR — the reverse ratio. Neither
number predicts the other, which is the argument for reporting both.

**FINDING: on the lexical half every arm scores MRR 1.0000.** "Report the MRR for
each" produces one informative column and one column of ones, exactly as recall
did.

### 3 — The retriever has no parameters to tune

`sentence_transformers`, `torch` and `transformers` are all absent — and the
lesson's dense arm has nothing to fine-tune anyway. `fake_dense_rank` is
arithmetic over token sets; two independent constructions return identical
rankings, so "pre- and post-fine-tune recall" is **0.50 twice**.

**MECHANISM: substituting a fitted encoder makes the comparison run** — and shows
what it would measure.

| SVD components | 2 | 4 | 8 | 16 | span |
|---|---:|---:|---:|---:|---:|
| paraphrase recall@5 | 0.20 | 0.20 | **0.40** | 0.20 | 0.2000 |
| lexical recall@5 | **0.70** | 0.90 | 1.00 | 1.00 | **0.3000** |

**FINDING: a hyperparameter of the untrained model spans the range fine-tuning
would claim** — and it is not monotone.

**FINDING: the confound is larger on the half everyone assumes is solved.** Only
the two widest configurations get the lexical queries right at all.

**FINDING: the target is 0.4000 against a reachable 1.0000.** Stating that gap is
more useful than a pre-and-post pair of numbers with no target between them.

**CONTROL: the loss the exercise names is decided by the batch.**
`MultipleNegativesRankingLoss` takes the rest of the batch as negatives, so 500
pairs over 10 topics is 50 per topic and about **10%** of any batch is a false
negative — a document that answers the query and is being pushed away from it.
Fixed by how the pairs were built, before training starts.

### A note on file lengths

The three files run 150 / 105 / 113 lines of code; the first is at D14's
150-line ceiling. Its overrun is the query set — twenty queries in two labelled
halves, which is the instrument the whole lesson turns on — plus a real dense arm
to compare the lesson's toy against. Exercises 2 and 3 import all of it.
