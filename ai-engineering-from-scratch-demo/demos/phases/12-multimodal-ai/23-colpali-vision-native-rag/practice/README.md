<!-- generated:start -->
# 12-multimodal-ai / 23-colpali-vision-native-rag

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/23-colpali-vision-native-rag/) · upstream spec
`phases/12-multimodal-ai/23-colpali-vision-native-rag/docs/en.md`

```bash
uv run demo practice run 23-colpali-vision-native-rag --ex 1
uv run demo explain 23-colpali-vision-native-rag --ex 1
uv run pytest demos/phases/12-multimodal-ai/23-colpali-vision-native-rag
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | A 200-page annual report at 729 patches per page, 128-dim emb, 4-byte floats. Compute raw sto… | code | T0 | `ex01_the_index_size_is_a_vision_tower_choice.py` |
| 2 | MaxSim is Σ_i max_j cos(q_i, p_j). What does this sum capture that a simple mean similarity d… | code | T0 | `ex02_maxsim_is_invariant_to_padding_and_linear_in_query_length.py` |
| 3 | ColPali indexes pages as patch sets. What changes if we instead index at the word level (as C… | code | T0 | `ex03_word_level_is_smaller_and_costs_the_thing_colpali_removed.py` |
| 4 | Design the end-to-end pipeline for a 1M-page corpus with a latency budget of 500ms per query.… | code | T0 | `ex04_two_stage_or_ten_thousand_times_the_budget.py` |
| 5 | Read M3DocRAG (arXiv:2411.04952). Describe the multi-page attention pattern and how it differ… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Four code solutions and one prose one. Exercises 1, 3 and 4 are one arithmetic
argument in three places: the index is large, the compression is the cheapest way
to shrink it, and at corpus scale a single stage is unaffordable in memory before
it is unaffordable in time.

### 1 — the index size is a vision-tower choice

| | per page | 200 pages |
|---|---:|---:|
| ColPali raw, 729 × 128 × 4 | **364.5 KiB** | **71.2 MiB** |
| ColPali PQ 8× | **45.6 KiB** | **8.9 MiB** |
| text-RAG / VisRAG | 3.0 KB | 0.6 MB |

**ANSWER: 71.2 MiB raw, 8.9 MiB compressed** — reproducing the lesson's own "365
KB" and "46 KB" rows.

**FINDING: even compressed, the index is 15.2× a text-RAG one** (raw: 121.5×).
The compression closes most of the gap and none of the comparison.

**FINDING: the 729 is a tower setting, not a document property.** It is 27 × 27 —
SigLIP SO400m at 384. The same page through Qwen2.5-VL at 1280×720 is **4,641**
patches (Lesson 12.01), **6.37×** the storage at **2.27 MiB** a page.

**FINDING: PQ is a lossy codec applied to exactly what MaxSim maximises over.**
The lesson reports 8× and **zero** recall figures. The same shape as Lesson
12.21's FAST tokenizer, which reports a ratio and ships no inverse.

### 2 — MaxSim is invariant to padding and linear in query length

| page patches | 4 | 8 | 20 | 68 | **729** |
|---|---:|---:|---:|---:|---:|
| MaxSim | 1.4776 | 1.4776 | 1.4776 | 1.4776 | **1.4776** |
| mean | 0.5007 | 0.2752 | 0.1399 | 0.0763 | **0.0522** |

**ANSWER: it captures the best evidence per query token and ignores the rest of
the page.** A mean asks whether a page is *about* the query; a max asks whether
the answer is *on* it.

**FINDING: which is why it cannot be compared across pages of different sizes.**
The invariance is the feature; its cost is that a 729-patch page and a 4-patch
page are scored on one scale with no normalisation for how many chances each had.

**FINDING: the score is linear in query length** — 0.774, 1.478, 2.955, 14.776 at
1, 2, 4, 20 tokens (**0.7388** each). A fixed relevance threshold is a different
threshold for every query.

**FINDING: one patch can serve every query token** — **1.481** for one shared
patch against **2.000** for two specialised ones. The same double-counting Lesson
12.17 finds in its grounding metric and Lesson 12.19 in its diarization scorer: a
max with no assignment step.

### 3 — word-level is smaller, and costs the thing ColPali removed

**ANSWER: the index shrinks 31.4% and the pipeline grows an OCR pass.** 500 words
is **250.0 KiB** against 729 patches' **364.5 KiB** — the same *order*, because a
word and a patch are about the same size on a page.

**FINDING: quantisation beats changing the unit by 5.5×.** ColPali at PQ 8× is
**45.6 KiB** against word-level raw's 250.0 — and it does not require reading the
page.

**FINDING: what the unit change actually costs is everything not in the text
stream** — checkboxes, stamps, table rules, chart bars, signatures, column
alignment. Lesson 12.22 measures the alternative: LayoutLMv3's bbox stream
restores position, gives up scale invariance, and depends on an OCR pass whose
per-field error that lesson states at **2%**.

**FINDING: and the failure modes invert.** Patches degrade gracefully; words fail
discretely. A word that was never extracted is unretrievable at any recall, and
MaxSim over patches has no equivalent of a missing row.

### 4 — two-stage, or ten thousand times the budget

| | index | ops per query |
|---|---:|---:|
| single-stage MaxSim over 1M pages | 46.7 GB (PQ) | **1.87 trillion** |
| pooled ANN (VisRAG) | **3.1 GB** | milliseconds |
| MaxSim rerank, top 100 | — | **186.6 million** |

**ANSWER: VisRAG first, ColQwen2 second, top-100 rerank.**

**FINDING: single-stage MaxSim is 10,000× too much work** — the same arithmetic
on **0.01%** of the corpus fits the 500 ms budget with room for the generator.

**FINDING: and it is unaffordable in storage before it is in compute.** 373.2 GB
raw, 46.7 GB compressed, against 3.1 GB pooled — **15.2×**. The first stage is
not a speed optimisation; it is the only stage that fits in memory.

**ANSWER: the justification is the recall hand-off.** A pooled vector asks
whether a page is *about* the query — what Exercise 2 shows a *mean* similarity
does — which is the right question for a stage that must not miss. MaxSim then
asks whether the answer is *on* the page, the right question once there are a
hundred candidates. The number to watch is **first-stage recall@100**, because
nothing downstream recovers a page it did not return.

### 5 — how M3DocRAG differs from single-page retrieval

Drawing on **M3DocRAG**, which the lesson describes as extending "multi-modal
retrieval to multi-page multi-document reasoning… composes a multi-page context
for the VLM".

**ColPali retrieves pages; M3DocRAG retrieves *and composes*.** The difference is
one step, and it changes what questions are answerable.

**What single-page retrieval assumes.** Top-*k* pages are scored **independently**
and handed to the generator as *k* separate images. Exercise 2 shows why that is
the natural shape of MaxSim: the score is a sum over query tokens of a max over
*one page's* patches, so a page that holds half the answer scores half as well as
one that holds all of it, and there is no term for two pages holding complementary
halves. The scoring function has no way to prefer a *set*.

**What that makes unanswerable, concretely:**

- "How did segment revenue change between the 2023 and 2024 reports?" — the
  answer is on two pages in two documents, and neither page alone scores highly
  on the full query.
- "Which of these three contracts has the shortest notice period?" — requires all
  three, and a top-3 that returns three pages of *one* contract is a plausible
  MaxSim outcome.
- Anything whose answer is an aggregation, a comparison or an absence.

**The three things a multi-page design has to add, in order:**

1. **Cross-page attention at generation time.** The pages enter one context, so
   the VLM attends across them rather than being called *k* times and asked to
   merge the answers afterward. This is where the token budget bites: Lesson
   12.01's arithmetic puts a native-resolution page in the thousands of tokens,
   so "compose a multi-page context" is a context-length problem before it is a
   modelling one.
2. **Set-aware selection, not top-*k*.** If two pages carry the same evidence,
   the second adds nothing; if they carry complementary halves, both are needed
   and neither ranks well alone. That is a diversity or coverage objective over
   the retrieved set, and MaxSim computes a score per page with no term for what
   the other candidates already provide.
3. **Document-level structure in the index.** Page 47 of report A and page 47 of
   report B are not interchangeable, and "the 2024 report" is a filter rather than
   a similarity. A flat page index cannot express it.

**And the honest caveat.** Points 2 and 3 are retrieval changes; point 1 is a
generation change, and it is the one the lesson's one-line description actually
names. A system that retrieves independently and merely *concatenates* the top-*k*
into one prompt gets the cross-page reasoning and none of the set-aware
selection — which is the cheap version, is what most implementations do, and
fails precisely on the comparison questions above, because the pages it needed
were never in the top *k* to concatenate.
