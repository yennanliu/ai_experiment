<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 15-topic-modeling

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/15-topic-modeling/) · upstream spec
`phases/05-nlp-foundations-to-advanced/15-topic-modeling/docs/en.md`

```bash
uv run demo practice run 15-topic-modeling --ex 1
uv run demo explain 15-topic-modeling --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/15-topic-modeling
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Fit LDA with 5 topics on the 20 Newsgroups dataset. Print top 10 words per topic. Label… | code | T0 | `ex01_lda_lands_a_few_points_above_random.py` |
| 2 | Medium. Fit BERTopic on the same 20 Newsgroups subset. Compare the number of topics found, to… | code | T1 | `ex02_both_arms_sit_inside_a_coin_flip_of_random.py` |
| 3 | Hard. Compute c_v coherence for both LDA and BERTopic on your corpus. Run each with 5, 10, 20… | code | T1 | `ex03_coherence_rises_with_topic_count.py` |
<!-- generated:end -->

## Answers

Three exercises whose conclusions all rest on a baseline the lesson never
computes. Exercise 1 asks whether LDA found the real categories; against random
assignment it is three to eight points ahead. Exercise 2 asks which method
surfaces them more cleanly; the two methods differ by less than either one's
spread across seeds. Exercise 3 asks you to plot coherence against topic count;
the curve climbs, so the plot mostly reports where the sweep stopped.

Exercise 1 is **T0**; 2 and 3 are **T1** (scikit-learn for the clustering arm).

The corpus is 24 documents in four categories — finance, AI, politics, sport —
with the labels known by construction, since 20 Newsgroups is not downloadable.

### 1 — LDA lands a few points above random

**ANSWER: purity 0.5156, against a random baseline of 0.4353.**

| | purity |
|---|---:|
| one cluster containing everything | 0.2500 |
| **four random clusters** | **0.4353** |
| LDA, K=4, mean of 8 seeds | **0.5156** |
| the true partition | 1.0000 |

A purity number is unreadable without that middle value beside it, and the
exercise does not ask for it.

**FINDING: `tokenize` keeps stopwords.** It drops only tokens of two characters
or fewer, so `the`, `for`, `was` and `after` enter the 136-word vocabulary — **7
of the 32** top words across four topics are function words. The first topic
reads `['the', 'new', 'earnings', 'signed', 'yields']`.

**FINDING: removing them makes the topics readable and the assignments worse.**
Stripped, the first topic reads `['cut', 'rates', 'stocks', 'fed', 'central']` —
recognisably finance — and purity falls to **0.4635**, closer to the baseline.
Whatever makes a topic look like a category to a reader is not what is separating
the documents.

**CONTROL: the seed moves purity nearly as far as the method does.** Across 8
seeds, 0.4583 to 0.5417 — comparable to the whole margin over random.

### 2 — Both arms sit inside a coin flip of random

BERTopic is absent, so the stand-in is its shape without its encoder: TF-IDF,
truncated SVD, k-means, terms read off each centroid.

| | mean purity | spread over 4 seeds |
|---|---:|---|
| LDA | 0.4375 | 0.4167 – 0.5000 (0.0833) |
| clustering | **0.4792** | **0.3750 – 0.6250 (0.2500)** |
| random | 0.4353 | — |

**MECHANISM: the gap between the methods (0.0417) is smaller than either one's
spread across seeds.** "Which surfaces the real categories more cleanly" is
answered by the seed.

**MECHANISM: "compare the number of topics found" does not apply.** Both arms
return exactly the 4 they were asked for. BERTopic discovers its count from
embedding density; neither of these does, so the quantity the exercise treats as
an output is an input on both sides.

**FINDING: both produce readable top-word lists** — `['cut','rates','stocks',…]`
and `['championship','striker','extra',…]`. That is what both methods deliver and
what does not distinguish them.

### 3 — Coherence rises with topic count

`gensim` is absent, so c_v is replaced by **UMass** — a mean log co-document ratio
over each topic's top 8 words, needing no sliding window or reference corpus.

| K | 2 | 4 | 5 | 10 | 20 | span |
|---|---:|---:|---:|---:|---:|---:|
| LDA | −0.0990 | 0.0351 | 0.1370 | **0.1716** | 0.1159 | **0.2706** |
| clustering | −0.3177 | −0.0144 | 0.0825 | 0.2451 | **0.2967** | 0.6144 |

**MECHANISM: more topics means fewer documents per topic**, so each topic's top
words co-occur in a larger share of the documents that produced them — which is
the quantity being scored.

**FINDING: the clustering curve is monotone**, so reading a maximum off it
returns wherever the sweep stopped. The exercise stops at 50 for reasons that are
not in the metric.

**ANSWER: LDA is more than twice as stable** — 0.2706 against 0.6144.

**FINDING: stability and coherence point at different methods.** The less stable
arm is the better one at K=20 and the worse one at K=2, so the exercise's
stability question and the plot above it can be answered from the same numbers
and disagree.
