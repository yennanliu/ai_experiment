<!-- generated:start -->
# 04-computer-vision / 20-image-retrieval-metric

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/20-image-retrieval-metric/) · upstream spec
`phases/04-computer-vision/20-image-retrieval-metric/docs/en.md`

```bash
uv run demo practice run 20-image-retrieval-metric --ex 1
uv run demo explain 20-image-retrieval-metric --ex 1
uv run pytest demos/phases/04-computer-vision/20-image-retrieval-metric
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Run the toy example above. Plot the embeddings with PCA before and after training to s… | code | T1 | `ex01_pca_before_after_margin.py` |
| 2 | (Medium) Add a ProxyNCA loss implementation: one learned "proxy" per class, standard cross-en… | code | T1 | `ex02_proxynca_vs_triplet_speed.py` |
| 3 | (Hard) Take 1,000 ImageNet validation images, embed with DINOv2 via HuggingFace, build a FAIS… | code | T1 | `ex03_recall_index_not_model.py` |
<!-- generated:end -->

## Answers

Metric learning is a field where the metric is the thing most likely to be
wrong, and all three exercises here turn out to be about that rather than about
the encoder. Two of the three ask for a number that is already saturated before
training starts.

### 1 — What the two PCA plots actually differ by

| view | between/within variance | recall@1 |
|---|---:|---:|
| raw 128-d input | 0.312 | **1.000** |
| untrained encoder | 0.230 | 0.700 |
| trained encoder | **14.339** | 1.000 |

**ANSWER: the margin grows 62×** — variance ratio 0.230 → 14.339, recall@1
0.700 → 1.000, triplet loss 0.21601 → 0.00000.

**FINDING: the six clusters are in the input; nothing forms them.** `main()`
builds every sample as `protos[label] + 0.15 · randn`, so the raw vectors
already retrieve at **recall@1 = 1.000**. Training multiplies the margin by 46×;
it does not create the structure the exercise asks you to watch appear.

**CONTROL: "before training" is worse than no encoder at all.** The untrained
`Encoder` is a random 128 → 64 projection and it *loses* separability the input
had — recall@1 **0.700** against the raw **1.000**. The left-hand plot is not a
neutral starting point.

**MECHANISM: six clusters span exactly five dimensions.** Explained variance per
component is 0.2482, 0.2137, 0.2023, 0.1746, 0.1548, then **0.0017** — a 92×
collapse at the sixth, because C centroids span C−1 dimensions. So the plot
shows 46.2% of a five-dimensional object; on data this easy the discarded half
costs nothing, and recall@1 read from the two-component projection alone is
still 1.000.

**FINDING: two thirds of the run has no gradient.** The triplet loss first
reaches exactly 0.0 at step **16** and is zero on **132 of 200** steps — relu
clamps it once every triplet clears the 0.2 margin.

**CONTROL: `semi_hard_negatives` is hardest-negative mining in practice.** The
semi-hard window `d_ap < d_an < d_ap + margin` is empty for a mean **45.3 of
48** anchors — 4 at the first step, **48 at the last** — so the function's own
fallback chooses for 94% of them. Its name describes the intent, not the
behaviour on this data.

### 2 — ProxyNCA against triplet, measured on something they share

Their loss values are not comparable (a hinge bottoms out at 0, a cross-entropy
does not), and retrieval saturates, so the comparison runs on the margin.

| arm | steps to variance ratio 5.0 | final ratio | zero-loss steps | extra params |
|---|---:|---:|---:|---:|
| triplet | **20 / 20** | **14.20** | 132 / 131 of 200 | 0 |
| ProxyNCA | 30 / 30 | 7.79 | 0 / 0 | 384 |

**ANSWER: on the exercise's own metric there is nothing to report** — all four
runs reach recall@1 = 1.000 by step 10. Convergence speed measured on recall is
a tie by construction.

**ANSWER: on the margin, triplet converges 1.5× sooner and ends 1.8× higher**,
with both seeds agreeing and the step counts identical across them.

**MECHANISM: the proxies replace the batch.** Triplet scores a 48×48 =
**2,304**-entry distance matrix each step and then picks from it; ProxyNCA scores
48×6 = **288** similarities — 8× fewer — and has no mining rule to get wrong: no
positive search, no semi-hard window, no fallback.

**FINDING: the triplet loss is dead on most steps and still wins.** Its gradient
is sparse in time, but each surviving step pushes the hardest negative it can
find, and that is what carries the larger final margin.

**CONTROL: ProxyNCA's convenience is paid for in parameters that never ship** —
6×64 = **384** learned values, 1.55% on top of the encoder's 24,768, discarded at
inference.

### 3 — "(should be 1.0)" tests the index, not the model

`transformers`, `faiss` and `datasets` all raise `ModuleNotFoundError`, so the
retrieval is rebuilt on the lesson's own `Encoder` and `recall_at_k` over an
exact inner-product search — which is what a FAISS *flat* index computes, and the
one index type for which the exercise's parenthesis is guaranteed.

| arm | self-query @1/5/10 | held-out @1/5/10 |
|---|---|---|
| trained | 1.0000 / 1.0000 / 1.0000 | **1.0000** / 1.0000 / 1.0000 |
| untrained | **1.0000 / 1.0000 / 1.0000** | 0.8150 / 0.9750 / 1.0000 |
| collapsed | 0.1500 / 0.6780 / 0.8440 | — |
| closed-form null | — | **0.1674 / 0.5993 / 0.8387** |

**ANSWER: self-query recall is 1.0 as predicted — and 1.0 untrained too.** The
check passes identically whether or not the model has learned anything.

**MECHANISM: a normalised vector is its own nearest neighbour.** Cosine
similarity to itself is exactly 1.0, the maximum, so top-1 returns the query and
its label matches by construction. The only way to fail is ties — which is why a
collapsed encoder scores 0.1500 instead of 1.0000.

**FINDING: recall@10 on six classes is 0.84 before any model exists.** For
embeddings carrying no class information, recall@k is `Σ_c f_c(1−(1−f_c)^k)`;
random unit vectors match that closed form to **0.0037**. The exercise asks for
the number without asking for the baseline.

**FINDING: only recall@1 separates trained from untrained** — a gap of 0.1850 at
k=1 that closes to **0.0000** by k=10. Reporting all three hides the only one
carrying information.

**CONTROL: a collapsed index scores at the null, not at 1.0** — within 0.0787 of
the closed form, because ties resolve by index order rather than uniformly.

At full scale the exercise wants `facebook/dinov2-base` over the ImageNet val
split: an estimated ~350 MB of weights plus the split, and seconds of GPU time
for 1,200 embeddings. It is a download problem, not a compute one.

### A note on file lengths

The three files run 144 / 140 / 145 lines of code, over D14's 120-line target and
under its 150-line ceiling. Exercises 2 and 3 import exercise 1's `world` and
`spread` through `practice.load_module` rather than carrying a second and third
copy of the lesson's sampler.
