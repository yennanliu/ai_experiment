<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 10-attention-mechanism

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/10-attention-mechanism/) · upstream spec
`phases/05-nlp-foundations-to-advanced/10-attention-mechanism/docs/en.md`

```bash
uv run demo practice run 10-attention-mechanism --ex 1
uv run demo explain 10-attention-mechanism --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/10-attention-mechanism
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement `softmax` masking so padding tokens in the encoder get attention weight zero.… | code | T0 | `ex01_a_fully_masked_row_returns_uniform_weights.py` |
| 2 | Medium. Add multi-head attention to the Luong `general` form. Split `d_h` into `n_heads` grou… | code | T0 | `ex02_one_head_matches_exactly_two_do_not.py` |
| 3 | Hard. Train a GRU encoder-decoder with Bahdanau attention on the toy copy task from lesson 09… | code | T1 | `ex03_the_width_stops_growing_with_length.py` |
<!-- generated:end -->

## Answers

Three exercises where the stated task is correct and its edges are not.
Exercise 1's masking is right on the case it names and wrong on the two beside
it. Exercise 2's verification passes bit-for-bit, and everything above one head
is a different function rather than the refactor the wording implies. Exercise 3's
predicted gap is real, and the number underneath it is better than a gap.

Exercises 1 and 2 are **T0** (the lesson's `code/main.py` is pure Python);
exercise 3 is **T1** (numpy).

### 1 — A fully masked row returns uniform weights

**ANSWER: a large negative sentinel before the softmax is exactly right.** With 3
real positions of 5, the padded weights come back **0.0** and the context matches
`dot_attention` over the unpadded prefix to **1.11e-16**. Unmasked, the same
batch sends **26.6%** of its attention mass to two all-zero vectors.

**FINDING: a row where everything is padding comes back uniform.** `softmax`
subtracts the maximum, so five identical sentinels become five identical
exponentials: weights `[0.2] × 5`, summing to **1.0**, no error raised.

**MECHANISM: the answer it returns is the plain mean of every state.** The
uniform context is `[0.32, 0.28, 0.12]`, exactly the unweighted mean of all five
encoder states — padding included. And the case is not hypothetical: it is a
batch row whose source is empty, or one truncated below the batch minimum.

**FINDING: `-inf` fixes that case by breaking it differently.**

| sentinel | partially masked | fully masked |
|---|---|---|
| −1e9 | exact zeros | **uniform 0.2** |
| −inf | exact zeros | **nan everywhere** |

The two obvious sentinels fail on the same input in opposite directions, and only
one of them says so.

**FINDING: zeroing *after* the softmax scales the context by the mass it
dropped.** The surviving weights sum to **0.733804** rather than 1, so the
context is the right direction times the retained mass — the ratio to the correct
context is `[0.733804, 0.733804, 0.733804]`, identical in every dimension. A pure
scaling, which is why it survives a spot check.

**CONTROL: a zero vector is not a neutral one.** The padded states are all-zero,
so their raw score is 0.0 — which earns more softmax mass than any single real
position does after masking.

### 2 — One head matches exactly, two do not

**ANSWER: with one head the match is exact, not close.** Context and weights both
differ from the single-head implementation by **0.0** — splitting an 8-vector
into one group is the identity and every operation runs in the same order, so the
verification is available as an equality rather than a tolerance.

**MECHANISM: above one head it is a different function.**

| heads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| context distance from single-head | **0.0** | 0.6436 | **2.1025** | 1.8818 |
| per-head argmax | [2] | [2, 2] | **[0, 0, 0, 5]** | — |

One head takes a single softmax over the full inner products; h heads take h
softmaxes over disjoint slices, and **softmax does not distribute over a sum**.
At 4 heads they land on two positions, neither of them the single head's 2. At 2
heads they agree about *where* to look and still move the context — so the weight
divergence is not guaranteed and the output divergence is.

**CONTROL: with the projection set to the identity, multi-head still disagrees.**
No learned parameters remain for the heads to differ on, and two heads still move
the context **0.9323**, with the second head peaking at 3 where the single head
peaks at 1. Nothing here is a parameter difference; it is the normalisation being
applied to slices.

**FINDING: eight heads is closer to single-head than four.** With one dimension
per head each softmax is over six nearly-tied scores, so every head approaches a
uniform average and the concatenation approaches the mean of the encoder states.

**CONTROL: the form the exercise extends is not in the lesson's code.**
`code/main.py` ships `dot_attention` and `additive_attention` and no `general`,
so the exercise extends something that has to be written first.

### 3 — The width stops growing with length

torch is not installed, so both arms are fixed random encoders with least-squares
decoders — no optimisation to confound the comparison.

**ANSWER: at d=16 the gap widens from nothing to everything.**

| length | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| no attention | 1.0000 | 0.4150 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| attention | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

**MECHANISM: the baseline is under-provisioned, not broken.** Give it the width
its rank demands and it comes back exactly:

| length | rank `L(V−1)+1` | exact-match at that width |
|---:|---:|---:|
| 2 | 19 | **1.0000** |
| 4 | 37 | **1.0000** |
| 8 | 73 | **1.0000** |

**FINDING: that requirement grows linearly** — 577 dimensions to copy 64 symbols.
Doubling the source doubles the context the baseline needs.

**ANSWER: the width attention needs does not move with the sequence at all.** It
reaches 1.0000 at **d = 8** for lengths 4, 16 and 64 alike. Its requirement is a
function of the *vocabulary*, not the sequence — which is what lifting the
bottleneck means, stated as a number rather than as a gap.

**FINDING: below its own threshold attention degrades with length too.** At d=4
it scores 0.625 / 0.015 / 0.000 at lengths 4 / 16 / 64 — nothing ran out of room,
a per-token error rate compounds. So the accuracy-against-length plot the
exercise asks for moves for **two** reasons, and only one of them is the
bottleneck.
