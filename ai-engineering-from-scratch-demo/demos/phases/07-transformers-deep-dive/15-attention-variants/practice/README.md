<!-- generated:start -->
# 07-transformers-deep-dive / 15-attention-variants

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/15-attention-variants/) · upstream spec
`phases/07-transformers-deep-dive/15-attention-variants/docs/en.md`

```bash
uv run demo practice run 15-attention-variants --ex 1
uv run demo explain 15-attention-variants --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/15-attention-variants
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Verify SWA at `window=4` zeroes everything outside the last 4 token… | code | T0 | `ex01_the_first_window_rows_are_identical.py` |
| 2 | Medium. Implement causal SWA with `window=1024` on top of the Lesson 07 capstone. Train for 1… | code | T0 | `ex02_a_window_of_1024_on_64_tokens_is_full_attention.py` |
| 3 | Hard. Implement a Gemma-3-style 5:1 layer mix (5 SWA, 1 global) in the capstone model. Compar… | code | T0 | `ex03_the_global_sixth_is_ninety_six_per_cent_of_the_cache.py` |
| 4 | Hard. Implement differential attention with a learned `λ` per head. Train on a synthetic retr… | code | T0 | `ex04_an_unlearned_lambda_subtracts_the_needle_too.py` |
<!-- generated:end -->

## Answers

The lesson is 157 lines of mask builders plus a KV-cache table. Exercise 1's own
demo contradicts its own caption. Exercises 2 and 3 specify a window and a layer
ratio that the capstone they name cannot express. Exercise 4 asks for a metric
that cannot see the effect it is about. All four are **T0**.

### 1 — the first `window` rows of an SWA mask are causal already

**ANSWER: both claims hold exactly.** `swa_mask(n, n) == causal_mask(n)` cell for
cell, because `max(0, i − n + 1)` is 0 for every row. At `window=4` on 8 tokens,
row `i` attends `[max(0, i−3) .. i]` — 26 of the 36 causal cells.

**FINDING: only rows 4–7 differ.** The window is not full until row `window−1`,
so the first four rows are unchanged. A sliding window does nothing until the
sequence is longer than the window — which is Exercise 2's entire content.

**FINDING: the demo's own draw has the *least* weight on position 0.** `main()`
prints `[0.018, 0.068, 0.335, 0.031, 0.054, 0.112, 0.156, 0.227]` next to
"notice the weight bleeding to position 0 — the attention sink". Position 0 is
**0.018**, the smallest of the eight and a seventh of the uniform 0.125. Over 200
fresh draws the mean is **0.1206** against a uniform 0.1250 — still below. The
sink is a property of *trained* models; a softmax over random scores has no
reason to produce one.

### 2 — a window of 1,024 on a 64-token block is full attention

| window | equals causal at n=64 | cells | rows changed |
|---:|---|---:|---:|
| 1024 | **yes** | 2,080 | **0** |
| 64 | **yes** | 2,080 | **0** |
| 32 | no | 1,552 | 32 |
| 8 | no | 484 | 56 |

**ANSWER: val loss regresses by nothing and memory drops by nothing**, and no
training run is needed to say so. `swa_mask(64, 1024) == causal_mask(64)`
identically; the KV cache is `min(window, n)` deep and `min(1024, 64) = 64`.

**FINDING: the exercise's own numbers are 16× apart.** The window is sixteen
times the context. Both halves of the question have the same answer for that one
reason, and it is 0.000 rather than "about zero" because it is a property of the
mask, not of the scores.

### 3 — the one global layer in six is 96% of the mix's cache

At 128K context, 80 layers, 8 KV heads, `d_head=128`, fp16:

| configuration | KV cache | vs full |
|---|---:|---:|
| full attention | 42.9 GB | 1.0× |
| **Gemma 5:1, W=1024** | **7.44 GB** | **5.8×** |
| pure SWA, W=1024 | 0.34 GB | 128× |

**ANSWER: 5.8×, against a ceiling of exactly `n_layers / global_layers = 6`** —
96% of what the ratio allows.

**FINDING: 96.2% of that cache is the single global layer.** The five SWA layers
contribute **3.76%**; quadrupling the window from 1024 to 4096 moves the total
only from 7.44 to 8.28 GB. The 5:1 mix is, to within 4%, "one sixth of full
attention".

**FINDING: against pure SWA the mix is a 22× regression**, bought deliberately to
keep retrieval over the whole context. Same number, other direction.

**FINDING: the capstone cannot express the ratio.** 3 layers; 5:1 wants a
multiple of 6. And at `block_size=64` the pure-SWA and pure-global baselines are
bit-identical models, so the loss and generation comparisons are undefined too.

### 4 — an unlearned λ subtracts the needle along with the noise

One planted needle whose key is a scaled copy of the query, among Gaussian
distractors:

| distractors | λ=0 (single) | λ=0.5 | λ=0.8 |
|---:|---:|---:|---:|
| 64 | 100.0% | 99.0% | 94.0% |
| 512 | 99.0% | 98.5% | 83.5% |
| 2,000 | **100.0%** | 95.5% | **72.5%** |

**ANSWER: single attention is 100% at 2,000 distractors; a fixed λ is worse, and
monotonically.** The map this implementation subtracts is a noisy copy of the
first (`K2 = K1 + noise`), so it scores the needle highly too — removing it
removes signal and noise together. That is exactly what the word "learned" is
carrying in the exercise's own sentence.

**FINDING: the weights sum to `1 − λ`, not to 1** — measured at 1.000000,
0.700000, 0.500000 and 0.200000. `diff_attention_row` returns the difference of
two normalised distributions, unnormalised, so the output vanishes at λ = 1. The
Differential Transformer normalises after the subtraction; this does not.

**FINDING: retrieval accuracy cannot see the phenomenon.** Plain attention holds
100% from 8 distractors to 2,000 while the needle's share of the mass falls from
**0.94 to 0.28**. The argmax stays right long after the distribution stops being
peaked, so the metric the exercise asks for is the insensitive one.

**CONTROL: the lesson's own KV figure overcharges it.** `main()` prints
differential attention at `full × 2` = 85.9 GB. It caches `K1`, `K2` and one
shared `V` — three units against full attention's two — so it is **1.5×**,
64.4 GB. The extra query projection costs parameters, not cache.
