<!-- generated:start -->
# 07-transformers-deep-dive / 14-build-a-transformer-capstone

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/14-build-a-transformer-capstone/) · upstream spec
`phases/07-transformers-deep-dive/14-build-a-transformer-capstone/docs/en.md`

```bash
uv run demo practice run 14-build-a-transformer-capstone --ex 1
uv run demo explain 14-build-a-transformer-capstone --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/14-build-a-transformer-capstone
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Verify your trained model's final-step validation loss is under 2.0… | code | T1 | `ex01_it_never_gets_under_two_and_stops_improving_at_five_hundred.py` |
| 2 | Medium. Replace learned positional embeddings with RoPE. Apply the rotation to Q and K inside… | code | T1 | `ex02_rope_is_lower_on_every_seed_and_smaller.py` |
| 3 | Medium. Implement a KV cache in the sampling loop. Generate 500 tokens with and without cache… | code | T1 | `ex03_the_cache_is_worth_block_size_not_five_to_twenty.py` |
| 4 | Hard. Add a second head to the model that predicts the next-plus-one token (MTP — Multi-Token… | code | T1 | `ex04_the_second_head_helps_by_less_than_the_seed_spread.py` |
| 5 | Hard. Replace the single FFN per block with a 4-expert MoE. Router + top-2 routing. See how v… | code | T1 | `ex05_matched_active_the_moe_is_not_better.py` |
<!-- generated:end -->

## Answers

The capstone is a 277-line PyTorch decoder that degrades to a parameter-count
printout when `torch` is missing — which it is. So the model is rebuilt in numpy
at the lesson's own configuration (`block_size=64`, `d_model=64`, 4 heads, 3
layers, `lr=3e-4`, Adam, the embedded text split 90/10) with a hand-written
backward pass. The SwiGLU sublayer is omitted from Exercises 1, 2 and 4 to keep
each file inside the repo's own 150-line ceiling; Exercise 5 puts it back,
because it is the subject there. Where two arms are compared, both share the
omission, so the comparison is unaffected.

Every training exercise is scored at its **best** checkpoint over
{250, 500, 1000} across **3 seeds**, because Exercise 1 shows the last step is
the wrong place to look. All five are **T1** (numpy).

### 1 — it never gets under 2.0, and it stopped improving before step 500

| step | 100 | 500 | 1000 | 2000 | 5000 |
|---|---:|---:|---:|---:|---:|
| train | 3.20 | 2.59 | 2.16 | 1.67 | **1.13** |
| val | 3.23 | **3.15** | 3.31 | 3.56 | **4.98** |

**ANSWER: no, and 2.0 is never reached.** Validation bottoms at **3.15** and
rises. At 5,000 steps it is **4.98** — past `ln 46 = 3.83`, the loss of guessing
uniformly among the 46 characters. Training longer makes the model worse than
having no model.

**FINDING: 63 parameters per training character.** 999 characters of text, 899
of them training, against 56,192 parameters. No configuration of this script
makes validation loss a measurement rather than a memorisation counter.

**FINDING: the validation set is 100 characters and offers 36 windows.**
`len(val_data) - block_size = 36`, drawn 16 at a time *with replacement*. Every
evaluation overlaps itself.

**FINDING: `max_steps` is 500 in the file, not 2,000.** The exercise asks you to
change a number the code does not contain — and 500 is where the curve bottoms.
The module docstring also advertises "4 layers, d_model=128, seq_len=128" where
`try_train` sets 3, 64 and 64.

### 2 — RoPE is lower on every seed, with 4,096 fewer parameters

| seed | 0 | 1 | 2 | mean |
|---|---:|---:|---:|---:|
| learned positions | 3.151 | 2.999 | 2.988 | 3.046 |
| RoPE | **2.837** | **2.987** | **2.988** | **2.937** |

**ANSWER: at least as low — lower, on 3 of 3 seeds**, and with **7.3%** fewer
parameters: the `block_size × d_model` table is not there at all. The rotation is
a function of the index, so there is nothing to learn and nothing to overfit — on
899 training characters that is the whole difference.

**CONTROL: the gradient of a rotation is the rotation transposed.** `rope` with
`sign=-1` *is* the backward pass, which is why the substitution costs nine lines.

### 3 — the cache is worth `block_size`, which is 64

**ANSWER: 64× in arithmetic.** Uncached, each step pushes the whole 64-token
window through 3 layers: **4,907,008** multiply-accumulates. Cached: **76,672**.
The ratio is exactly `block_size`, and layers, width, vocabulary and token count
all cancel — 64.0 at (3, 64), (12, 768) and (32, 4096) alike.

**FINDING: the wall clock is 3.7×, and that is where "5–20×" comes from.** 107 ms
against 29 ms for 500 tokens. At `d_model=64` a `(1, 64)` matmul and a `(1, 1)`
matmul cost about the same in dispatch, so the exercise's range is a statement
about interpreter overhead, not about the cache.

**CONTROL: the cached loop emits an identical 80-token continuation**, and the
lesson's `generate` already crops to `idx[:, -block_size:]` — which is why the
answer is 64 rather than 500.

### 4 — the second head helps by less than the seed spread

| seed | 0 | 1 | 2 | mean |
|---|---:|---:|---:|---:|
| single head | 3.188 | 3.033 | **3.022** | 3.081 |
| plus MTP (weight 0.5) | **2.879** | **3.019** | 3.083 | **2.993** |

**ANSWER: marginally — lower on 2 of 3 seeds, by 0.088 nats.**

**FINDING: the single-head arm alone spans 0.166 across those seeds** — twice the
effect. One run of each arm would report a sign chosen by the draw.

**FINDING: what it plausibly does here is regularise.** The second head costs
2,944 parameters and asks one hidden state to carry two futures. On 899
characters that is worth more as a constraint than as a prediction. DeepSeek-V3's
version is about speculative-decoding throughput — a benefit no validation loss
can show.

### 5 — at matched active parameters the MoE is a wash, for 1.5× the memory

Matched exactly: 4 experts of hidden `d_model` at top-2 activate
`2 × (2·64·64) = 16,384` FFN weights per layer, against the dense block's
`2·64·128 = 16,384`.

| seed | 0 | 1 | 2 | mean | params |
|---|---:|---:|---:|---:|---:|
| dense FFN | 2.939 | **2.896** | **2.828** | **2.887** | 105,344 |
| 4-expert MoE | **2.836** | 2.987 | 2.850 | 2.891 | **155,264** |

**ANSWER: 0.003 nats apart** — inside the **0.111** the dense arm alone spans
across the same seeds — with the MoE winning 1 seed of 3, for **1.47×** the
memory. Identical FLOPs per token; twice the FFN weights to hold.

**FINDING: there is nothing here for four experts to divide.** 899 training
characters over a 46-symbol vocabulary contain no four separable regimes, so each
expert sees a quarter of an already tiny distribution. That is Lesson 11's
locality finding arriving from the other direction: an expert earns its memory
only if the tokens it sees have something in common.
