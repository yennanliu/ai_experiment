<!-- generated:start -->
# 10-llms-from-scratch / 04-pre-training-mini-gpt

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/04-pre-training-mini-gpt/) · upstream spec
`phases/10-llms-from-scratch/04-pre-training-mini-gpt/docs/en.md`

```bash
uv run demo practice run 04-pre-training-mini-gpt --ex 1
uv run demo explain 04-pre-training-mini-gpt --ex 1
uv run pytest demos/phases/10-llms-from-scratch/04-pre-training-mini-gpt
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify the model to use 24 layers and 16 heads instead of 12/12. Count the parameters. How do… | code | T1 | `ex01_the_head_count_costs_nothing.py` |
| 2 | Implement the GELU activation function (GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))) and repla… | code | T1 | `ex02_the_gap_is_the_head_start.py` |
| 3 | Add a KV cache to the generation function. Store K and V tensors for each layer after the fir… | code | T1 | `ex03_the_cache_dies_at_token_49.py` |
| 4 | Implement top-k sampling (only consider the k highest-probability tokens) and top-p sampling… | code | T1 | `ex04_the_nucleus_holds_227_of_256_tokens.py` |
| 5 | Build a training loss curve plotter. Train the model for 1000 steps and plot loss vs step. Id… | code | T1 | `ex05_there_are_not_three_phases.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a numpy GPT: embeddings, multi-head attention, a feedforward
block, a manual backward pass, a sampler. The five exercises go after
parameters, activations, the KV cache, sampling and the loss curve — and three
of the five turn out to be measuring something other than what they name. The
one fact that shapes the rest is in the backward pass: **the attention weights
never receive a gradient.** `train_mini_gpt` takes the residual path around the
attention sub-block, so `W_q`, `W_k`, `W_v`, `W_out`, `ln1` and `pos_embed` stay
at their random initialisation for every step of every run below.

All five are **T1** on the `math` group (`uv sync --extra math`) — numpy
throughout, plus `scipy.special.erf` for exercise 2's GELU.

### 1 — the head count costs nothing

Every count is `MiniGPT`'s own `count_parameters`.

| configuration | parameters | vs baseline |
|---|---:|---:|
| 12 layers, 12 heads, 768 dim | **124,402,944** | — |
| 12 layers, **16 heads**, 768 dim | 124,402,944 | **+0** |
| **24 layers**, 12 heads, 768 dim | 209,420,544 | +68.3% |
| 24 layers, 16 heads, 768 dim *(asked)* | 209,420,544 | +68.3% |
| 12 layers, 12 heads, **1536 dim**, `ff_dim` fixed | 305,392,128 | +145.5% |
| 12 layers, 12 heads, **1536 dim**, `ff_dim` follows | 418,675,200 | **+236.5%** |

The baseline is exactly GPT-2 Small's published 124M, which is the check that
the counter is right.

**FINDING: the head count contributes none of the change.** `head_dim =
embed_dim // num_heads`, so the four projections stay `768 × 768` whether they
are cut into 64-wide heads or 48-wide ones. **Heads are a reshape.** The
exercise moves two knobs and one of them is free, so the entire +68.3% is depth.

**ANSWER to the comparison: depth is 3.5× cheaper than width.** Per-block
matrices are `O(d²)` and there are six of them, so width squares what depth
multiplies — and width also doubles the 38.6M embedding table (31% of the
baseline), which depth leaves untouched.

**FINDING: "doubling the width" has two readings 113M parameters apart.** The
constructor defaults `ff_dim=3072` independently of `embed_dim`. Holding it
fixed gives +145.5%; letting it follow gives +236.5%. The gap between the two
defensible readings of the exercise's own question is nearly a whole GPT-2
Small, and the exercise names neither.

### 2 — the gap is the head start

500 steps, five seeds, only the activation swapped — in the forward *and* in
`ffn_backward`, because a GELU forward with a ReLU derivative trains a model
that does not exist.

| | final loss | drop from step 0 | seconds/run |
|---|---:|---:|---:|
| ReLU | 4.2954 ± 0.0887 | **1.2559** | 1.72 |
| GELU | 4.3720 ± 0.0938 | **1.2008** | 3.18 |

**ANSWER: a tie.** The gap is +0.0766 against a pooled standard deviation of
0.0913 — **0.84 σ**. The exercise's one-run comparison is a coin flip.

**FINDING: the gap is the head start.** At identical initialisation, step 0 is
**5.4744** under ReLU and **5.5522** under GELU. That +0.0778 is the same 0.08
that survives to step 500: `GELU(x) ≠ ReLU(x)` at random weights, so the final
loss records where the runs began. By *drop* the two are within 4%.

**FINDING: GELU costs 1.85× the wall clock for it** — `erf` runs twice per step,
once forward and once for the derivative, where ReLU is a comparison.

**FINDING: the comparison runs on 70% of a model.** After training, `token_embed`,
`ffn.W1`, `ffn.W2`, `ln2` and `ln_f` have moved; `W_q`, `W_k`, `W_v`, `W_out`,
`ln1` and `pos_embed` are bit-for-bit unchanged. The FFN happens to be one of
the two things that *does* train here, which is the only reason this exercise
has an answer at all.

### 3 — the cache dies at token 49

**ANSWER: about 3× inside the window, and exact.** 48 generated tokens, greedy
decoding, identical token sequence from both arms. Exactness is the thing to
check first — a cache that changed the output would be a different model, not a
faster one.

**FINDING: 48 of the requested 200 tokens can use it.** `train_mini_gpt` passes
`max_seq_len=seq_len=64` and `generate` slices `tokens[-64:]`. A 16-token prompt
leaves 48 positions before the window starts sliding — so the measurement the
exercise asks for is available on **24%** of the generation it names.

**MECHANISM: sliding the window invalidates the whole cache.**

```text
positions 0..63   pos_embed[0..63]   cache valid
window slides →   every token's pos_embed index shifts by one
                  every cached K and V was computed from an input that no longer exists
```

The cache has to be rebuilt at each of the remaining 152 steps, which costs more
than never caching at all.

**FINDING: a position scheme removes the first obstacle, not the last one.**
Nothing about the cache is wrong. Absolute position is — so RoPE or ALiBi would
let the surviving entries keep their indices when the window slides. That
rescues layer 0 only.

```text
change the token at position 0, hold every other token and position fixed:
  layer 0   ΔK = 0.00      depends on its own token and position, nothing else
  layer 1   ΔK = 0.32      computed from a state that attended over position 0
  layer 2   ΔK = 0.35
  layer 3   ΔK = 0.27
```

Every entry above layer 0 was computed from a hidden state that attended over
the token the window is about to evict, so it carries context the rolling window
no longer contains. Exact rolling-window attention needs those entries rebuilt.
A relative position scheme buys a cache that is cheap and approximate, not one
that is exact.

### 4 — the nucleus holds 227 of 256 tokens

Output quality scored as the share of generated bytes that are printable ASCII;
the training corpus is 100% printable, so anything else is the model failing.
Five draws of 120 bytes at temperature 0.8.

| sampler | tokens kept | mass kept | printable output |
|---|---:|---:|---:|
| top-k = 50 | 50 | 0.4796 | **82%** |
| top-p = 0.95 | **227** | 0.95 | 52% |
| no truncation | 256 | 1.00 | 56% |

**ANSWER: top-k=50 wins, by 30 points.** Neither arm decodes as valid UTF-8.

**FINDING: top-p=0.95 is not doing anything.** It removes 29 tokens where top-k
removes 206 — a factor of seven — and scores within noise of no truncation at
all. The exercise pairs the two settings as comparable and one of them is a
no-op.

**MECHANISM: a nearly flat distribution makes mass and count stop agreeing.**
Entropy 4.941 nats against a maximum of `ln(256) = 5.5452` — 89% of uniform —
with a largest probability of 0.134. When no token is confident, 95% of the mass
needs nearly every token. The two rules agree on 50 of the 227 tokens in their
union: **Jaccard 0.220**.

**FINDING: top-k=50 is the aggressive setting, not the safe one.** Its 50 tokens
carry 0.4796 of the probability — it throws away more than half of what the
model believes — and only 29 of those 50 are printable bytes. It wins by cutting
hardest, on a model flat enough that cutting hard helps.

### 5 — there are not three phases

1000 steps, the curve `train_mini_gpt` already prints.

| | steps 0–320 | steps 340–660 | steps 680–980 |
|---|---:|---:|---:|
| loss drop | **0.8135** | 0.4694 | 0.3771 |
| ratio to previous third | — | 1.73× slower | 1.24× slower |

**ANSWER: one decelerating descent, not three phases.** Each third is slower
than the last by a *shrinking* factor. There is no knee anywhere to put a
boundary at.

**FINDING: the third phase has not happened.** The minimum is at logged point 48
of 50, and over the last 100 steps the loss moves 0.1816 against a local spread
of 0.0995 — still descending faster than it wobbles. It ends at 3.7270 against
the uniform-byte baseline `ln(256) = 5.5452`, having covered 32% of the way to a
byte entropy it never reaches.

**FINDING: the overfitting is real, 2.4%, and on an axis the plot lacks.**
Overfitting is the *gap* between training and held-out loss, so it is measured at
four checkpoints with the untrained model as a control for corpus difficulty:

| step | training | held-out | gap |
|---:|---:|---:|---:|
| 0 (untrained) | 5.4889 | 5.4516 | −0.0372 |
| 250 | 4.7183 | 4.6875 | −0.0307 |
| 500 | 4.2356 | 4.2303 | −0.0053 |
| 750 | 3.9175 | 3.9416 | +0.0241 |
| 1000 | 3.7052 | 3.7573 | **+0.0522** |

The gap is monotone, so the model *is* beginning to fit the training sentences
specifically. But the untrained model already scores −0.0372 on the same two
corpora, purely because one is easier to predict than the other, so the
corpus-corrected figure at step 1000 is **+0.0894 — 2.4%** of the training loss.

Every number in that table comes from a held-out corpus the requested plot does
not contain. The third phase is named after a quantity that is not on the axis,
at a point where the loss is still falling.

**FINDING: the curve is produced with 30% of the model frozen.** `W_q`, `W_k`,
`W_v`, `W_out` and `pos_embed` are bit-for-bit unchanged after 1000 steps. "The
shape of this curve is the same whether you are training a 128-dim model or
GPT-4" is claimed for a curve produced with attention frozen at random
initialisation — which is the part of GPT-4 that the curve is mostly about.
