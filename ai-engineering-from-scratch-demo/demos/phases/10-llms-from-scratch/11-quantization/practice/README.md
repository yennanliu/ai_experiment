<!-- generated:start -->
# 10-llms-from-scratch / 11-quantization

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/11-quantization/) · upstream spec
`phases/10-llms-from-scratch/11-quantization/docs/en.md`

```bash
uv run demo practice run 11-quantization --ex 1
uv run demo explain 11-quantization --ex 1
uv run pytest demos/phases/10-llms-from-scratch/11-quantization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement group quantization. Instead of one scale per channel, use one scale per group of 12… | code | T0 | `ex01_halving_the_group_buys_a_third_of_a_bit.py` |
| 2 | Build a mixed-precision quantizer. Quantize the first and last layers of a multi-layer networ… | code | T0 | `ex02_which_layers_get_the_bits_does_not_matter.py` |
| 3 | Implement the straight-through estimator (STE) for quantization-aware training. Insert fake q… | code | T0 | `ex03_qat_closes_the_gap_and_loses_the_race.py` |
| 4 | Build an outlier-aware quantizer inspired by LLM.int8(). Detect channels where the activation… | code | T0 | `ex04_no_channel_reaches_three_times_the_mean.py` |
| 5 | Implement a quantization quality dashboard. Given a weight matrix, compute and display: the w… | code | T0 | `ex05_the_worst_channel_is_just_the_biggest_one.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a quantisation tour: bit-level float anatomy, symmetric and
asymmetric INT8, per-channel scales, a bit-width sweep, a simulated transformer
layer, and simulated GPTQ and AWQ. The five exercises all extend it, and four of
the five extensions turn out to be measurements of the same underlying quantity
— how uneven the weights or activations are — on a matrix where the answer is
"barely at all".

All five are **T0** on the `math` group (`uv sync --extra math`).

### 1 — halving the group buys about a third of a bit

**ANSWER: monotone in both directions, with a steady exchange rate.**

```
group   32   20.26 dB   4096 scales   12.5% overhead
group   64   19.36 dB   2048 scales    6.2%
group  128   18.63 dB   1024 scales    3.1%
group  256   18.00 dB    512 scales    1.6%
group  512   17.45 dB    256 scales    0.8%   <- per-channel
```

Each halving buys **0.55–0.90 dB** and doubles the scale count. The curve is
close to linear in `log2(group)`, so there is no knee — nothing in the sweep
identifies the 128 the exercise names as the right answer.

**FINDING: 12.5% is quoted against the wrong denominator to sound small.** A
16-bit scale per 32 4-bit weights is 12.5% of the payload, the same cost as
moving from 4 bits to 4.5. Against an fp16 baseline it reads 3.1%, which is how
the number is usually reported and why group quantisation sounds free.

**FINDING: the lesson's per-channel quantiser is the bottom row of this table.**
`quantize_per_channel(axis=0)` takes one scale per row, which on a 512-wide row
*is* group-512. The exercise presents group quantisation as what GPTQ and AWQ
use, distinct from what the lesson ships; it is the same quantiser at a
different setting.

**MECHANISM: the groups buy protection from the row maximum, and this matrix has
none.** The largest per-row magnitude is 1.74x the smallest, so the gain is a
steady fraction of a bit rather than the order of magnitude group quantisation
buys on real weights.

### 2 — which layers get the bits does not matter

**ANSWER: the mixed plan sits exactly where its bit budget says.**

```
all INT8                  cos 0.999923   mse 6.81e-06   16384 bytes
INT8 ends / INT4 middle   cos 0.986921   mse 1.17e-03   12288 bytes
INT4 ends / INT8 middle   cos 0.987401   mse 1.13e-03   12288 bytes
all INT4                  cos 0.976028   mse 2.18e-03    8192 bytes
```

75% of all-INT8's memory, about half the INT4→INT8 gap recovered.

**FINDING: the swap is not worse — it is marginally better.** INT4 at the ends
scores 1.13e-03 against the exercise's 1.17e-03 at identical memory. The premise
that the first and last layers are the ones needing precision does not hold on
this network, and the sign is the wrong way.

**MECHANISM: `tanh` compresses whatever the previous layer got wrong.** Every
output is squashed into (-1, 1) before the next matmul, so an early error is
attenuated rather than amplified. The ends are special in real networks because
the embedding and the output head carry outlier-heavy weights, not because of
their position.

**FINDING: the real decision is INT4 at all.** fp32→INT8 costs 6.81e-06;
INT8→INT4 costs **320x** that. The mixed plan is a way of answering "not quite"
while paying 75% of the price.

### 3 — QAT closes the gap and loses the race

**ANSWER: PTQ 0.0464, QAT 0.0574 at INT4.** On the number the exercise asks for,
the quantisation-aware arm loses by 24%.

**FINDING: QAT does the job it exists for, and that job is not what is scored.**
The *quantisation gap* — a model's own fp32 loss to its INT4 loss — is **1.51x**
under PTQ and **1.02x** under QAT. QAT made 4-bit inference nearly free; it just
optimised to a worse place while doing it (0.0563 fp32 against PTQ's 0.0307).

**MECHANISM: the STE lies about the gradient, and the lie costs optimisation.**

```python
first -= LR * (inputs.T @ grad_hidden / len(inputs))   # STE: straight through
```

The backward pass pretends `round()` has derivative 1, so every step is computed
at a point the forward pass never evaluated — on a 4-bit grid the master can
move without the quantised forward changing at all. That is the right trade at
scale and the wrong one on a two-layer regression that fits in fp32 anyway.

**FINDING: "compare final loss" cannot show what QAT is for.** Scoring the
endpoints charges a gap-shrinking method for its optimisation handicap. The two
numbers that answer the question are 1.51x and 1.02x.

### 4 — no channel reaches three times the mean

**ANSWER: all three thresholds select the same channels, and that number is
zero.** The Gaussian activation entering `simulate_transformer_layer` peaks at
**1.3x** its own mean, so 3x, 6x and 10x each pick 0 of 256 and the end-to-end
cosine is 0.999948 at every threshold. The outlier-aware quantiser is INT8
quantisation with extra bookkeeping.

**FINDING: a heavier tail does not help.** A lognormal activation — far more
skewed than any LayerNorm output — peaks at **1.9x**. The mean of 256 absolute
values is a stable number, and beating it sixfold takes genuinely separate
scales, not merely a long tail.

**FINDING: planting outliers makes all three thresholds identical too.** Scaling
three channels by 12x gives a peak of **13.0x**, and 3x, 6x and 10x then select
the same 3 channels. The sweep has one outcome either way: the cut only bites on
channels sitting *between* 3 and 10 times the mean, and neither arm produces one.

**MECHANISM: LLM.int8()'s 6x describes a trained residual stream.** Those
outliers are systematic — the same few hidden dimensions carry large values
across almost every token. A random matrix has no such structure, so the
detector is correct and its input has nothing in it.

### 5 — the worst channel is just the biggest one

**ANSWER: the dashboard's five panels carry one number between them.** Per-row
reconstruction RMSE correlates **0.987** with the row's own scale factor, so
"which channel quantised worst" and "which channel had the biggest weights" are
the same question and four panels are views of the fifth.

**FINDING: there is barely a range to rank.** Per-row RMSE runs 2.11e-03 to
3.68e-03 across 256 channels — a spread of **1.74x**. A dashboard whose purpose
is finding the outliers is looking at a distribution with none.

**FINDING: acting on the ranking is a bad trade.** Keeping the worst 8 rows in
fp16 improves mse by **5%** for **9.4%** more memory. The same 9.4% spent on a
smaller group size (Exercise 1) buys 0.6 dB, about 14% of mse.

**FINDING: the panel closest to user-visible quality has the least dynamic
range.** Output cosine across 100 random inputs is 0.991166 plain and 0.991671
with the eight worst channels upgraded — a difference of 5e-04, below what the
other four panels spend their space ranking.

### Two findings that did not fit an exercise

`simulated_awq`'s per-row scaling is **exactly a no-op** under per-row
quantisation: scaling a row by `s` before taking `max|w|/qmax` scales the step by
`s` too, so the reconstruction divides it straight back out
(`np.allclose(res_awq, res_pc)`, max difference 6.9e-18). And `simulated_gptq`
is **1.02x worse** than plain per-column quantisation on the lesson's own matrix
— mse 6.414e-06 against 6.297e-06, SNR 17.95 dB against 18.03 dB.
