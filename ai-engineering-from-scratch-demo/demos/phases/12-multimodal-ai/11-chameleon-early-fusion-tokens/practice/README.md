<!-- generated:start -->
# 12-multimodal-ai / 11-chameleon-early-fusion-tokens

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/11-chameleon-early-fusion-tokens/) · upstream spec
`phases/12-multimodal-ai/11-chameleon-early-fusion-tokens/docs/en.md`

```bash
uv run demo practice run 11-chameleon-early-fusion-tokens --ex 1
uv run demo explain 11-chameleon-early-fusion-tokens --ex 1
uv run pytest demos/phases/12-multimodal-ai/11-chameleon-early-fusion-tokens
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Chameleon uses K=8192 codebook entries and 1024 tokens per 512x512 image. Estimate the compre… | code | T0 | `ex01_three_of_the_four_images_get_the_same_codes.py` |
| 2 | A 4K image (3840x2160) at the same VQ-VAE density produces how many image tokens? Can a Chame… | code | T0 | `ex02_context_breaks_at_exactly_one_megapixel.py` |
| 3 | Implement QK-Norm in pure Python. Given a 64-dim query and key, show the dot product before a… | code | T0 | `ex03_qk_norm_makes_the_logit_invariant_to_its_input_scale.py` |
| 4 | Read Chameleon Section 2.3 on training stability. Describe the exact failure mode the paper o… | explain | T0 | prose, below |
| 5 | Extend the toy decoder to emit a mixed-modality response given a text-only prompt. Measure ho… | code | T0 | `ex05_no_generation_ever_puts_text_after_the_image.py` |
<!-- generated:end -->

## Answers

Four code solutions and one prose one. Exercises 1 and 5 both find the lesson's
toy failing in a way that is *not* a toy artefact: the tokenizer collapses three
of its four images into one code sequence, and the decoder can never produce the
mixed-modality output it exists to demonstrate.

### 1 — three of the four images get the same codes

**ANSWER: 472.6×, and yes.**

| | |
|---|---:|
| 512×512 at 24 bits | 6,291,456 bits |
| 1,024 tokens × log₂(8192) | **13,312 bits** |
| ratio | **472.6×** |
| rate | **0.0508** bits/pixel |
| vs a 1 bpp JPEG (stated reference) | **19.7×** below |

A rate that far below JPEG is not compressing an image; it is describing one.

**FINDING: on the lesson's own tokenizer, three of four images are the same
image.**

| image | codes | patch MAE |
|---|---|---:|
| red | (34, 36, 36, 36) | 3.188 |
| **blue** | **(38, 38, 38, 38)** | 1.562 |
| green | (34, 36, 36, 36) | 1.562 |
| gray | (34, 36, 36, 36) | 1.688 |

Four visually distinct inputs, **2** distinct code sequences. Everything
downstream of the tokenizer — including the bigram trained on them — cannot tell
red from green from gray.

**FINDING: the reconstruction error is comparable to the signal.** On a scale
whose whole range is 0–9, the nearest codebook entry to `[8, 7, 7, 8]` is
`[6, 1, 4, 7]`.

**FINDING: the toy compresses 15× less than the thing it illustrates** — 32.0×
against 472.6×. The collisions happen at a fifteenth of the real compression,
which is what makes them worth reporting rather than dismissing.

### 2 — context breaks at exactly one megapixel

**ANSWER: 32,400 tokens, and no** — **7.9×** a 4,096-token context.

| limit | threshold | resolution |
|---|---:|---|
| **context**, 4,096 tokens | **1.0 MP** | **1024 × 1024** |
| KV cache, 16 GiB | 8.4 MP | 2896 × 2896 |
| tokenizer quality | — | never |

**ANSWER: context breaks first, at exactly 1024×1024** — 4,096 tokens is one
megapixel to the pixel, and the cache has **8×** more headroom.

**FINDING: tokenizer quality never breaks, because it does not depend on
resolution.** The rate is 0.0508 bits per pixel at 512, 1024, 2048 and 3840 —
identical. The third candidate in the exercise's list is the one quantity that is
scale-invariant: it was already broken at 512×512 and it gets no worse.

**FINDING: the 4K cache is 15.82 GiB, which is the model again.** At 512 KiB a
token, a 4K image costs more in KV than a 7B model costs in bf16 weights.

### 3 — QK-Norm makes the logit invariant to its input scale

**ANSWER: −1.9794 before, −1.7674 after** — barely a change, because the fixture
is already unit-variance. Which is exactly why one pair demonstrates nothing:

| input scale | raw q·k | after LayerNorm |
|---:|---:|---:|
| 1× | −1.979 | −1.767434 |
| 2× | −7.918 | −1.767446 |
| 10× | −197.939 | −1.767450 |
| 100× | **−19,793.943** | **−1.767451** |

**FINDING: the raw logit grows as s², because both operands scale.** The
normalised one does not move to six decimals.

**FINDING: after normalisation the logit is bounded, and the bound is d.** Every
normalised vector has norm exactly **8.0** = √64, so the dot is `64·cos θ` ∈
[−64, 64] — [−8, 8] after the 1/√d attention scale — whatever the model does
upstream. A softmax over that range cannot saturate.

**FINDING: at depth the input scale is not a constant.**

| residual growth | norm after | logit after |
|---|---:|---:|
| 1.05× per layer, 48 layers | 10.4× | **108×** |
| 1.10× per layer, 32 layers | 21.1× | **446×** |
| 1.50× per layer, 32 layers | 4.3e5× | **1.9e11×** |

The quantity QK-Norm bounds is the one that compounds, and it compounds squared.

### 4 — the norm-explosion signature at 34B

Drawing on **Training stability — QK-Norm, dropout, LayerNorm ordering**, which
is where the lesson lists the three interventions and says all of them were
needed at scale.

**The failure mode is a divergence that looks stable until it isn't.** What the
paper describes at 34B is not a loss that degrades — it is a loss that tracks the
7B curve, sometimes for a fifth of training, and then spikes and never recovers.
That is what makes it expensive: the run has to be most of the way finished
before it tells you it was wrong.

**The signature is a slow growth of the norms feeding attention, and a fast
collapse of what attention does with them.** Two curves, one lagging the other:

1. **The residual-stream norm creeps upward with depth and with step.** Each
   block adds to the residual and nothing removes; under post-normalisation the
   later layers see progressively larger inputs. This part is gradual, monotone,
   and looks like nothing — a few percent a layer.
2. **The attention logits follow it squared,** because the logit is a dot product
   of two vectors that are both growing. Exercise 3 puts numbers on the
   amplification: 5% per layer over 48 layers is 10.4× in norm and **108×** in
   the logit; 10% is 21× and **446×**.
3. **Then the softmax saturates,** the attention distribution goes effectively
   one-hot, its entropy collapses, and the gradient through it goes to zero. The
   layer stops learning while its norms keep growing, and the loss spike is the
   visible end of a process that started long before.

**Why 34B and not 7B.** The two runs differ in depth, so the compounding in (1)
has more layers to compound through — the effect is exponential in depth and the
intervention that sufficed shallower does not. The lesson's heading records that
asymmetry directly: QK-Norm plus dropout held the smaller model; the larger one
needed the **normalisation ordering** changed as well, because dropout attacks
the symptom in the attention map while the norm growth is in the residual stream.

**Why early fusion makes it worse, which is the part specific to this lesson.**
Chameleon's residual stream carries text tokens and image tokens in one
sequence, one vocabulary, one loss. The two populations have different
statistics — a 13-bit VQ code drawn from 8,192 entries is not distributed like a
BPE token — so their embedding norms and their attention entropies differ from
step one. A shared softmax then puts them in competition for the same
normalisation, and the modality with the larger norms takes the attention mass.
Late-fusion architectures do not have this problem because the image never enters
the residual stream as tokens the LM's own softmax must rank.

**What QK-Norm does about it, precisely.** It does not stop the residual stream
from growing; it stops the growth from *reaching the logit*. LayerNorm on Q and K
makes the dot product depend on the angle between them and not on their
magnitudes, which — per exercise 3 — bounds the logit at ±d regardless of what
arrives. The norms still grow. They just stop being able to saturate anything.

### 5 — no generation ever puts text after the image

**ANSWER: from a text-only prompt, 100% of generations are text-first** — forced,
since the prompt is text. The measurable question is what comes *after*:

| over 1,000 draws | |
|---|---:|
| open an image | **1,000** |
| end at `</image>` | **1,000** |
| emit any text after the image | **0** |

**FINDING: `generate` stops at the first `</image>`.** Its loop breaks on
`SEP_CLOSE` whenever the output holds any text token — and a text prompt always
does. The 40% of the corpus that is image-then-text contributes its bigrams and
can never be reproduced, so the mixed-modality response this exercise asks for is
text-then-image and nothing else.

**FINDING: the corpus split is 37.9%, not 40%.** `make_dataset` flips
`random.random() < 0.4` per sequence. Over 1,000 draws the realised share is
**0.379**; at the lesson's own n = 40 the same coin has a standard error of
**7.7 points** — a fifth of the parameter being estimated.

**FINDING: generation length is bounded by the image, not by `max_len`.** Samples
run **7 to 25** tokens against a `max_len` of 30, and **0** reach it. The upper
end is set by how long the bigram wanders inside the image block, so the one knob
the caller has does not bind.
