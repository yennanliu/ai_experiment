<!-- generated:start -->
# 12-multimodal-ai / 01-vision-transformer-patch-tokens

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/01-vision-transformer-patch-tokens/) · upstream spec
`phases/12-multimodal-ai/01-vision-transformer-patch-tokens/docs/en.md`

```bash
uv run demo practice run 01-vision-transformer-patch-tokens --ex 1
uv run demo explain 01-vision-transformer-patch-tokens --ex 1
uv run pytest demos/phases/12-multimodal-ai/01-vision-transformer-patch-tokens
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compute the patch-token sequence length for Qwen2.5-VL at native 1280x720 input with patch si… | code | T0 | `ex01_both_dimensions_the_exercise_names_raise.py` |
| 2 | A 1080p frame (1920x1080) at patch 14 produces how many tokens? At 30 FPS over a 5-minute vid… | code | T0 | `ex02_nothing_that_keeps_the_frames_fits_a_128k_context.py` |
| 3 | Implement mean pooling over patch tokens in pure Python. Verify that mean-pool over 196 token… | code | T0 | `ex03_the_registers_move_the_mean_by_fifty_two_degrees.py` |
| 4 | Read Section 3 of "Vision Transformers Need Registers" (arXiv:2309.16588). Describe in two se… | explain | T0 | prose, below |
| 5 | Modify `code/main.py` to support patch-n'-pack: given a list of images of different resolutio… | code | T0 | `ex05_packing_without_the_mask_costs_more_than_padding.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 4 is prose, below, citing
the lesson section it rests on.

Three of the five exercises name a configuration the lesson's own `code/main.py`
cannot hold — `ViTConfig` has a single square `image_size`, and `grid_shape`
refuses any side that is not divisible by the patch size. That is not a defect
found by reading; it is what running the exercise produces, and it is the same
finding three times: **the patch-token primitive the lesson teaches is square,
divisible and fixed-length, and every 2026 use of it is none of those.**

### 1 — both dimensions the exercise names raise

| | tokens | floats |
|---|---:|---:|
| patch grid, 91 × 51 | **4,641** | 5,940,480 |
| CLS only | 1 | 1,280 |

**ANSWER: 4,641 against 1**, a ratio of 4,641 either way — both sides carry the
same hidden size, so the CLS vector is **0.02%** of what the grid holds.

**FINDING: the lesson cannot express the input the exercise names.**
`grid_shape` raises `ValueError` on 1280 *and* on 720 (1280 % 14 = 6,
720 % 14 = 6), and `ViTConfig.image_size` is a single square field. The number
above is floor division done by hand.

**FINDING: floor and ceil disagree by 143 tokens.** 91 × 51 cropped or 92 × 52
padded — 4,641 against 4,784, **3.1%** of sequence length and therefore of
attention cost. The exercise names neither convention.

**FINDING: Qwen2.5-VL's own answer is 1,125, not 4,641.** The 2 × 2 spatial
merge quarters the count the language model sees and needs both sides to be
multiples of 28, so 1280×720 is first cropped to **1260×700** — 20 pixels of
each dimension discarded — giving 4,500 patches and **1,125** tokens.

**FINDING: the lesson's own Qwen entry is a square 896 crop.** `ZOO[-1]` is
`Qwen2.5-VL ViT @ 896x896`, 64 × 64 patches plus a CLS token = **4,097**. The
entry named after the native-resolution model is the one configuration that
model does not use.

### 2 — nothing that keeps the frames fits a 128K context

**ANSWER: 10,549 tokens a frame, 94,941,000 for the five minutes.** 1920/14 and
1080/14 crop to a 137 × 77 grid; 30 FPS × 300 s is 9,000 frames.

| lever | tokens | saving |
|---|---:|---:|
| baseline, 30 FPS full grid | 94,941,000 | — |
| 2 × 2 token merging | 23,256,000 | 4.08× |
| frame sampling to 1 FPS | 3,164,700 | 30× |
| mean pool to one vector a frame | 9,000 | **10,549×** |

**ANSWER: pooling saves most, by 352× over its nearest rival, and it is the one
answer that cannot be used.** One vector per frame cannot say where anything is
in the frame, which is the whole of video VQA and grounding. The question
"which saves you most" is answered by the arm that deletes the task.

**FINDING: nothing that keeps the frames fits a 128K context.** Over the
10-point sampling × merging grid the cheapest point is 0.25 FPS merged —
**193,800** tokens, still **1.48×** over. Fitting needs **50** merged frames of
the 9,000: one every **6.0 seconds**.

**FINDING: all three ratios are constants, so duration cancels out.** Doubling
the video doubles every arm and the ranking never moves. The three differ in
what they destroy — within-frame layout, between-frame motion, fine detail —
and none of that is a token count.

**FINDING: one 1080p frame is 2.6× the lesson's largest configuration**, whose
4,097 tokens cover a whole 896 × 896 image.

### 3 — the registers move the mean by 52°

**ANSWER: 196 is not a DINOv2 count.** `grid_shape(224, 14)` is 16 × 16 =
**256** patches and `seq_length` is **261** with the CLS token and 4 registers.
196 is `grid_shape(224, 16)` — ViT-B/16. The exercise names one model's token
count and another model's patch size.

**MEASUREMENT: the pure-Python mean is exact to 1e-15.** Over 256 tokens of
1,536 dimensions a naive `sum(...) / n` agrees with an `math.fsum` reference
within 1e-15 on every dimension, and pooling 196 copies of one vector returns
it unchanged.

**FINDING: pooling the wrong 1.5% of the sequence moves the answer by 52°.**

| pooled over | cosine vs patch-only | norm |
|---|---:|---:|
| 256 patch tokens | 1.0000 | 2.455 |
| 256 patches + 4 registers | **0.6191** (51.8°) | 3.884 (**+58%**) |

The fixture is labelled and synthetic: 256 unit-scale Gaussian tokens plus 4
registers scaled to **10×** the median patch norm, which is the artifact
magnitude the lesson's register section describes. "Registers get discarded
before handoff" is not bookkeeping — it is the difference between two different
embeddings.

**FINDING: pooling is linear, which is why `forward` can hand it back for
free.** Pool-then-project and project-then-pool agree to **4.4e-16** across a
1,536 → 8 projection, so a pooled embedding costs one extra mean over tokens
the model already computed, not a second pass.

### 4 — what the registers absorb

Drawing on **CLS token, pooled output, and register tokens**, which is where
this lesson states the result Section 3 of arXiv:2309.16588 establishes.

Two sentences, as asked:

> The registers absorb the small set of **high-norm outlier patch tokens** that
> a large ViT develops during pretraining in redundant, low-information regions
> of its middle-to-late layers — tokens that have discarded the local patch
> content they are positioned over and are instead being reused by the model as
> scratch space for global, image-level computation.
> That matters for dense prediction because segmentation, depth and
> correspondence all read the per-patch feature map **position by position**, so
> an artifact patch is a position whose feature no longer describes its own
> patch; giving the model explicit registers to hold the global state leaves the
> patch tokens describing patches, which is what smooths the attention maps and
> lifts dense-probe quality.

The mechanism worth keeping is the trade the network makes on its own: it
needs somewhere to put global state, the architecture gives it nowhere, so it
takes the cheapest patches it has. Registers are not a denoising trick applied
to the output; they are the missing slot, added to the input. That is also why
they are **discarded before the LLM handoff** in a VLM — they were never
content — while the CLS token, which is the same idea used as an *output*, is
kept. The lesson's section makes the practical consequence explicit: DINOv2 and
SigLIP 2 both ship with registers, and the choice among CLS / mean pool /
registers is a choice about which tokens are asked to carry what.

### 5 — packing without the mask costs more than padding

A batch of four images at patch 14 — 224×224, 336×448, 630×392, 154×322:

| | tokens |
|---|---:|
| packed (patch-n'-pack) | **2,537** |
| padded to the largest, 4 × 1,260 | 5,040 |

**ANSWER: padding wastes 49.7% of the sequence**; packing wastes none.

**FINDING: the mask is not an optimisation of packing, it is the whole of it.**

| attention regime | query-key pairs | vs padded |
|---|---:|---:|
| packed, no mask | 6,436,369 | **+1.4%** |
| padded batch | 6,350,400 | — |
| packed, block-diagonal mask | **2,306,969** | **−63.7%** |

Packing alone is a *regression*: one 2,537-long sequence is a bigger square
than four 1,260-long ones. The block diagonal is **35.84%** of that square, and
every token of the saving comes from the mask.

**FINDING: the packed sequence has no positions to be embedded with.**
`pos_embed_params` is a table of exactly `seq_length(cfg)` rows, and 2,537
exceeds **four of the five** `ZOO` entries — 197, 261, 577, 733. Only Qwen's
4,097 rows (5,244,160 params) suffice, and even then rows are addressed by a
single index while this batch needs four independent 2D origins. This is why
patch-n'-pack ships with 2D-RoPE rather than a learned table.

**FINDING: the block structure is exactly the segment identity.** All 2,537
rows sum to their own image's token count, and the toy batch's materialised
11 × 11 mask matches `seg[i] == seg[j]` in all 121 cells, 49 of them `True`.
