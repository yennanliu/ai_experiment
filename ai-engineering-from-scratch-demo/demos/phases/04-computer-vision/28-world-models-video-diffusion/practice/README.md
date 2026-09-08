<!-- generated:start -->
# 04-computer-vision / 28-world-models-video-diffusion

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/28-world-models-video-diffusion/) · upstream spec
`phases/04-computer-vision/28-world-models-video-diffusion/docs/en.md`

```bash
uv run demo practice run 28-world-models-video-diffusion --ex 1
uv run demo explain 28-world-models-video-diffusion --ex 1
uv run pytest demos/phases/04-computer-vision/28-world-models-video-diffusion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Compute the token count for a 5-second 360p video at patch-t=2, patch-h=8, patch-w=8.… | code | T1 | `ex01_token_count_memory_wall.py` |
| 2 | (Medium) Swap the divided attention block above for a full joint attention block and measure… | code | T1 | `ex02_joint_attention_costs_less.py` |
| 3 | (Hard) Build a minimal latent-action video model: take a dataset of (frame_t, action_t, frame… | code | T1 | `ex03_action_separates_frames.py` |
<!-- generated:end -->

## Answers

The three exercises are about arithmetic, a swap, and a demonstration — and each
one turns out to contain a claim its own lesson does not support.

### 1 — The token count, and the memory wall behind it

| reading of "5-second 360p" | tokens | joint attention | as float32 |
|---|---:|---:|---:|
| `main()`'s 480×360 @ 30 fps | **202,500** | 41,006,250,000 | **164.0 GB** |
| standard 16:9 640×360 | 270,000 (+33%) | 72,900,000,000 | **291.6 GB** (+78%) |

**ANSWER: 202,500 tokens, and one attention matrix is 164 GB** — a single matrix
in a single head of a single layer, before any weights or activations.

**FINDING: the exercise does not fix its own answer.** 360p is a *height*, and no
frame rate is named. The two readings differ 33% in tokens but **78%** in memory,
because the count is linear and the matrix is quadratic.

**MECHANISM: dividing the attention is what makes it payable** — 2,700·75² +
75·2,700² = **561,937,500** pairs against 41,006,250,000, **73× fewer**, or 2.2 GB
against 164.0 GB.

**CONTROL: the model in the same file does not use these patch sizes.**
`count_tokens` defaults to (2, 8, 8); `TinyVideoDiT` hardcodes **(2, 2, 2)**. The
same clip through the model's own patching is **3,240,000** tokens — 16× the
number the lesson prints. The token arithmetic and the network describe two
different systems.

**CONTROL: halving each side divides memory by 16.7, not 16** — because
`180 // 8` truncates 22.5 to 22 and loses half a patch row.

### 2 — Swapping divided attention for joint

| block | parameters | attention entries (5 s clip) |
|---|---:|---:|
| divided (lesson's own) | **66,752** | 561,937,500 · 2.2 GB |
| joint (the swap) | **49,984** | 41,006,250,000 · 164.0 GB |

**ANSWER: same output shape, and the joint block is 25% *smaller*.** The
difference is exactly one `nn.MultiheadAttention` (16,640) plus the third
`LayerNorm` that fed it (128).

**FINDING: so the necessity is not about parameters — divided *costs* them.**
Divided attention buys a smaller matrix by paying for a second set of
projections. The two blocks are different functions, differing by **1.4971** on
the same tokens, so this is a trade rather than a refactoring — and the usual
one-line explanation has it backwards.

**CONTROL: the lesson's own demo is far too small to show the problem.** At its
(4, 8, 8) grid the joint matrix is **65,536** entries (262 KB) and merely **3.8×**
the divided cost. The real clip is **625,706×** larger, so `main()` cannot exhibit
the constraint the exercise asks you to explain.

**CONTROL: the block normalises the same tensor three times per attention** —
`self.ln1(xt)` is passed separately as query, key and value, so two of the three
calls are wasted work in a block whose whole purpose is to be cheap.

### 3 — A demonstration that a model can pass without learning

| arm | final loss | separation across the 4 actions |
|---|---:|---:|
| conditioned, 300 steps | 0.01571 | **0.469** |
| action-blind, 300 steps | 0.01538 | **exactly 0.0** |
| conditioned, 3000 steps | 0.01538 | 0.284 |
| all-zero frame (trivial) | **0.015625** | — |
| ground truth | — | **2.828** |

**ANSWER: the four actions do produce different next frames** — 0.469 apart,
**17%** of the achievable 2.828, and exactly 0.0 with the action removed, so all
of it is attributable to the action input.

**FINDING: and it means nothing.** The dot is four lit pixels of 256, so
predicting an all-zero frame scores MSE **0.015625** — and *both* arms end within
0.0003 of it. Neither has learned the game, and the exercise's demonstration is
satisfied anyway. Nor is it undertraining: at **10× the steps** the loss is still
0.01538 and the separation *falls* to 0.284. Training drives both arms onto the
trivial predictor, so the visible difference is largest when the model knows
least.

**MECHANISM: 128 numbers carry the whole demonstration.** Zeroing the trained
embedding — nothing else of 17,412 parameters touched — drops the spread from
0.469 to exactly 0.0.

**MECHANISM: on a single frame the video block's time attention mixes nothing.**
`patch_t=1` gives grid (1, 8, 8), so time attention runs on sequences of length 1
— **64 self-pairs** against space's **4,096**. Half the block cannot move
information.

**CONTROL: `TinyVideoDiT` cannot be conditioned without editing it** — its
`forward` is declared `['self', 'x']`, with no action argument at all.

### A note on file lengths

The three files run 106 / 118 / 146 lines of code. The first two are inside
D14's 120-line target; `ex03` is over it and clear of the 150-line ceiling,
carrying three trained arms, a control model and a hand-built dataset.
