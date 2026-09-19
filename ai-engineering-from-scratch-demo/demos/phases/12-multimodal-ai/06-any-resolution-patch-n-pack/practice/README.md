<!-- generated:start -->
# 12-multimodal-ai / 06-any-resolution-patch-n-pack

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/06-any-resolution-patch-n-pack/) · upstream spec
`phases/12-multimodal-ai/06-any-resolution-patch-n-pack/docs/en.md`

```bash
uv run demo practice run 06-any-resolution-patch-n-pack --ex 1
uv run demo explain 06-any-resolution-patch-n-pack --ex 1
uv run pytest demos/phases/12-multimodal-ai/06-any-resolution-patch-n-pack
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | A receipt is 600x1500 (1:2.5). At patch size 14, how many native-resolution tokens? How many… | code | T0 | `ex01_the_loss_is_anisotropic_and_the_receipt_is_landscape.py` |
| 2 | Build the block-diagonal mask for a batch of four images with lengths 256, 576, 729, 1024. Ve… | code | T0 | `ex02_six_point_seven_million_cells_to_express_five_integers.py` |
| 3 | For a 1792x896 image at patch 14, compare: (a) square-resize to 336 then encode, (b) AnyRes 2… | code | T0 | `ex03_the_thumbnail_is_what_breaks_the_square_law.py` |
| 4 | Implement fractional patch dropping: given a packed sequence, drop 50% of tokens uniformly at… | code | T0 | `ex04_dropping_half_the_tokens_leaves_the_sparsity_where_it_was.py` |
| 5 | Read Section 3.2 of the Qwen2-VL paper (arXiv:2409.12191). Describe in two sentences what `mi… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Four code solutions and one prose one. Three of the four find the same shape of
problem: the quantity the exercise asks for is not the quantity that decides the
outcome. Tokens are not what breaks OCR (exercise 1), sparsity is not what the
drop changes (exercise 4), and the token ratio only means something once the
thumbnail is removed (exercise 3).

### 1 — the loss is anisotropic, and the receipt is landscape

**ANSWER: 4,494 native against 576 square — a 7.80× reduction.** 600 and 1500
are not multiples of 14, so the lesson crops to 588 × 1498 first; the grid is
42 × 107.

**FINDING: the token ratio and the area ratio are the same number, and neither
is why OCR breaks.**

| | scale |
|---|---:|
| vertical, 336/588 | 0.5714 |
| horizontal, 336/1498 | 0.2243 |
| **anisotropy** | **2.548×** |

A glyph that was taller than wide comes out wider than tall. Square-resize loses
more OCR accuracy, and the mechanism is the distortion, not the budget — the
same 7.80× spent isotropically would cost far less accuracy than this.

**FINDING: the lesson's own receipt is landscape.** `Image(name, h, w)` takes
height first, so `Image("receipt 600x1500", 600, 1500)` is 600 tall and 1500
wide. The token count cannot notice — a grid product is symmetric — but
`anyres_cost` tiles it **1 × 2** as given and **2 × 1** with the sides swapped.
The one strategy that depends on orientation is reading it backwards.

**FINDING: the crop is not free either.** 600 → 588 discards **2.0%** of the
short side, 1500 → 1498 **0.13%** of the long one. On a receipt the short side
carries character width.

### 2 — 6.7 million cells to express five integers

**ANSWER: 2585 × 2585 = 6,682,225 cells, 1,977,329 non-zero — 29.59%.**
`pack_batch` returns both without materialising anything, and `build_dense_mask`
reproduces them on the lesson's own toy batch (100 of 196).

**FINDING: the same mask is five integers.** `cu_seqlens` is
`[0, 256, 832, 1561, 2585]` — exactly what FlashAttention's varlen kernels take.
The dense form is **1,336,445×** larger and carries nothing extra. The lesson
materialises 196 cells in its demo; this exercise asks for **34,092×** that.

**FINDING: the density is `(1 + CV²) / k`.**

| | density |
|---|---:|
| four equal blocks summing to 2,585 | **25.00%** |
| these four (CV = 0.4285) | **29.59%** |
| excess | **+18.4%** |

`1.1836 / 4` reproduces the measured figure to the digit.

**FINDING: so the batch sampler sets the attention bill, not the packer.**
Packing takes the density from 100% to 1/k; the *spread* of the lengths moves it
off 1/k. Both levers live in how the batch was assembled, and neither is in the
mask.

### 3 — the thumbnail is what breaks the square law

| strategy | tokens | effective resolution (h, w) |
|---|---:|---|
| (a) square-resize to 336 | **576** | (0.1875, 0.375) — **2× distorted** |
| (b) AnyRes 2 × 1 + thumbnail | 1,728 | (0.375, 0.375) |
| (c) M-RoPE native | **8,192** | (1.0, 1.0) |

**ANSWER: square-resize is fewest; native preserves most detail.** A **14.2×**
span, and the lesson's own scorer independently picks the 2 × 1 grid the
exercise names.

**FINDING: excluding the thumbnail, tokens are exactly the square of linear
resolution.** AnyRes's two tiles are 1,152 tokens against native's 8,192 —
**7.111×** — and each tile downscales a 896 × 896 region to 336, a linear ratio
of 2.6667 whose square is **7.111**. Exact. Add the 576-token thumbnail and the
ratio becomes 4.74 and means nothing.

**FINDING: square-resize is the only one of the three that distorts**, so the
three differ on two axes rather than one, and the cheapest is cheap partly by
changing the shape of the content.

### 4 — dropping half the tokens leaves the sparsity where it was

**ANSWER: the sparsity does not change.**

| | density |
|---|---:|
| before | 29.591% |
| after, mean of 200 uniform 50% drops | **29.609%** |
| range across seeds | 28.36% – 30.75% |
| after, exact per-image halving | 29.592% |

`Σn² / (Σn)²` is scale-free: halve every block and it cancels.

**ANSWER: what does change is the absolute cost, by 4×.** Attended cells fall
from **1,977,329** to **493,968** — a ratio of **4.003** — while the sequence
goes 2,585 → 1,292. The count is quadratic in a length that was halved, so
dropping half the tokens buys three quarters of the attention.

**FINDING: uniform and per-image dropping differ only in variance.** The
stratified version is free, exact, and removes a 2.38-point spread that comes
from short blocks being over- or under-sampled.

**FINDING: so sparsity is the wrong thing to measure here.** It is invariant to
the operation being performed. A report that the sparsity held at 29.6% is a
report that the drop was uniform, not that it was cheap.

### 5 — what `min_pixels` and `max_pixels` control

Drawing on **Token budgets**, which is where the lesson states the 2026
production rule — "pick a per-task max-pixels cap, encode at native aspect ratio
up to that cap, pack the batch, and skip padding" — and names these two knobs as
the interface to it.

Two sentences, as asked:

> `max_pixels` and `min_pixels` are a *ceiling and a floor on the image's pixel
> count*, applied by resizing the image — preserving its aspect ratio — before
> it is patched, so together they bound the visual token count per image without
> ever fixing its shape.
> Both bounds matter because they guard different failures: the ceiling is a
> cost bound, since attention within an image is quadratic in its own token
> count and one uncapped 4K screenshot will take the context and the latency
> budget with it; the floor is a quality bound, since a small image left at its
> own size produces so few patches that the encoder has nothing to attend over
> and, below one merged patch on a side, rounds to no tokens at all.

Why the knob is in **pixels** rather than in tokens is the part worth keeping.
Qwen2-VL's token count is `pixels / (patch² × merge²)` — `pixels / 784` at patch
14 with a 2 × 2 merge — so a pixel budget *is* a token budget, exactly, and it
happens to be one that does not name a width or a height. That is what makes it
compatible with native-resolution encoding: a 1:1 photo and a 1:2.5 receipt with
the same `max_pixels` get the same token count and keep their own shapes, which
is precisely what exercise 1 shows square-resize destroying.

The interaction with packing is the other half. Exercise 2 measures the
attention density as `(1 + CV²)/k`, where CV is the coefficient of variation of
the sequence lengths in the batch. `min_pixels` and `max_pixels` are what bound
that CV: the ratio `max_pixels / min_pixels` is the widest any two images in a
batch can differ, so tightening the pair does not only bound the per-image cost,
it flattens the block-length distribution and moves the packed mask's density
back toward 1/k. Setting `max_pixels` alone caps the worst case and leaves the
spread — and the spread is worth 18.4% of the attention bill in the batch
measured here.
