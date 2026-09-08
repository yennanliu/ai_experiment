<!-- generated:start -->
# 04-computer-vision / 01-image-fundamentals

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/01-image-fundamentals/) · upstream spec
`phases/04-computer-vision/01-image-fundamentals/docs/en.md`

```bash
uv run demo practice run 01-image-fundamentals --ex 1
uv run demo explain 01-image-fundamentals --ex 1
uv run pytest demos/phases/04-computer-vision/01-image-fundamentals
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Create a 2x2 RGB `uint8` array with four distinct colors. Convert HWC to CHW and back,… | code | T0 | `ex01_hwc_chw_round_trip.py` |
| 2 | (Medium) Write `standardize(img, mean, std)` and its inverse that together pass a `roundtrip_… | code | T0 | `ex02_standardize_round_trip.py` |
| 3 | (Hard) Take a 3-channel ImageNet-standardized tensor and run it through a 1x1 conv that learn… | code | T1 | `ex03_one_by_one_conv.py` |
<!-- generated:end -->

## Answers

The three exercises look like warm-ups and each one is standing on a defect in
the lesson's own `code/main.py`:

| exercise | what the lesson's code does |
|---|---|
| 1 | `hwc_to_chw` is `transpose` — a **view**, so the round trip it asks you to prove cannot fail |
| 2 | `deprocess_imagenet` truncates instead of rounding — off by one on **18%** of uint8 values |
| 3 | `rgb_to_grayscale` truncates too, so the conv the exercise asks for **cannot** match it |

And exercise 1's English and Chinese texts ask for two entirely different pieces
of work.

**1 — the round trip preserves every value because nothing is ever copied.**

A four-colour 2×2 image goes `(2, 2, 3) → (3, 2, 2) → (2, 2, 3)`, all 12 values
equal.

**MECHANISM: it cannot fail.** `hwc_to_chw` is `arr.transpose(2, 0, 1)` — a
permutation of *strides*. The CHW array does not own its buffer (`OWNDATA` False),
its `.base` **is** the HWC input, and the round-tripped array still shares memory
with it. Nothing is copied, cast or rounded anywhere in the path, so there is no
arithmetic for the proof to catch.

**FINDING: the two arrays alias, so "convert" is the wrong word.** Writing 7 into
the CHW array's `[0, 0, 0]` changes the HWC original from **255 to 7**. A reader
who treats `hwc_to_chw` as a conversion and then edits the result has edited the
input too.

**FINDING: what the transpose does cost is contiguity** — `C_CONTIGUOUS` is True
for HWC and **False** for CHW. Every consumer that wants a flat buffer — a
framework tensor, `.tobytes()`, a memcpy to the GPU — pays for
`ascontiguousarray`, which *does* allocate. That copy, not the transpose, is the
cost of the layout change, and the exercise never mentions it.

**FINDING: the lesson asks two different questions in its two languages.**

| | exercise 1 asks for |
|---|---|
| `docs/en.md` | create a 2×2 RGB array, convert HWC↔CHW, prove the round trip |
| `docs/zh.md` | load a JPEG with **OpenCV and Pillow**, print both shapes and pixel (0,0), reconcile the channel order |

They share **0** words over three letters. Exercises 2 and 3 match; this one does
not, so the manifest carries both verbatim (D12) rather than picking one.

**2 — a correct pair round-trips exactly; the lesson's is off by one on 18% of values.**

Over a ramp carrying all 256 uint8 values:

| pair | max diff | samples off |
|---|---:|---:|
| `destandardize(standardize(x))` | **0** | 0 / 768 |
| `deprocess_imagenet(preprocess_imagenet(x))` | **1** | **137 / 768 (18%)** |

The exercise's bar is `roundtrip_max_diff <= 1`. The lesson's own pair passes it —
**and the bar is set exactly where its defect is.**

**MECHANISM: one truncating cast, not float32.** `deprocess_imagenet` ends in
`clip(x * 255.0, 0, 255).astype(uint8)`, which truncates: 0.9999 becomes 0. Round
the lesson's own CHW output instead and the error is **0 on every one of the 768
samples**. The precision was never the problem.

**MECHANISM: "the same call" is a broadcasting problem.** With `mean` of shape
`(3,)`, the naive `(x - mean) / std`:

| layout | result |
|---|---|
| HWC `(256, 3)` | returns `(1, 256, 3)` ✓ |
| NCHW `(2, 3, 8, 8)` | **ValueError** |
| NCHW `(2, 3, 8, 3)` | **silently returns `(2, 3, 8, 3)`** ✗ |

NumPy aligns from the right, so a real NCHW batch raises — and one whose width
happens to be 3 does not, returning a wrong answer with no complaint. Picking the
channel axis by shape fixes it: per-channel means agree exactly across both
layouts (0.0655, 0.1964, 0.4178).

**3 — the conv is the mixture, and it matches neither thing the exercise names.**

With weights `(0.299, 0.587, 0.114)` and `requires_grad_(False)` — 0 trainable
parameters:

| compared against | max difference |
|---|---:|
| the exact sum `img @ w` | **1.53e-05** (float32 epsilon at values up to 255) |
| the lesson's `rgb_to_grayscale` | **0.99998** |

**FINDING: `rgb_to_grayscale` truncates too.** It ends in `.astype(np.uint8)`, so
**775 of 1536** pixels come out one level low. "Within floating-point error" holds
against the float sum and **fails against the function the exercise names**.

**FINDING: on a *standardized* tensor these weights are not grayscale.**
Standardizing divides each channel by a different number, so the mixture the conv
then applies is effectively `w/std = [1.3057, 2.6205, 0.5067]` — no longer
proportional to the luma weights. Fitting the best *affine* function of true
grayscale to `conv(standardized)` still leaves **0.0183** of residual, so the two
are not even the same up to scale and offset. Standardize-then-mix and
mix-then-standardize commute only when `std` is equal across channels — and
ImageNet's is not.

**ANSWER: every *linear* colour transform is a 1×1 conv.** The 3×3 YCbCr matrix
loaded as three 1×1 filters reproduces the manual `tensordot` to **exactly 0.0**.
A 1×1 conv *is* a per-pixel matrix multiply, so RGB→YIQ, RGB→XYZ, channel swaps
and grayscale are one operation with different constants.

**CONTROL: HSV provably is not one.** A linear map obeys `f(a+b) = f(a)+f(b)`; the
lesson's own `rgb_to_hsv` misses that by **204.0** on one pair of colours. It takes
a max, a min and a divide by their difference, so no matrix — and therefore no 1×1
conv — can express it.
