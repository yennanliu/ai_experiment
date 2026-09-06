<!-- generated:start -->
# 04-computer-vision / 12-video-understanding

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/12-video-understanding/) · upstream spec
`phases/04-computer-vision/12-video-understanding/docs/en.md`

```bash
uv run demo practice run 12-video-understanding --ex 1
uv run demo explain 12-video-understanding --ex 1
uv run pytest demos/phases/04-computer-vision/12-video-understanding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Compute FLOPs (approximate) for FramePool with T=8 vs an I3D-style 3D ResNet with T=8.… | code | T1 | `ex01_inflated_flop_ratio.py` |
| 2 | (Medium) Generate a synthetic video dataset: random balls moving in random directions, labell… | code | T1 | `ex02_pooled_motion_ceiling.py` |
| 3 | (Hard) Build an R(2+1)D-18 by replacing every Conv2d in a ResNet-18 with `Conv2Plus1D`. Infla… | code | T1 | `ex03_r2plus1d_stride_repair.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` runs and does what it says. What the three exercises
find is that **each of them quotes a number, or asks for a step, that the lesson's
own code cannot produce**: the "3-5x" of exercise 1 is a single exact value,
the "near-chance" of exercise 2 is 2/3 rather than 1/3, and two of exercise 3's
three instructions are impossible against `Conv2Plus1D` as written.

Every model here is built with `weights=None` / `pretrained=False`. Nothing is
downloaded, and every run starts from random init — including the one place the
exercise asks for ImageNet weights. Shapes and FLOP counts are propagated on
torch's `meta` device, so the 43 GMAC of inflated convolution in exercise 1 is
counted but never executed.

### 1 — inflated FLOP ratio: it is exactly 3.0000, and a real 3D ResNet-18 is 5.61x

**ANSWER: the ratio is exactly the temporal kernel — 3.0000, at every resolution.**
Hooking every conv and linear in the lesson's own `FramePool`, then pushing each
traced `Conv2d` through the lesson's own `inflate_2d_to_3d`:

| | 64×64 | 224×224 |
|---|---:|---:|
| FramePool convolutions, T=8 | 1.184 GMAC | 14.508 GMAC |
| inflated, `time_kernel=3` | 3.553 GMAC | 43.525 GMAC |
| ratio | **3.0000** | **3.0000** |
| inflated, `time_kernel=5` — ratio | **5.0000** | **5.0000** |

(Double any of these for the mul+add FLOP convention; ratios are unaffected.)

**MECHANISM: `inflate_2d_to_3d` hardcodes temporal stride 1 and padding
`time_kernel // 2`**, so `T_out == T_in == 8` at all 20 convolutions and each
one's cost is multiplied by `time_kernel` and by nothing else. The ratio is
therefore independent of T, H, W, depth and channel width — "3-5x" is not a range
of outcomes, it is `time_kernel in {3, 4, 5}`. The lesson's own prose ("T/8 more
FLOPs … for temporal kernel of 3") would predict 1x at T=8, which is not what the
measurement says.

**FINDING: a real 3D ResNet-18 costs 5.61x, above the band.** `torchvision`'s
`r3d_18(weights=None)` measures 6.644 GMAC at 64×64 and 81.393 GMAC at 224×224 —
**5.6100x** and **5.6100x** over the same FramePool.

**MECHANISM: that extra factor is spatial, not temporal.** Decomposed at 64×64:

| r3d_18 variant | cost vs FramePool |
|---|---:|
| as shipped | 5.610x |
| + ResNet-18's stem `MaxPool` restored | **1.549x** |
| + its three stride-2 time steps un-strided | **2.979x** |
| pure inflation of ResNet-18 (for comparison) | 3.000x |

`r3d_18` has no max-pool after its stem, so `layer1` runs at four times the
spatial positions — a factor of **3.62** on its own, more than the whole time
axis contributes. Put the pool back and un-stride time and a real 3D ResNet-18
lands on 2.979x, back on the inflation's 3.000x.

**CONTROL: measured wall clock lands inside 3-5x, for a different reason.** One
224×224 T=8 clip, two timed forwards after a warm-up on two threads — on the
machine this was written on, FramePool **115 ms**, `r3d_18` **503 ms**, **4.36x**
against a MAC ratio of 5.61x; the file re-measures it on every run. That gap
is memory traffic and kernel efficiency, and it is machine-dependent — which is
why "approximate FLOPs" only answers this question loosely.

**CONTROL: capacity moves by the same factor.** FramePool holds 11,381,712
parameters, 11,166,912 of them in convolutions; inflation multiplies exactly that
by `time_kernel` to **33,500,736**, and `r3d_18` independently arrives at
**33,371,472**. One number — the temporal kernel — sets cost and capacity together.

### 2 — pooled motion ceiling: "near-chance" is 2/3, not 1/3

The dataset is the exercise's own three classes on a 16×16 canvas, T=4, 24 clips
per class for training and 24 more per class from a second seed for test. Each
`right-to-left` clip is generated as the **bit-exact time-reversal** of its
`left-to-right` partner, so the two classes are the same multiset of frames and
appearance cannot separate them even in principle.

**ANSWER: FramePool lands near 2/3, not near the 1/3 the exercise predicts.**
Eight epochs of Adam(lr=1e-3), batch 8:

| seed | test accuracy | chance | ceiling |
|---|---:|---:|---:|
| 0 | **0.625** | 0.333 | 0.667 |
| 1 | **0.667** | 0.333 | 0.667 |

**FINDING: the honest ceiling is 2/3, and both seeds land at or just under it.**
Per-class recall (`left-to-right` / `right-to-left` / `diagonal-up`):

| seed | L→R | R→L | diagonal-up | L→R + R→L |
|---|---:|---:|---:|---:|
| 0 | 0.917 | 0.083 | 0.875 | **1.000** |
| 1 | 0.542 | 0.458 | 1.000 | **1.000** |

Only two of the three classes are order-defined. `diagonal-up` visits a different
set of y positions, so a per-frame model reads it straight off appearance, while
the two horizontal classes must share one prediction — their recalls sum to at
most 1 by construction, which caps the three-class score at 0.667.

**MECHANISM: mean-pooling is permutation-invariant, so the shared row is exact.**
The confusion rows for `left-to-right` and `right-to-left` are element-wise
identical — `[22, 2, 0]` on seed 0, `[13, 11, 0]` on seed 1. A clip and its
reversal score within **1.9e-06 / 9.5e-07** of each other against logit scales of
**10.16 / 11.92**: float summation noise, not a decision. It is architectural
rather than learned — an *untrained* `FramePool(pretrained=False)` is already
**1.49e-08** apart at a logit scale of 0.082, a relative **1.8e-07**.

**CONTROL: the two classes are literally the same frames.** Their sorted pixel
multisets differ by exactly **0.0** over 73,728 values, so any leakage would have
to come from frame *order* — the one thing mean-pooling discards.

**CONTROL: the label is fully present in the data.** A zero-parameter read-out —
the sign of the intensity centroid's total displacement, `dy < -3.00` for
`diagonal-up`, else `dx > 0` — labels the same 72 test clips at **1.000**. The
task is not hard; it is invisible to the architecture, which is what the exercise
set out to show.

### 3 — R(2+1)D stride repair: two of the three instructions do not survive

**ANSWER: the repaired R(2+1)D-18 beats FramePool, and clears the ceiling
FramePool cannot.** Six epochs of Adam(lr=1e-3) on exercise 2's own 72 clips, both
arms on the identical budget, scored on 72 held-out clips:

| model | seed 0 | seed 1 |
|---|---:|---:|
| R(2+1)D-18 (repaired) | **0.972** | **1.000** |
| FramePool | 0.639 | — |
| FramePool's provable ceiling | 0.667 | 0.667 |

**FINDING: the exercise's inflation step cannot be done at all — offline or
online.** The lesson's `mid_c` formula makes `conv1` a `3 → 110 → 64` sandwich:

| tensor | shape | values |
|---|---|---:|
| ImageNet ResNet-18 `conv1` | (64, 3, 7, 7) | 9,408 |
| `Conv2Plus1D.spatial` | (110, 3, 1, 7, 7) | 16,170 |
| `Conv2Plus1D.temporal` | (64, 110, 7, 1, 1) | 49,280 |
| the lesson's `inflate_2d_to_3d` output | (64, 3, 3, 7, 7) | 28,224 |

Nothing fits anything. This is why the R(2+1)D paper trains from scratch, and it
is a stronger statement than "we are offline": no checkpoint would have helped.

**MECHANISM: taken literally the replacement deletes every stride — and still
runs.** `Conv2Plus1D` accepts only `(in_c, out_c, kernel_size)`, so a literal swap
drops the stride from all 7 strided convolutions, the 3 downsample 1×1s included
— which is exactly why the residual adds still line up and nothing raises:

| | map reaching `avgpool` | cost per clip |
|---|---|---:|
| literal replacement | (1, 512, 4, 8, 8) | 8.547 GMAC |
| repaired | (1, 512, 4, 1, 1) | **0.194 GMAC** |
| | | **43.9x** |

The repair stays inside the lesson's code: build `Conv2Plus1D` unchanged, then
re-set `spatial.stride`, which `nn.Conv3d` reads at call time.

**MECHANISM: the win is a temporal kernel that is not permutation-invariant.**
Reversing a test clip moves the trained R(2+1)D's logits by **11.19 / 8.71**
against scales of 7.65 / 5.41 — relative **1.46 / 1.61** — while the same reversal
moves FramePool's by **1.9e-06** at a scale of 13.62. The `k×1×1` temporal
convolution sees frame order; a mean over time cannot.

**CONTROL: (2+1)D buys no parameters back, and at Kinetics scale it is a GPU
job.** The converted network holds **33,217,329** parameters against ResNet-18's
11,178,051 — **2.972x**, not the "fewer parameters" the lesson's Key Terms table
claims. The `mid_c` formula is chosen to *match* a 3D convolution's budget, which
exercise 1 measures at exactly 3.0000x. At Kinetics settings (16 frames, 112×112)
the same network measures **25.0 GMAC** per clip, so 240,000 clips × 45 epochs × 3
for the backward pass is **1.62 EFLOP** — an estimated **9 GPU-hours** at an
assumed 50 TFLOP/s sustained. That is an optimistic floor: the real recipe,

```bash
torchrun --nproc_per_node=8 references/video_classification/train.py \
  --data-path <kinetics400> --model r2plus1d_18 --batch-size 16 --lr 0.64 \
  --epochs 45 --amp
```

is bound by video decoding, not by the model.
