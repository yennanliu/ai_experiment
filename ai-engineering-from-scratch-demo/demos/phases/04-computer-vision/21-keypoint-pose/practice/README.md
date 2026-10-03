<!-- generated:start -->
# 04-computer-vision / 21-keypoint-pose

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/21-keypoint-pose/) · upstream spec
`phases/04-computer-vision/21-keypoint-pose/docs/en.md`

```bash
uv run demo practice run 21-keypoint-pose --ex 1
uv run demo explain 21-keypoint-pose --ex 1
uv run pytest demos/phases/04-computer-vision/21-keypoint-pose
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Train the tiny keypoint model on the synthetic 4-point dataset. Report mean L2 error b… | code | T1 | `ex01_identical_markers_leave_keypoint_identity_at_chance.py` |
| 2 | (Medium) Add sub-pixel refinement: given the argmax position, fit a 1D parabola along x and y… | code | T1 | `ex02_lesson_refine_recovers_a_tenth_of_the_offset.py` |
| 3 | (Hard) Build a 2-person synthetic dataset where each image shows two instances of the 4-keypo… | code | T1 | `ex03_learned_paf_beats_proximity_only_where_proximity_fails.py` |
<!-- generated:end -->

## Answers

The lesson is 98 lines: a Gaussian heatmap target, a four-layer `TinyKeypointNet`,
an argmax decoder with a `subpixel_refine`, and a synthetic dataset of four black
squares. It imports torch at module scope, so every exercise is **T1** on the
`vision` extra and skips cleanly without it. Exercise 3 is built at the scale the
lesson's own net supports: 64x64 images, two 4-keypoint chains, 600 steps.

One fact decides exercises 1 and 2: `make_synthetic_sample` draws all four
keypoints as **the same 4x4 black square**, in random order. Nothing in the image
says which square is keypoint 0.

### 1 — identical markers leave keypoint identity at chance

**ANSWER: 12.458 px** mean L2 after 200 steps, read from the lesson's own
`main()` (seed 0, 8 images) — a fifth of the 64-pixel canvas.

**FINDING: the error is identity, not localisation.** Retrained the same way and
scored on 64 images:

| | ordered L2 | to *nearest* square | heatmap peak |
|---|---:|---:|---:|
| lesson's black squares | 14.59 px | **0.51 px** | **0.26** |
| uniform pick among the 4 squares | 17.03 px | — | — |
| keypoint k in its own colour (control) | **0.84 px** | — | 0.74 |

The net finds the squares to half a pixel; it cannot say whose they are, so
each channel learns the MSE-optimal hedge — a quarter-height bump on every
square.

**FINDING: `main()`'s `F.interpolate` is a no-op.** `TinyKeypointNet` already
returns 64x64 (two stride-2 convs, two stride-2 transposed convs); the "upsample
pred to full resolution" step changes the heatmaps by **0.0**.

**CONTROL:** colour-coding the keypoints, with the same net, seed and 200 steps,
gives **0.84 px**. The model was never the bottleneck.

### 2 — the lesson's refinement recovers a tenth of the offset

Scored on 200 of the lesson's own `gaussian_heatmap` targets at continuous
centres, where sub-pixel truth exists:

| decoder | height 1.0 | height 0.26 |
|---|---:|---:|
| integer argmax | 0.400 px | 0.400 px |
| lesson's `subpixel_refine` | 0.357 px | 0.389 px |
| **1D parabola (the exercise)** | **0.0117 px** | **0.0117 px** |
| parabola on log heatmap | 9e-08 px | 2e-07 px |

**ANSWER: 0.400 → 0.0117 px, a 34x gain.**

**FINDING: `subpixel_refine` is not a parabola.** `x + 0.25 * (r - l)` uses a
heatmap *value* difference as a pixel distance: it moves a 0.3 px offset
**10.9%** of the way (the parabola: 96%), and it shrinks with heatmap height —
at the 0.26 peak the lesson's trained net produces, it barely moves at all.

**FINDING: on the lesson's own data there is nothing to refine.** Keypoints come
from `rng.integers`, so argmax on the targets is exact (**0.0 px**); on the
trained 200-step pipeline the parabola moves 14.59 → 14.63 px, because the error
is which square, not where in it.

**CONTROL:** the log-parabola, exact for a Gaussian, lands within 2e-07 px.

### 3 — the learned PAF beats proximity only where proximity fails

Two colour-coded chains 0-1-2-3 (8 px limbs, 3 PAFs = 6 channels) on the
lesson's own `TinyKeypointNet(num_keypoints=10)` under its plain MSE, decoded
bottom-up: two peaks per type via `heatmap_to_coords`, then each limb's 2x2
pairing by PAF line integral. OKS uses s = 24 px, k = 0.1. Proximity grouping
(minus the distance) is scored on the same peaks as the baseline.

| 64 images | learned PAF | proximity | target PAF | proximity on targets |
|---|---:|---:|---:|---:|
| random layouts — grouped | 73.4% | **87.5%** | 100% | 92.2% |
| random layouts — OKS | 0.880 | 0.937 | 1.000 | 0.978 |
| antiparallel — grouped | **100%** | 0.0% | 100% | 0.0% |
| antiparallel — OKS | **0.963** | 0.489 | 1.000 | 0.500 |

**ANSWER: OKS 0.880 (random) and 0.963 (antiparallel).**

**FINDING: where proximity fails, the PAF is decisive.** When the second chain
runs back along the first 4 px to the side, the middle limb's wrong pairs are
4 px long and the right ones 8 px: proximity groups no image correctly, the
learned PAF every one.

**FINDING: everywhere else the learned PAF is the worse grouper.** On random
layouts it groups 73.4% against 87.5% for proximity. The field it learned has a
dot of only **0.41** with the true limb direction on limb pixels (1.0 in the
target), so close-lying chains let a wrong pair collect as much as the right one.
"Elegant and scales to arbitrary crowd sizes" holds for the target field; a
small net's estimate of it needs a proximity prior beside it.

**CONTROL:** the same decoder on the target fields groups 100% of both layouts —
the gap is in the learned field, not the decoding.
