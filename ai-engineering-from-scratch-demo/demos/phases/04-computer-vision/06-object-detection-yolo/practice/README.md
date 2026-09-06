<!-- generated:start -->
# 04-computer-vision / 06-object-detection-yolo

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/06-object-detection-yolo/) · upstream spec
`phases/04-computer-vision/06-object-detection-yolo/docs/en.md`

```bash
uv run demo practice run 06-object-detection-yolo --ex 1
uv run demo explain 06-object-detection-yolo --ex 1
uv run pytest demos/phases/04-computer-vision/06-object-detection-yolo
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Implement `box_iou` and run it against `torchvision.ops.box_iou` on 1,000 random box p… | code | T1 | `ex01_box_iou_oracle_parity.py` |
| 2 | (Medium) Port `yolo_loss` to a version that uses `CIoU` box loss instead of MSE. Show on a 10… | code | T1 | `ex02_ciou_vs_mse_box_loss.py` |
| 3 | (Hard) Implement multi-scale inference: feed the same image at three resolutions through the… | code | T1 | `ex03_multi_scale_inference_lift.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is correct — `box_iou`, `nms`, `encode`/`decode`,
`assign_targets`, `yolo_loss` and `postprocess` all do what they claim, and the
encode/decode round trip is exact. What the three exercises turn up is that **two
of the three ask you to confirm a conclusion rather than measure one**, and in
both cases the conclusion survives only under a parameter the exercise never
names: an epoch budget in exercise 2, a resolution ladder in exercise 3.

### 1 — box IoU against the named oracle, and three that are not it

`torchvision.ops.box_iou` is compared against whenever the `vision` extra
provides it (it is optional, and the check says which case it ran in). But it
cannot carry the exercise on its own: it evaluates the *same* algebra — edge
lengths multiplied into areas — so agreeing with it to `1e-6` mostly proves that
two spellings of one formula round the same way. Three oracles that share none of
that structure carry the weight.

| oracle | what it computes | pairs | max abs. difference |
|---|---|---:|---:|
| `torchvision.ops.box_iou` | the same algebra, someone else's code | 1,000 | **0.000e+00** |
| scalar Python loop | `min`/`max` on floats, no broadcasting | 1,000 | **0.000e+00** |
| rasteriser | counts unit cells in two boolean masks | 200 | **0.000e+00** |
| Monte-Carlo | 200,000 uniform points per pair | 50 | 1.46e-03 |

**ANSWER: max absolute difference is 0.0, sixteen orders below the `1e-6` the
exercise asks for.** Over 1,000 random `xyxy` pairs IoU spans 0.000 to 0.778 and
479 pairs actually overlap, so the comparison is not dominated by trivial zeros.
The rasteriser is the interesting one: it never multiplies two edge lengths at
all, and it still reproduces the lesson bit for bit on whole-pixel boxes — the
half-open `x2 - x1` convention is exactly the one that counts pixels.

**FINDING: the third oracle agrees only to its sampling error.** 200,000 points
per pair get within **1.46e-3**, three orders *above* the tolerance. Exactness in
the first two is a property of computing an area in closed form, not of computing
it correctly; a Monte-Carlo implementation of the same correct definition would
fail the exercise's own test.

**CONTROL: `1e-6` is not a floating-point test.** Re-running the same boxes at
4,000-px coordinates in **float32** and scoring against the float64 oracle gives
**1.51e-07** — inside `1e-6`, but only by 7x, where float64 clears it by every
digit it has (0.0). IoU is a ratio in [0, 1], so coordinate magnitude cancels and
only the storage dtype's ~1e-7 relative epsilon survives. The threshold is
calibrated to float32, not to the implementation.

**CONTROL: what `1e-6` does catch is the box convention.** Scoring the same 1,000
pairs under the historical PASCAL VOC `+1` width convention moves IoU by up to
**0.109** (mean **0.0099** across the 479 overlapping pairs) — 1e+05x the
tolerance. That is the class of error the exercise's test really discriminates.

**FINDING: `clip(union, 1e-8)` makes the function total.** A zero-area box against
itself, against a real box, and an inverted box (`x2 < x1`) all return `0.0`
rather than a `0/0` NaN, and the `1000x1000` pairwise call's diagonal matches the
1,000 single-pair calls to 0.0, so the `[:, None]` broadcast is not silently
transposing one argument.

### 2 — CIoU vs MSE box loss: a rate advantage, not a ceiling

The port is literal. `ref.yolo_loss(..., lambda_coord=0.0)` hands back the
lesson's own objectness and class terms untouched, and only the box term is
replaced, so the two arms differ in exactly one function. Both use the lesson's
`lambda = 5`; both train 30 epochs on the same 100 synthetic 128-px images
(8x8 grid, stride 16, three anchors, three classes) and are scored on a fresh
120-image held-out set, three seeds each.

| epoch | MSE mAP@0.5:0.95 | CIoU mAP@0.5:0.95 | lead | per-seed lifts |
|---:|---:|---:|---:|---|
| 10 | 0.081 | **0.233** | **+0.152** | +0.143, +0.173, +0.140 |
| 30 | 0.286 | 0.306 | +0.020 | −0.044, +0.065, +0.038 |

**ANSWER: at 10 epochs CIoU leads on every seed, by +0.152 mAP@0.5:0.95.** The
gain is concentrated at the loose threshold: AP@0.5 goes **0.230 → 0.721**
(+0.489, +0.477, +0.508 per seed) while AP@0.75 barely moves (+0.001, +0.073,
+0.020). CIoU gets boxes over the 0.5 line early; it does not make them tight.

**CONTROL: twenty more epochs mostly close it, so the exercise's claim is
budget-dependent.** From epoch 10 to 30 the MSE arm gains **+0.206** mAP while
CIoU gains **+0.073** — the MSE arm goes on improving, the CIoU arm has largely
plateaued. The lead falls from +0.152 to **+0.020**, and its per-seed values now
range over **0.110**, wider than the lead itself and wider than the MSE arm's own
seed spread of 0.047. "Show that CIoU converges to a better *final* mAP in the
same number of epochs" answers whichever way the epoch budget happens to fall;
what is actually measurable is a convergence *rate*, and it needs two checkpoints
to see.

**MECHANISM: the port swaps exactly the box term and nothing else.**
`lambda_coord=0.0` removes **5.5416** from the lesson's own loss against the
**5.5416** its own reported `parts["box"]` predicts (5.0 x MSE), so the shared
objectness and class terms are the lesson's, untouched. And CIoU of a box against
itself is **0.0** — overlap, centre distance and aspect ratio vanish together —
which is what makes a single shared `lambda` a fair comparison rather than a
rescaling of one arm.

### 3 — multi-scale inference: the lift is the ladder, not the method

"Run a single NMS at the end" is implementable without touching the lesson's
code: `ref.postprocess(..., iou_threshold=1.0)` makes its internal `nms` a no-op,
so each scale decodes with suppression off, the boxes are rescaled back to 128-px
coordinates, unioned, and one `ref.nms` runs over the union. One detector per
seed, three seeds, 120 held-out images.

| inference | mAP@0.5:0.95 | AP@0.5 | boxes kept | best-matched IoU |
|---|---:|---:|---:|---:|
| single scale, 128 | 0.286 | 0.751 | 137 | 0.630 |
| union 112/128/144 (±12.5%) | **0.326** | **0.868** | 213 | **0.702** |
| union 96/128/160 (±25%) | 0.249 | 0.719 | 286 | 0.691 |

**ANSWER: the ±12.5% union lifts AP@0.5 on every seed (+0.110, +0.116, +0.125)
and mAP@0.5:0.95 by +0.039 on the mean.** The mAP lift is +0.084, −0.000, +0.034
per seed — positive on the mean, but not on every seed, and that is reported as
what it is.

**CONTROL: the same construction at ±25% gives the loss back.** Widening the
ladder turns +0.039 mAP into **−0.037** (−0.041, −0.036, −0.034 per seed) and
+0.117 AP@0.5 into −0.032. The exercise says "measure the mAP lift", which
presumes there is one; the sign of the answer is decided by a parameter the
exercise never names.

**MECHANISM: the detector is not scale-equivariant, and anchors in absolute
pixels are why.** Read at 144 px and mapped back, the top box moves **7.80 px**
and its area becomes **0.843x** the native one — a scale-equivariant detector
would give 0 px and 1.000. Alone, no off-scale view is competitive:

| resolution | 96 | 112 | **128** | 144 | 160 |
|---|---:|---:|---:|---:|---:|
| mAP@0.5:0.95 alone | 0.124 | 0.258 | **0.286** | 0.265 | 0.171 |

**MECHANISM: the union does raise box quality even where it loses mAP.** Mean IoU
of each ground-truth box's best prediction goes 0.630 → 0.702 (near) → 0.691
(far): the far union really does contain better boxes. It also contains **286**
predictions against 137, and NMS cannot delete a false box that overlaps nothing.
The cost of a bad ladder is precision and ranking, not localisation.

**CONTROL: "a single NMS at the end" really is single.** `ref.nms` on four
overlapping boxes returns all **4** at `iou_threshold=1.0` (indices `[0, 3, 1, 2]`
— score order, nothing suppressed) against `[0, 3]` at 0.45. Every pair has
IoU ≤ 1, so the `ious <= iou_threshold` filter never drops anything: per-scale
suppression is off by construction, not by reimplementing the lesson's
`postprocess`.
