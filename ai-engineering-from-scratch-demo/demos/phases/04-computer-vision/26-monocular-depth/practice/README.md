<!-- generated:start -->
# 04-computer-vision / 26-monocular-depth

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/26-monocular-depth/) · upstream spec
`phases/04-computer-vision/26-monocular-depth/docs/en.md`

```bash
uv run demo practice run 26-monocular-depth --ex 1
uv run demo explain 26-monocular-depth --ex 1
uv run pytest demos/phases/04-computer-vision/26-monocular-depth
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Run Depth Anything V2 on any 10 images of your desk. Save depth as grayscale PNGs and… | code | T1 | `ex01_grayscale_png_erases_metric_scale.py` |
| 2 | (Medium) Given RGB + depth from Depth Anything V2, lift to a point cloud and render with `ope… | code | T1 | `ex02_scale_invisible_shift_bends.py` |
| 3 | (Hard) Take five pairs of images that differ only by a known object's position (e.g. bottle m… | code | T1 | `ex03_affine_alignment_invents_the_delta.py` |
<!-- generated:end -->

## Answers

All three exercises name a model that is not installable — `depth_anything_v2`,
`transformers`, `timm`, `open3d` and `unidepth` every one raise
`ModuleNotFoundError`, and nothing here downloads a weight. So the depth
predictor under test is the lesson's own: `main()`'s `pred = gt + 0.4 * randn`,
plus the four functions around it — `abs_rel_error`, `delta_accuracy`,
`align_scale_shift` and `depth_to_point_cloud`.

That turns out to be the more interesting object, because monocular depth is
dense with *exact* structure, and all three exercises walk into it. A
relative-depth prediction is ambiguous up to `a * d + b`. Each exercise asks a
question whose answer is determined by those two parameters rather than by the
prediction: exercise 1 saves the map through an encoding that quotients by them,
exercise 2 renders a lift on which one of them is invisible and the other is a
warp, and exercise 3 asks for centimetres from a pipeline that fits them to the
ground truth first.

These run at **T1** (they need torch and pillow), so they skip in the T0 CI gate
and run nightly.

### 1 — The grayscale PNG erases exactly what the exercise asks you to inspect

`(d - min) / (max - min)` is precisely the affine family `align_scale_shift`
solves for, so encoding a depth map as an 8-bit PNG quotients out scale and shift
and keeps everything else. Five metric rescalings of one scene:

| depth map | metric extent | PNG |
|---|---|---|
| `1.0·d + 0.0` | 1.00 – 4.96 m | **identical** |
| `3.0·d + 0.7` | 3.70 – 15.57 m | **identical** |
| `0.25·d + 10.0` | 10.25 – 11.24 m | **identical** |
| `12.0·d − 4.0` | 8.00 – 55.50 m | **identical** |
| `2.0·d + 0.0` | 2.00 – 9.92 m | **identical** |

**ANSWER: no object can be called wrong from the PNG.** Those five scenes encode
to **1** distinct 211-byte file. A 55-metre scene and a 5-metre scene are the
same bytes.

**MECHANISM: the invariance is exactly affine, and nothing wider.** Moving the
object instead — a non-affine edit — through 2.0, 1.5, 2.5, 3.4 and 1.0 m gives
**5 distinct PNGs of 5**. Scale and shift go; everything else stays, precisely.

**FINDING: the same gray level means a different distance in every file.** Pixel
value 128 decodes to:

| scene | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| depth at gray 128 (m) | 2.99 | 9.66 | 10.75 | **31.84** | 5.97 |

A **10.7×** spread. Ten PNGs of ten desk photos are ten incomparable scales.

**CONTROL: the lesson's own metrics are just as blind, so they cannot flag it.**

| metric | before PNG | after PNG |
|---|---:|---:|
| aligned absRel | 0.130487 | 0.130497 |
| raw absRel | 0.136754 | 0.136782 |

The only measurable cost is quantisation, **0.02485 m** per gray level over a
6.337 m span. Everything expensive was destroyed where no metric looks.

**ANSWER: there is no monocular cue here to fail.** `gt + 0.4 * randn` reads no
image, so its errors are Gaussian rather than perspective or size-constancy
failures, and its mean over the object is **2.0031 m** against a true 2.0 —
unbiased, which no real monocular predictor is. What does look wrong is noise:
the object covers **992** pixels at 2.0 m against a ramp averaging 3.396 m
there, so **0.50%** of its pixels still read farther than the ramp behind them.

### 2 — A scale error is invisible in the render; a shift is a warp

`depth_to_point_cloud` is a pinhole lift, and it is exact — reprojecting the
cloud through `main()`'s own `(96, 96, 48, 48)` intrinsics returns the pixel grid
and the depths to **0.0e+00**. Everything below is therefore a property of the
depth map and never of the lift.

**ANSWER: no render can grade the `a` in `a·d + b`.** Rescaling the depth map and
re-lifting keeps all **62,250** sampled pairwise distances in the same
proportion:

| rescale | max/min ratio of pairwise distances |
|---|---:|
| `a = 2.0` | **1.000000000000** |
| `a = 1.7` | 1.000001893069 |

`a = 2.0` is binary-exact so it holds to the last bit; `a = 1.7` drifts
**1.9e-06**, which is the `.astype(np.float32)` cast and not geometry — so the
identity is asserted at 1e-5, not tighter. The render is the old render with the
camera moved.

**MECHANISM: a shift is not a similarity**, because `x = (u − cx)·z / fx` carries
`z` as a factor, so adding a constant to `z` moves near points further than far
ones:

| shift | spread of the same distances |
|---|---:|
| `d + 1.0` | **2.0982×** |
| `2.0·d + 0.7` | 1.3806× |

`align_scale_shift` fits both parameters — so the metric removes the shift a
render could have caught and keeps the scale it could not.

**ANSWER: the outdoor scene looks more believable, by 8.9×.** On genuine ground
planes (built with `1/z` linear in image row), flatness is σ₃/σ₁ of the lifted
points:

| scene | untouched | `+0.5 m` shift | `×2` scale |
|---|---:|---:|---:|
| indoor, 1–5 m | 2.092e-08 | **0.03296** | 2.092e-08 |
| outdoor, 10–50 m | 1.987e-08 | **0.00371** | 1.987e-08 |

**CONTROL: nothing about indoor *content* is harder.** The scaled clouds stay
planar to 1e-8, so the ranking is the shift alone, and the damage is `b/z` — the
farther scene hides the identical error.

**FINDING: the lesson's own fixture is not a scene a pinhole camera could see.**
`synthetic_depth` makes depth linear in image row, which lifts to flatness
**0.188207** — five orders off the 2.1e-08 of a real plane, whose *disparity* is
what is linear in row. Its floor would bow in `open3d` whatever produced it.

### 3 — The affine fit invents the 30 cm

The only route from a relative prediction to metres is `align_scale_shift`, a
least-squares fit of `a·pred + b` against the ground-truth depth map — the
quantity the exercise is asking you to measure. Five pairs, each moved exactly
**30 cm** closer:

| base distance | depth-space fit | error | disparity-space fit |
|---|---:|---:|---:|
| 1.60 m | **90.9 cm** | +60.9 | 30.0 cm |
| 2.00 m | 59.8 cm | +29.8 | 30.0 cm |
| 2.40 m | 40.0 cm | +10.0 | 29.8 cm |
| 3.00 m | 25.2 cm | −4.8 | 30.1 cm |
| 4.00 m | **15.3 cm** | −14.7 | 29.9 cm |

**ANSWER: the same true 30.0 cm is reported as 90.9 cm and as 15.3 cm.** Worst
error **60.9 cm**. The answer decays with distance and crosses the truth between
2.4 and 3.0 m, so one pair reports it exactly — by luck.

**MECHANISM: the missing step is the reciprocal, not a bigger fit.** A relative
model predicts disparity, and `1/z` is not affine in `z`. Fitting the *same*
`align_scale_shift` against `1/gt` and inverting lands within **0.19 cm** on all
five, and drops absRel from **0.2461** to **0.00292**.

**FINDING: the prediction's own scale contributes nothing.** Replacing `pred`
with `c·pred + d`:

| rescale | reported delta |
|---|---:|
| `1.0·pred + 0.0` | 59.7592 cm |
| `5.0·pred − 3.0` | 59.7591 cm |
| `−1.0·pred + 7.0` (sign flip) | 59.7592 cm |
| `0.02·pred + 100.0` (ill-conditioned) | **1.6161 cm** |

The first three are identical to 1e-4 — the fit undoes any affine map by
construction, so a model reporting metres and one reporting an arbitrary index
score exactly the same.

**CONTROL: that identity is exact in real arithmetic and fragile in float32.**
The ill-conditioned rescale pushes the whole signal into the last mantissa bits
and returns 1.6161 cm. The invariance belongs to the least-squares solution, not
to the `torch.linalg.lstsq` computing it, so it is asserted at 1e-3 cm and no
tighter.

**FINDING: `delta < 1.25` scores 1.0000 on every pair.** It is a step function,
and a 30 cm move at 2.0 m is only a 1.1765 ratio. Sweeping the base distance at
the same fixed physical move:

| base | 1.40 m | 1.45 m | 1.50 m | 1.55 m | 1.60 m |
|---|---:|---:|---:|---:|---:|
| does the metric see the move? | yes | yes | yes | **no** | **no** |

It fires only where `base / (base − 0.3)` reaches 1.25, at exactly 1.50 m, and
where it does fire the drop is **0.0549 = 225/4096** — the object's pixel share
and nothing more.
The metric the lesson reports alongside absRel cannot see the quantity this
exercise is about, at any distance a bottle is normally photographed from.

### A note on file lengths

The three files run 120 / 120 / 120 lines of code, exactly at D14's target and
well clear of the 150-line ceiling. `ex02` and `ex03` import `status` — the
`importlib.import_module` → `ModuleNotFoundError` probe — from `ex01` via
`practice.load_module` rather than repeating it.
