<!-- generated:start -->
# 04-computer-vision / 13-3d-vision-nerf

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/13-3d-vision-nerf/) · upstream spec
`phases/04-computer-vision/13-3d-vision-nerf/docs/en.md`

```bash
uv run demo practice run 13-3d-vision-nerf --ex 1
uv run demo explain 13-3d-vision-nerf --ex 1
uv run pytest demos/phases/04-computer-vision/13-3d-vision-nerf
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Show that PointNet is permutation-invariant: run the same cloud through twice, once wi… | code | T1 | `ex01_pointnet_permutation_invariance.py` |
| 2 | (Medium) Implement a minimal ray-generation function that, given camera intrinsics and pose,… | code | T1 | `ex02_camera_ray_generation.py` |
| 3 | (Hard) Train a TinyNeRF on a synthetic dataset of rendered views of a coloured cube (generate… | code | T1 | `ex03_tinynerf_cube_training.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is correct — `PointNet`, `positional_encoding`,
`TinyNeRF` and `volumetric_render` all do what they claim. What the three
exercises find is that **each of them is asked for with a tolerance, a
convention or a statistic that does not fit what the code actually does**: a
float tolerance that is never spent, a ray generator whose handedness and
normalisation nobody wrote down, and a training loss that stops answering the
question the exercise attaches to it.

### Exercise 1 — pointnet permutation invariance

**ANSWER: shuffling changes no bit of any logit, so the allowed tolerance is
never spent.** Twenty clouds — four seeds × five point counts, three of them
deliberately not powers of two — through the lesson's own `PointNet` in
`.eval()`:

| | measured |
|---|---:|
| max abs logit difference, shuffled vs original | **0.00e+00** |
| clouds bitwise identical | **20 / 20** |
| per-point features, after re-indexing | 0.00e+00 |
| post-max-pool global feature | 0.00e+00 |

The lesson's own `permutation_invariance_check` prints the same `0.00e+00`.

**MECHANISM: the symmetry is a selection, not a cancellation.** The shared MLP is
a `Conv1d(kernel_size=1)`, whose reduction runs over *channels* only, so
permuting the point axis permutes GEMM columns without touching any reduction
order; `torch.max(x, dim=-1)` then picks an element rather than summing one. No
float cancellation is involved anywhere in the path, which is why a tolerance is
the wrong test to write.

**FINDING: max-pool buys far more than permutation invariance.** Only the points
that win at least one of the 1,024 pooled channels — the *critical set* — reach
the classifier at all:

| cloud size | 512 | 1,024 | 4,096 |
|---|---:|---:|---:|
| critical points | 114 | **144** | 157 |
| share of the cloud | 22.3% | **14.1%** | **3.8%** |
| logit difference, critical set only | 0.0 | 0.0 | 0.0 |

The set saturates rather than growing with the cloud. PointNet is invariant to
*deleting* most of its input, not merely to reordering it.

**CONTROL: the lesson's `.eval()` is load-bearing.** Left in the default
`train()` mode, two forwards of the *identical* cloud differ by **1.086**,
because of the head's `Dropout(0.3)` — a naive version of this test appears to
disprove invariance. Switch dropout off and a shuffled cloud is back to
`0.00e+00`: BatchNorm over `(batch, points)` is itself a symmetric statistic.

**CONTROL: the invariance is per-cloud only.** Score the same two clouds alone
and then alongside two unrelated clouds scaled 4×, in train mode, and they move
by **0.673** — train-mode BatchNorm couples a cloud to its batch mates while a
permutation still moves it by `0.00e+00`. A single cloud cannot be scored at all:
`ValueError: Expected more than 1 value per channel when training`.

**CONTROL: the lesson's own parameter count is 2.0× smaller than its docs claim.**
`docs/en.md` says "About 1.6M parameters"; the model `code/main.py` builds has
**807,626**, with **660,234** of them in the head alone (1024→512→256→10). The
claim is not a rounding of this architecture — the paper's PointNet, with its two
T-Nets, is the 1.6M model.

### Exercise 2 — camera ray generation

Neither `code/main.py` nor `docs/en.md` defines a ray generator — the docs only
write "Cast a ray from the camera through pixel (u, v)" as prose — so the
function is written here, in NeRF's OpenGL/blender frame:
`d_cam = ((u - cx)/fx, -(v - cy)/fy, -1)`.

**ANSWER: every one of the 768 rays reprojects onto its own pixel centre.** With
`K = [[40,0,16],[0,40,12],[0,0,1]]` and a look-at pose at radius 4.0, pushing each
ray out to `t ∈ {0.5, 2.0, 7.5}` and projecting back through the inverse pose:

| | measured |
|---|---:|
| max abs pixel error, all 768 rays × 3 depths | **1.9e-05 px** |
| spread of ray origins around the camera centre | **0.0** |
| max abs rotation error, `R d` vs a turned pose's `d'` | 1.8e-07 |

That last row is **MECHANISM: the pose acts only on directions.** Ray generation
is one fixed camera-space grid that the pose then rigidly moves — the whole
function is 24×32 arithmetic and a single 3×3 matmul.

**FINDING: NeRF's own `get_rays` is half a pixel off, in both axes.** The original
implementation writes `(i - W*.5)`, which indexes pixel *corners*. Substituting
`offset=0.0` for the `0.5` used above shifts every reprojection by
**[-0.500006, -0.499996] px** in x and the same in y — a convention, not noise. It
is invisible inside a self-consistent train-and-render loop and shows up the
moment the poses come from a calibrated SfM pipeline.

**MECHANISM: directions are deliberately not unit.** With `d_z` pinned at -1 the
norm is exactly `1/cos(theta)`:

| | measured | closed form |
|---|---:|---:|
| min \|d\| (optical axis) | 1.000156 | 1 |
| max \|d\| (corner pixel *centre*) | **1.110321** | 1.118034 at the true corner |

`sqrt(1 + (W/2f)² + (H/2f)²) = 1.118034`; the measured maximum is smaller because
the corner pixel's centre sits half a pixel inside the frame corner. Field of view
43.6° × 33.4°.

**FINDING: normalising the directions silently changes what the lesson's own
renderer reports.** One fronto-parallel wall at camera depth 4.0, 128 samples over
`[2.0, 6.0]`, through `volumetric_render`:

| directions | depth min | depth max | depth std |
|---|---:|---:|---:|
| un-normalised (`d_z = -1`) | 3.92031 | 3.92031 | **4.8e-07** |
| normalised | 3.92092 | **4.34941** | — |

Un-normalised, `t` *is* depth along -z, so a flat wall is flat. Normalised, `t` is
Euclidean range, so the same flat wall bows across the frame.

**CONTROL: the two depth maps differ by exactly the direction norm.**
`max |depth_range/depth_z − |d||` over all 768 pixels is **8.7e-04** — the same
`1/cos(theta)` closed form — at the 0.0315 sample spacing this quadrature runs at.
Both renders carry the same weight mass (min 0.9998), so nothing else moved. Both
depths read below 4.0 because alpha compositing is front-biased across a slab of
finite thickness: the near face absorbs first.

### Exercise 3 — tinynerf cube training

The scene is a coloured cube ray-traced analytically — slab-method ray/box
intersection, six coloured faces on a white ground — at 12×12 from 24 poses on a
ring. Nothing is downloaded; there is no lego scene to fetch. Everything
downstream is the lesson's own `TinyNeRF` and `volumetric_render` at their
defaults, Adam at lr 5e-3, one shared 32-sample grid for every ray.

**ANSWER: the three requested losses, and a definition of "recognisable".** The
question has no answer until "recognisable" is a number, so it is pinned to a null
model: the best single colour for the scene scores **9.98 dB** on three held-out
poses at the midpoints of the training ring, and a render counts as recognisable
once it clears that by 3 dB, i.e. **12.98 dB**.

| epoch | 1 | 10 | 20 | 35 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| train MSE (24 views) | **0.20065** | **0.04939** | — | — | — | **0.00165** |
| novel-view PSNR (dB) | 10.22 | **13.01** | 15.57 | 16.02 | 15.60 | 15.65 |

Novel views first clear 12.98 dB at **epoch 10**, with 0.03 dB to spare.

**FINDING: the statistic the exercise asks for stops tracking the question it
asks.** From epoch 20 to 100 the training loss the exercise wants reported drops
**6.4×**, while novel-view PSNR moves **+0.08 dB** and the training-camera PSNR
that loss implies, `-10·log10(MSE)`, climbs **19.8 → 27.8 dB**. The best novel view
is at epoch 35, not 100. Past epoch 20 the reported number is measuring
memorisation of the rays the model was shown.

**CONTROL: at 8 views no epoch is ever the answer.** The same model and the same
schedule on 8 views instead of 24:

| | 24 views | 8 views |
|---|---:|---:|
| train MSE at epoch 100 | 0.00165 | **0.00017** |
| training-camera PSNR | 27.8 dB | **37.7 dB** |
| novel-view PSNR at epoch 100 | 15.65 dB | **10.28 dB** |
| its own constant-colour baseline | 9.98 dB | 9.97 dB |

The 8-view arm reconstructs its own cameras 10× better than the 24-view arm does,
and its novel views (9.09 / 11.53 / 11.80 / 10.77 / 10.61 / 10.28 dB at epochs
1/10/20/35/50/100) never clear 9.97 + 3 dB at any epoch. **The binding variable is
view count, not epoch.**

**MECHANISM: the quadrature is exactly `1 − prod(1 − alpha)`, so a zero density
renders pure black.** The transmittance product telescopes at *every* prefix, not
merely in total: over 64 random rays,
`max_k |cumsum(w)_k − (1 − cumprod(1−alpha)_k)|` is **6.0e-08**, float32 round-off
on a quantity of order 1. The `1e10` final delta then pins the total at
**1.000000** whenever `sigma > 0` — a partition of unity with **no background
term**. So wherever `torch.relu(self.sigma(h))` is 0 everywhere along a ray the
render is black, and ReLU's zero derivative closes the only path back:
**5 of 10 seeds** start at a total gradient norm of exactly **0.0** on an all-zero
image, pinned forever at the target's mean square **0.6912**. Half of this
lesson's initialisations cannot be trained at all.

**MECHANISM: 6 of the 10 encoding bands sit above the Nyquist limit for the spacing this
render samples.** 32 samples over `[1.8, 4.3]` is a step of **0.08065**, so band
`l` advances `2^l·pi·0.0806` rad per step:

| band l | 0–3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| phase per sample step | < pi | > pi | > pi | > pi | > pi | > pi | **20.6 cycles** |

Two points one step apart encode to a dot product of **-0.98894**, matching the
closed form `3·Σ_l cos(2^l·pi·d) = -0.98894` to **1.3e-06** — against **30** for
identical points. The renderer never samples the top six bands consistently, so
whatever the MLP puts there is noise to the integrator.
