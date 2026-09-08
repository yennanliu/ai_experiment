<!-- generated:start -->
# 04-computer-vision / 22-3d-gaussian-splatting

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/22-3d-gaussian-splatting/) · upstream spec
`phases/04-computer-vision/22-3d-gaussian-splatting/docs/en.md`

```bash
uv run demo practice run 22-3d-gaussian-splatting --ex 1
uv run demo explain 22-3d-gaussian-splatting --ex 1
uv run pytest demos/phases/04-computer-vision/22-3d-gaussian-splatting
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Run the 2D splat trainer above on a different synthetic image. Vary `num_splats` in `[… | code | T1 | `ex01_splat_count_diminishing_returns.py` |
| 2 | (Medium) Extend the 2D rasteriser to support per-Gaussian RGB colours that depend on a scalar… | code | T1 | `ex02_view_angle_harmonic_rank.py` |
| 3 | (Hard) Clone `nerfstudio` and train `splatfacto` on a 20-photo capture of any scene you have… | code | T1 | `ex03_absent_stack_render_invariants.py` |
<!-- generated:end -->

## Answers

Splatting is unusually rich in exact identities, and all three exercises here
turn out to be about the difference between an identity and a measurement. Two
of the three ask for a comparison whose winner is decided by the axis you plot
it against; the third asks for something no machine here can run, so what is
left is the arithmetic underneath it — asserted at float32 epsilon, never at a
fitted threshold.

### 1 — Where the returns actually diminish

Best MSE over 300 steps, two seeds, on a 40×40 target of yellow diagonal bands,
a blue ring and a dark-green ground:

| `num_splats` | best MSE (seed 0 / seed 1) | final MSE | 300 steps (this machine) | pixels below weight 0.5 |
|---|---:|---:|---:|---:|
| 16 | 0.04182 / 0.03477 | 0.04182 / 0.03493 | 0.47 s | 24.5% |
| 64 | 0.00717 / 0.00775 | 0.00728 / **nan** | 2.08 s | 37.4% |
| 256 | **0.00098 / 0.00067** | 0.00276 / 0.00330 | 13.15 s | 34.7% |

**ANSWER: on the axis the exercise asks you to plot, there is no knee.** Every
4× in `num_splats` divides best MSE by **5.8× then 7.3×** (seed 0) and **4.5×
then 11.5×** (seed 1). The returns *accelerate* across `[16, 64, 256]`, so
"identify the point of diminishing returns" has no answer on an MSE-vs-step
plot.

**FINDING: the knee is on the compute axis, near 12,800 splat-steps.** A step
is not a unit of work — `rasterise_2d` composites in a sequential Python `for`
loop, so one 256-splat step costs about 25× one 16-splat step. Re-plotted
against `count × steps`, a number no clock enters, the arms cross:

| budget (splat-steps) | 16 splats | 64 splats | 256 splats |
|---|---:|---:|---:|
| 1,600 | **0.04782 / 0.05168** | 0.05069 / **0.04778** | 0.11710 / 0.11210 |
| 4,800 | 0.04182 / 0.03477 | **0.01559 / 0.01378** | 0.04115 / 0.03872 |
| 12,800 | — | 0.00851 / **0.00874** | **0.00858** / 0.00745 |
| 19,200 | — | 0.00717 / 0.00775 | **0.00413 / 0.00366** |

256 splats is the *worst* arm below ~12,800 and the best above it, on both
seeds. That crossing is the point the exercise is looking for.

**FINDING: half the runs never reach step 300 in a usable state.** Three of six
survive, where surviving means finite and within 1.5× of the run's own best.
One 64-splat run ends `nan`; both 256-splat runs end at 0.00276 and 0.00330
against bests of 0.00098 and 0.00067. Reading the *final* MSE off the requested
plot therefore ranks the arms differently from reading the best — and neither
256-splat run survives.

**MECHANISM: the cost is linear in splats and cannot vectorise.** 0.47 s → 2.08
s → 13.15 s for 300 steps at 16 → 64 → 256 splats. The compositing loop
`for i in range(means.size(0))` needs the transmittance the previous iteration
left behind, so there is nothing to parallelise over.

**CONTROL: the renderer has no background, so a third of the frame is floored
at black.** `rasterise_2d` accumulates into `torch.zeros(H, W, 3)`, so a pixel
is only as bright as the weight it collects — and 24.5%, 37.4% and 34.7% of
pixels stay below half a unit of weight at 16, 64 and 256 splats. Sixteen times
more splats does not buy coverage. For scale, an all-black image scores 0.273
on this target and the best constant image 0.126, so 16 splats at 0.04182 beat
a flat colour by only 3.0×.

### 2 — A "degree-2 harmonic" in a scalar angle is five-dimensional, not nine

A scalar view angle is not a direction on the sphere, so the faithful reading
maps it to a great circle, `dir = (cos t, sin t, 0)`, and evaluates the lesson's
own `sh_degree_3_basis` there. Singular values of its first nine columns over
256 angles:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| σ | 6.7703 | 6.1804 | 6.1804 | 5.5279 | 5.5279 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**MECHANISM: rank exactly 5 of 9, and the four lost dimensions are named.**
Columns 2, 5 and 7 have peak magnitude **0.0e+00** — each carries a `z` factor,
and `z = 0` on the circle. Column 6 is exactly `-sqrt(5)/2` times column 0,
residual **6.0e-08**, one float32 eps: the tolerance an algebraic identity
earns, not a fitted one. So "a degree-2 harmonic in a scalar angle" is the
five-term Fourier basis 1, cos t, sin t, cos 2t, sin 2t wearing a spherical
name. All 16 degree-3 columns span only **7**.

Three arms trained on a pair of targets at t = 0 and t = π/2, geometry and
pixels from the lesson's own `Splats2D` and `rasterise_2d`, colours from its own
`eval_sh_degree_3`:

| arm | SH slots | MSE at t = 0 | MSE at t = π/2 |
|---|---:|---:|---:|
| degree 0 | 1 | 0.05534 | 0.05544 |
| degree ≤ 1 | 4 | 0.00178 | 0.00187 |
| degree ≤ 2 | 9 | **0.00167** | **0.00176** |

**ANSWER: the degree-2 model reconstructs both views.** The two targets differ
from each other by **0.21714**, so the fit sits **123×** below the thing it has
to distinguish and **31×** below the view-independent arm.

**CONTROL: a view-independent colour lands exactly on the mean image.** The
degree-0 arm scores 0.05534/0.05544 against the closed-form floor for *any*
model that renders one image for both views — the mean of the pair, at
**0.05429**. It is within **1.02×** of that bound, so it is not under-trained:
one colour per Gaussian provably cannot fit two different targets. That is the
control "verify it reconstructs both" leaves out.

**FINDING: three of the nine coefficient slots collect bitwise-zero gradient.**
Summed |grad| per slot over 200 steps:

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Σ\|grad\| | 0.212 | 0.412 | **0** | 0.405 | 5.6e-17 | **0** | 0.237 | **0** | 0.751 |

Slots 2, 5 and 7 are structurally dead on the circle. Slot 4 (`sin 2t`) adds
only **5.6e-17** — float noise, because both sampled angles have `sin 2t = 0`.
That is **480 of 1080** SH parameters that never move, three slots' worth of
them for any choice of angles at all.

**FINDING: reconstructing the pair does not rank the harmonic orders — an
unseen angle does.** Degree 1 fits the pair as well as degree 2, a ratio of
**1.06×**: two views can determine at most two of the five live dimensions, so
the exercise's own success criterion saturates. Rendered at the never-trained
angle π/4 the two arms disagree by max **0.163** per channel, MSE **0.00125** —
comparable to their own training error, and the only place the extra five
coefficients are visible.

### 3 — The stack is unreachable, so measure the arithmetic under it

**ANSWER: no training time, Gaussian count or fps can be reported.**

| package | import |
|---|---|
| `nerfstudio` | `ModuleNotFoundError: No module named 'nerfstudio'` |
| `gsplat` | `ModuleNotFoundError: No module named 'gsplat'` |
| `pycolmap` | `ModuleNotFoundError: No module named 'pycolmap'` |

Nothing may be downloaded, and there is no capture to pose. What is measurable
is the lesson's own rasteriser: 48 splats at 48×48 in **1.83 ms** on this machine — **548 fps**,
**60.6M splat-pixels/s**. At that rate the 6,000,000 splats the lesson says an
RTX 3080 Ti renders at 147 fps would need **57 hours** per 1080p frame here. The
10–30 minute `splatfacto` run and that fps figure are reported by the lesson,
not measured.

**MECHANISM: the composite telescopes, and its transmittance ignores order.**

| identity | measured | tolerance |
|---|---:|---:|
| max\|Σᵢ Tᵢaᵢ − (1 − T_final)\| | 3.9e-07 | 1e-06 |
| max\|T_final − Πᵢ(1 − aᵢ)\| | 2.4e-07 | 1e-06 |
| reverse compositing order: transmittance | 3.0e-07 | 1e-06 |
| reverse compositing order: image | **0.103** | — |
| reverse *input* order: image | **0.0e+00** | exact |

The tolerance is 1e-6 because float32 eps is 1.2e-7 and a pixel accumulates 48
terms; nothing here is fitted. Transmittance is a product and multiplication
commutes, so compositing order cannot move it — but it moves the image by 0.103
per channel. Handing `rasterise_2d` the same splats reversed changes no bit,
because `argsort(depths)` re-sorts them first.

**FINDING: `Splats2D.depth` cannot learn, and it is a tenth of the model.** One
backward pass of the lesson's own forward against its own `make_target`:

| parameter | max\|grad\| | count |
|---|---:|---:|
| `means` | 1.0e-03 | 96 |
| `log_scale` | 9.5e-03 | 96 |
| `rot` | **0.0e+00** | 48 |
| `colour_logits` | 1.7e-03 | 144 |
| `opacity_logit` | 4.7e-03 | 48 |
| `depth` | **None** | 48 |

`depth.grad` is not zero but *absent* — no graph path at all, because it reaches
the loss only through `torch.argsort`, which returns integers. An `Adam.step()`
then moves it by exactly **0.0**. Those **48 of 480** parameters (10%) are
handed to the optimiser and can never update. `rot` reads 0.0 for a milder,
recoverable reason: the initialiser makes every Gaussian isotropic, and rotating
a circle changes nothing.

**MECHANISM: `covs()` is `R S Sᵀ Rᵀ` to under one ulp — until float32 loses the
definiteness.** On entries reaching 320, max|Σ − Σᵀ| = **1.5e-05** and the
eigenvalues match `exp(2·log_scale)` to **3.1e-05** — **5e-08** and **1e-07**
relative, inside float32's 1.2e-7 ulp. So the output really is a rotation
conjugating `diag(s²)`, checked on the matrix rather than by rebuilding `R` from
the same cosines the lesson used. Yet `det(Σ)`, which that same algebra fixes at
exactly 1.0 for *every* anisotropy, comes back **0** at condition number
**5e+08**, where `Σ @ inv(Σ)` misses the identity by **10**. That is the
mechanism behind exercise 1's NaN: the clamp to 0.99 hides the overflowing
density in the forward pass, so only the backward pass reports it.

**CONTROL: degree-0 harmonics are view-independent to the bit.** With only the
degree-0 coefficient nonzero, `eval_sh_degree_3` over 500 random unit directions
spreads exactly **0.0** — every direction returns the same bits, because that
basis function is the constant C0 = 0.2821. Filling all 16 slots spreads
**7.84** over the same directions, so the measurement is not vacuous:
view-dependence is on offer, and degree 0 declines it.
