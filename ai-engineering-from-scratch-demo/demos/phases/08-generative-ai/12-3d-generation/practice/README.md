<!-- generated:start -->
# 08-generative-ai / 12-3d-generation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/12-3d-generation/) · upstream spec
`phases/08-generative-ai/12-3d-generation/docs/en.md`

```bash
uv run demo practice run 12-3d-generation --ex 1
uv run demo explain 12-3d-generation --ex 1
uv run pytest demos/phases/08-generative-ai/12-3d-generation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` with 4, 16, 64 Gaussians. Report final MSE vs target. | code | T0 | `ex01_sixty_four_overflows_on_step_zero_from_a_stale_base_loss.py` |
| 2 | Medium. Extend to color Gaussians (RGB). Confirm reconstruction matches the target color patt… | code | T0 | `ex02_only_the_hue_matches_and_a_colourless_model_beats_the_fit.py` |
| 3 | Hard. Using gsplat or Nerfstudio, reconstruct a real object from a 50-photo capture. Report f… | code | T0 | `ex03_half_the_capture_is_mirror_images_so_held_out_is_not_held_out.py` |
<!-- generated:end -->

## Answers

The lesson's code is 106 lines: a 12x12 target, 2D isotropic splats that are
summed rather than alpha-composited, and a forward-difference optimiser,
`finite_diff_step`. All three exercises run that optimiser **unchanged**.
Exercises 2 and 3 swap only `render` and `mse` in its module namespace, because
its list-valued branch already differentiates an RGB colour or a 3D position.
Exercise 3 names gsplat/Nerfstudio and a real capture, so it ships the
scaled-down runnable that `DESIGN D11` requires.

Two facts about the lesson decide the answers:

- **The target is exactly two Gaussians.** Splats with sigma sqrt(3) and colour 1
  at (3, 3), and sigma 2 and colour 0.5 at (8, 8), reproduce it to MSE **9.8e-34**.
  So every residual is the optimiser's.
- **`finite_diff_step` has a stale base loss.** It updates each coordinate in
  place but measures every finite difference against a loss taken before any
  update.

### 1 — 64 Gaussians overflow on step zero, from a stale base loss

| n | 4 | 16 | 24 | 64 |
|---|---:|---:|---:|---:|
| final MSE, lesson step | 0.0575 | 0.0280 | 0.1173 | **OverflowError, step 0** |
| final MSE, corrected step | | 0.0099 | | 0.0601 after 5 steps |

**ANSWER: 0.0575 and 0.0280; 64 has no number.** `gaussian_value` overflows on
`sigma ** 2` during the first step, for the lesson's seed and for seeds 0-4.

**FINDING: the crash is the optimiser's bug, not its learning rate.** At n=64
the probe loss passes 100x the base by probe **29 of 257**. Each later
"gradient" then includes every earlier update's loss change. If every
difference is taken first and all updates are applied together, the same lr,
eps and init run 64 Gaussians without overflow. They also end n=16 **2.8x
lower**.

**FINDING: more Gaussians is not monotone.** n=24 ends 4x worse than n=16, on a
target that n=2 can fit exactly.

### 2 — only the hue matches; a colourless model beats the fit

**ANSWER: the pattern matches only in hue.** At n=8 the dominant channel is
right at **50/50** bright pixels, but the red blob renders `(0.24, 0.18, 0.12)`
against `(1, 0.5, 0)`. n=16 also scores 50/50 while overshooting to
`(1.00, 0.73, 0.48)`. Hue agreement cannot tell the two failures apart.

**FINDING: a grey model beats the RGB fit on MSE**: 0.0245 against 0.0398. A grey
model holds one value per splat, shared by all three channels, so it cannot
represent colour at all.

**FINDING: colour is the easy part.** Rendering is linear in colour, so with the
geometry known a 2x2 least-squares solve recovers the colours to **6e-17**. From
grey on the same true geometry, 30 lesson steps reach only
`(0.69, 0.46, 0.22)`.

### 3 — half the capture is mirror images, so "held out" is not held out

There is no gsplat, Nerfstudio or torch. Instead, 50 orthographic cameras ring
three isotropic 3D Gaussians, and each photo is the lesson's `render` of the
projected splats. Nine front-arc views are trained on.

**ANSWER: ~2 s and 14,400 view renders.** Held-out SSIM on the 16 untrained
front-arc views rises from 0.674 to **0.904**. The training views score 0.902.
The real run is printed:
`ns-train splatfacto`, followed by `ns-eval`. The lesson estimates 5-30 GPU
minutes per scene.

**FINDING: 25 of the 50 photos carry no new information.** A summing renderer
has no depth order. So the camera at theta + 180 sees the mirror image of the
camera at theta, to **6.7e-16**. A view opposite a training view scores exactly
the training SSIM, 0.902488. If views are held out by random index, part of the
"held-out" score is a training score.

**FINDING: the back side cannot be hallucinated.** The 25 back-arc views, none
of them trained on, score 0.903. The lesson's "back-side hallucination" pitfall
needs occlusion, and its renderer has none.
