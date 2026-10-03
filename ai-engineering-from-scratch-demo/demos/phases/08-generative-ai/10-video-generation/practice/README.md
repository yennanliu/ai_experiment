<!-- generated:start -->
# 08-generative-ai / 10-video-generation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/10-video-generation/) · upstream spec
`phases/08-generative-ai/10-video-generation/docs/en.md`

```bash
uv run demo practice run 10-video-generation --ex 1
uv run demo explain 10-video-generation --ex 1
uv run pytest demos/phases/08-generative-ai/10-video-generation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. In `code/main.py`, compare frame-to-frame delta for (a) independent per-frame sampling,… | code | T0 | `ex01_joint_wins_but_is_still_half_again_rougher_than_the_data.py` |
| 2 | Medium. Add a first-frame condition: pin frame 0 to a given value and sample the rest. Measur… | code | T0 | `ex02_the_pin_fades_to_nothing_by_the_last_frame.py` |
| 3 | Hard. Use HuggingFace diffusers to run CogVideoX-2B on a local GPU. Time 20 inference steps a… | code | T0 | `ex03_full_3d_costs_about_one_factor_per_latent_frame.py` |
<!-- generated:end -->

## Answers

The lesson is a 1-D toy: six-frame "videos" `base + slope·t`, a three-layer
tanh MLP over the flattened frames plus sinusoidal positions, and a 40-step DDPM.
Exercises 1 and 2 retrain `main()`'s model exactly (seed 21, 3000 steps) and
sample it far more than `main()` does. Exercise 3 names CogVideoX-2B on a GPU,
which cannot run here, so it ships the scaled-down runnable `DESIGN D11` requires.

### 1 — joint sampling wins, but is still half again rougher than the data

| sampler (200 clips, 1000 deltas) | mean delta | variance |
|---|---:|---:|
| (a) `independent_per_frame` | 1.085 | 0.729 |
| (b) `sample_joint` | **0.367** | **0.113** |
| same model, frames shuffled across clips | 0.938 | 0.498 |
| the training data (`make_video`) | 0.243 | 0.034 |

**ANSWER: joint is 3.0x smaller in mean and 6.5x in variance.**

**FINDING: the baseline is not a sampler of this data.** `independent_per_frame`
is a hand-written `gauss(0,1) + 0.3t`; its frame means climb 0.05 → 1.43 while the
data's stay at zero. The fair baseline — the same model with the frames decoupled
— still gives 0.938, so the coupling is real.

**FINDING: joint is still 1.5x the data's roughness, and the wrong shape.** The
model's per-frame spread is U-shaped (0.97 → 0.69 → 1.11) where the data fans
out 0.92 → 1.75: it pinches clips in the middle instead of learning a slope.

**CONTROL:** the first five clips reproduce `main()`'s printed **0.61** — a
25-delta headline 1.7x the real mean — and the data's 0.243 matches the closed
form 0.246.

### 2 — the pin fades to nothing by the last frame

Frame 0 is pinned at sampling time inside `sample_joint`'s own reverse loop; the
propagation is the regression slope of E[frame k] on the pinned value. Every
`make_video` frame shares `base`, so the right answer is ~1.0 everywhere.

| frame | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| data | 1.00 | 1.01 | 1.02 | 1.03 | 1.03 | 1.04 |
| clean pin | 1.00 | 0.46 | 0.34 | 0.28 | 0.14 | **0.01** |
| noised (RePaint) pin | 1.00 | 0.47 | 0.33 | 0.27 | 0.12 | −0.01 |
| unpinned, lag on frame 0 | 1.00 | 0.75 | 0.55 | 0.37 | 0.07 | −0.10 |

**ANSWER: it propagates and decays to nothing** — frame 5 keeps 1% of a pin it
should keep all of.

**FINDING: the pin method is not the problem.** Both pins agree within 0.02, and
the unpinned model already forgets frame 0 by frame 4. The decay is in the model.

**FINDING: sampling starts far from pure noise.** `make_schedule(40)` ends at
alpha_bar = **0.667** — the chain begins where training saw 0.82·clip + 0.58·noise,
yet `sample_joint` hands it iid N(0,1). Each output frame keeps r = 0.37–0.76 with
its own independent starting draw: uncoupled noise no pin on frame 0 can reach.

### 3 — full 3D attention costs about one factor per latent frame

`diffusers`, `torch` and `transformers` are absent, so the profile is done by
counting over CogVideoX-2B's published shape (30 layers, width 1920, 8x/4x VAE,
2x2 patches, 226 text tokens, 49 frames = 6 s at 8 fps).

| | 480x720 (native) | 720p |
|---|---:|---:|
| tokens | 17,776 | 47,026 |
| attention share of layer FLOPs | 61% | **80%** |
| full / factorized | 12.88x | 12.95x |

**ANSWER: at 720p the attention core is the bottleneck** — 80% of FLOPs; 20 CFG
steps are 2.5e16 FLOPs (81 s at an A100's bf16 peak), and one head's score matrix
is 4.4 GB, 133 GB per layer, so only fused attention can run it.

**FINDING: the doc's "full 3D is 16–100x more expensive" does not hold here.**
Full over factorized is `Nt·Ns / (Nt + Ns)` ≈ Nt, the number of latent frames,
whatever the resolution. A 6 s clip has 13; 16x needs 17 (65 frames).

**FINDING: the lesson's "spatiotemporal DiT" has no attention at all** — three
dense layers (1824, 2304, 288 multiply-adds) over the flattened clip.

**CONTROL:** executed on a 4 x 6 grid with the lesson's `matmul`, full and
factorized attention compute 576 and 240 scores, 2.40 = 24/10 as predicted.
