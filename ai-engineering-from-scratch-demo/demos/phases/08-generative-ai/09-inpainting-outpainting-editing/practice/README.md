<!-- generated:start -->
# 08-generative-ai / 09-inpainting-outpainting-editing

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/09-inpainting-outpainting-editing/) · upstream spec
`phases/08-generative-ai/09-inpainting-outpainting-editing/docs/en.md`

```bash
uv run demo practice run 09-inpainting-outpainting-editing --ex 1
uv run demo explain 09-inpainting-outpainting-editing --ex 1
uv run pytest demos/phases/08-generative-ai/09-inpainting-outpainting-editing
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. In `code/main.py`, vary the fraction of dimensions masked from 0.2 to 0.8. At what frac… | code | T0 | `ex01_never_equal_one_pinned_dim_still_names_the_cluster.py` |
| 2 | Medium. Implement RePaint: at every 10th reverse step, jump back 5 steps (add noise) and re-d… | code | T0 | `ex02_repaint_helps_but_there_is_no_edge_to_heal.py` |
| 3 | Hard. Use Hugging Face diffusers to compare: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-F… | code | T0 | `ex03_pose_is_free_and_the_fill_model_is_not_better.py` |
<!-- generated:end -->

## Answers

The lesson is a 5-D DDPM trained on a two-cluster mixture, with an `inpaint`
that reinjects a forward-noised copy of the pinned dims at every reverse step and
pastes them back at the end. Every solution retrains the model exactly as
`main()` does (seed 5, 5000 steps, ~1.5 s). Exercises 1 and 2 are **T0**;
exercise 3 names two multi-gigabyte pipelines that cannot be loaded here, so it
ships the scaled-down runnable `DESIGN D11` requires.

One fact about the data decides most of what follows: `sample_data` draws every
dim **independently** around the same cluster centre (±1, noise 0.2). One pinned
dim identifies the cluster almost surely, and no dim is a neighbour of another.

### 1 — never equal: one pinned dim still names the cluster

With `d = 5`, the fractions 0.2–0.8 are exactly 1–4 masked dims.

| masked | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|
| inpaint residual | 0.339 | 0.365 | 0.438 | 0.640 | 1.173 |
| unconditional residual | 1.187 | 1.164 | 1.134 | 1.119 | 1.106 |
| cluster match | 98% | — | — | 71% | — |

**ANSWER: at no fraction in 0.2–0.8.** Even one pinned dim leaves inpainting
**43%** better; the two meet only at 1.0, where nothing is pinned.

**FINDING: the information never runs out — the replacement trick does.** A
perfect conditional sampler would score `2·0.2/√π = 0.226` at *every* fraction,
because one dim names the cluster. The rise to 0.640 is reinjection picking the
wrong cluster more often (98% → 71% match), not the context going quiet.

**CONTROL:** with every dim masked, the lesson's `inpaint` and
`sample_unconditional` agree to **0.0** from one seed — the same code path.

### 2 — RePaint helps, but there is no edge for it to heal

The jump re-noises with the closed-form kernel after reverse steps 30, 20 and 10
— at `T = 40`, "every 10th step" is three resamplings, **55** network calls
instead of 40.

| mask | edge residual, plain → RePaint | cluster match, plain → RePaint |
|---|---|---|
| dims 3–4 | 0.344 → **0.307** | 96.0% → **98.6%** |
| dims 1–4 | 0.793 → **0.696** | 65.8% → **75.0%** |

**ANSWER: yes, modestly.**

**FINDING: there is no edge.** The plain sampler's edge (dim 3) and interior
(dim 4) residuals are **0.344** and **0.367** — the edge is no worse, so no seam
forms; the dims are exchangeable.

**FINDING: what RePaint fixes is the cluster.** The gain is global coherence, the
content agreeing with its context, which is the mechanism behind RePaint's seam
reduction on real images too.

**CONTROL:** with no jumps the loop reproduces the lesson's `inpaint` to **0.0**.

### 3 — pose control is free, and the "proper" fill model is not better

`diffusers` and `torch` are absent. Twenty face tasks (dims 3–4 masked, a
requested pose `x3 − x4 = ±0.4`), five seeds each. *Inpaint + ControlNet* is the
lesson's sampler plus a pose-guidance pull; *Fill* is a context-conditioned
denoiser trained with the lesson's own `init_net`/`forward`/`backward`/`apply`.

| pipeline | pose error | identity kept |
|---|---:|---:|
| Inpaint + ControlNet | **0.021** | **99%** |
| Fill | 0.483 | 92% |
| Inpaint, no control | 0.593 | 94% |

**ANSWER: Inpaint + ControlNet wins pose outright and does not lose identity.**

**FINDING: pose control costs identity nothing.** Pose is `x3 − x4`, identity
lies along `x3 + x4`: orthogonal, so the guidance cuts pose error **28-fold**
while identity rises.

**FINDING: the context-seeing model is no better at identity than the naive
trick** the doc says "works... badly" — **92%** against **94%** at the lesson's
training budget.

**CONTROL:** identity measured outside the mask is **0.0** residual for every
pipeline, since each pastes the source back — so it must be scored inside.
