<!-- generated:start -->
# 04-computer-vision / 11-stable-diffusion

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/11-stable-diffusion/) · upstream spec
`phases/04-computer-vision/11-stable-diffusion/docs/en.md`

```bash
uv run demo practice run 11-stable-diffusion --ex 1
uv run demo explain 11-stable-diffusion --ex 1
uv run pytest demos/phases/04-computer-vision/11-stable-diffusion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Generate the same prompt with `guidance_scale` in `[1, 3, 5, 7.5, 10, 15]`. Describe h… | code | T1 | `ex01_cfg_scale_artefact_onset.py` |
| 2 | (Medium) Take any real photograph, run it through `StableDiffusionImg2ImgPipeline` at `streng… | code | T1 | `ex02_img2img_strength_signal_floor.py` |
| 3 | (Hard) Train a LoRA on 10-20 images of a single subject (a pet, a logo, a character) and gene… | code | T1 | `ex03_lora_rank_step_budget.py` |
<!-- generated:end -->

## Answers

This lesson ships no runnable model. `code/main.py` is five print statements and a
`text_to_image_stub` that checks `has_diffusers()` and `torch.cuda.is_available()`
and returns `None` when either fails — which, here, both do. Exercise 1 measures
exactly that (`False`, `False`, `None`), and downloading `runwayml/stable-diffusion-v1-5`
is out of bounds for these solutions. So all three exercises run against surrogates
built out of the prerequisite lesson's own code (`10-image-generation-diffusion`:
`linear_beta_schedule`, `precompute_schedule`, `q_sample`, `sample_ddim`, `TinyUNet`,
`train_step`, `synthetic_circles`), chosen so that the mechanism the question asks
about is exact rather than approximated. Two of the three find that **the lesson's
own text contradicts itself**, and the third finds that **the quantity the exercise
names is not the one that binds**.

### 1 — cfg scale artefact onset

The surrogate makes classifier-free guidance closed-form: prompt embeddings
`e ~ N(0, 0.15²)`, latents `x0 | e ~ N(e, 0.45²)`, so both halves of
`eps_uncond + w (eps_cond - eps_uncond)` are exact and go through lesson 10's own
`sample_ddim` at the 25 steps the lesson's own stub passes.

**ANSWER: artefacts have no onset.** Measured as sample values outside the data's
own observed range (`|x| > 2.125`, so the data itself scores 0.00% by construction):

| w | 1 | 3 | 5 | 7.5 | 10 | 15 |
|---|---:|---:|---:|---:|---:|---:|
| out-of-gamut | 0.00% | 0.01% | 0.31% | 2.34% | 5.72% | **12.93%** |
| count of 49,152 | 0 | 4 | 153 | 1,149 | 2,811 | 6,355 |
| sample std | 0.425 | 0.548 | 0.723 | 0.936 | 1.121 | 1.401 |
| cosine to prompt | 0.350 | 0.762 | 0.900 | 0.957 | 0.978 | 0.993 |

Four stray values in 49,152 at `w=3` is not an onset, it is noise — the peak value
is 1.858 at `w=1` and 2.204 at `w=3`. The curve is smooth, so the answer is the
curve. The reference's own `cfg_sweep_demo` labels 5.0 and 7.5 both `'standard'`
while the rate rises **8x** inside that one label.

**FINDING: the lesson's own sweep mislabels `w=1`.** `cfg_sweep_demo` prints
`'unconditional'` for `w=1.0`, but `w=1` cancels `eps_uncond` outright and leaves
the plain conditional. Cosine to the prompt is **-0.0021** at `w=0` and **+0.3495**
at `w=1`, against the data's own **0.3100**. The prose two sections above it has
this right — "w=0 is unconditional, w=1 is plain conditional" — and the code
beside it does not.

**MECHANISM: guidance buys adherence with diversity, at a closed-form rate.** An
affine `eps` collapses the whole 25-step chain to `x_0 = G x_T + H e`. `G`, all the
diversity one prompt has left, falls **0.398 → 0.169** while `H` climbs
**0.997 → 9.276**; `sqrt(G² + H² M²)` then reproduces the measured std above to
**6.8e-04**.

**MECHANISM: oversaturation is CFG not sampling the density it is named after.**
`p_cond^w p_uncond^(1-w)` is Gaussian with std shrinking **0.450 → 0.290** across
the sweep — guidance is *supposed* to narrow the distribution. Applying that score
at every `t` instead widens it to **1.401**, so `w=15` samples are **4.8x** wider
than the density they name. The guided scores are not a diffusion path, and that
gap is the artefact.

**CONTROL: adherence saturates long before the artefacts do.** From `w=7.5` to
`w=15` adherence gains **+0.0356** while the out-of-gamut rate goes 2.34% → 12.93%.
That is why the stub's own default is 7.5. Across two `x_T` seeds the std moves at
most 0.0015 and adherence 0.0030.

### 2 — img2img strength signal floor

`StableDiffusionImg2ImgPipeline` is rebuilt from the parts that decide the answer:
diffusers' img2img is the ordinary reverse loop entered late, so the loop here is
lesson 10's `sample_ddim` body with its `torch.randn` replaced by lesson 10's own
`q_sample`. At strength 1.0 the two agree to **0.0093**, which is checked rather
than asserted. The latent's variance is split into a coarse 4×4-block band
(`SL=1.2`) and its complement (`SH=0.35`), because an isotropic latent cannot tell
composition from texture. Guidance is 7.5, the value the lesson's own img2img
snippet passes.

**ANSWER: strength 1.0 ignores the input because `q_sample` scales it by 0.006353.**
That is `sqrt(alpha_bar_999)` on the lesson's own default linear schedule — the
whole photograph enters `x_T` as 0.6% of one unit of noise. Downstream: swapping in
a **different photograph** at the same noise moves the output by at most **0.0121**,
while changing the **noise** with the same photograph moves it **2.387** — a factor
of **198**.

**ANSWER: of the exercise's five, only 0.2 keeps composition.**

| strength | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|
| coarse-band gain | **0.814** | 0.453 | 0.158 | 0.027 | -0.008 |
| detail-band gain | 0.221 | 0.078 | 0.026 | 0.007 | 0.001 |
| style (cosine to new prompt) | 0.835 | 0.910 | 0.929 | 0.932 | 0.932 |
| product | **0.729** | 0.442 | 0.157 | 0.027 | -0.008 |
| denoiser calls | 5 | 10 | 15 | 20 | 25 |

The product peaks at 0.2 on both noise seeds, and style there already reaches
**90%** of its ceiling. `docs/en.md` says "0.5-0.7 is the standard range for style
transfer"; across that range composition runs **0.158 → 0.027**. A real photograph
holds on further, having more of its variance in the coarse band — but that is the
image's spectrum talking, not the pipeline, and the lesson states the range flat.

**MECHANISM: the reverse chain loses no composition at all — `q_sample` loses all
of it.** The coarse-band gain tracks `sqrt(alpha_bar)` at the entry step
(0.812 / 0.442 / 0.161 / 0.039 / 0.006) to within **0.0146**: whatever survives
`add_noise` survives the denoiser untouched. The detail band does not — a further
**3.7x to 6.1x** down, because the model's prior puts `SH=0.35` against `SL=1.2`
and shrinks what it believes is texture.

**CONTROL: strength is also the compute knob.** diffusers' `get_timesteps` keeps
`int(steps * strength)` of the grid, so 0.2 is **5** U-Net calls out of 25, not 25.

**CONTROL: guidance halves the raw similarity without removing any of it.** At
guidance 1 the plain cosine to the photograph is 0.625 at strength 0.2; at 7.5 it is
0.342. But the coarse-band gain is **0.829 against 0.814** — nothing was lost, the
prompt component simply grew around it. Scoring "composition preserved" by raw
similarity would blame strength for what guidance did.

### 3 — lora rank step budget

The frozen base is lesson 10's own `TinyUNet` (**37,395** parameters) trained by
lesson 10's own `train_step` for 60 epochs. The adapter is the real construction —
a rank-`r` down-projection into every `nn.Conv2d` plus a zero-initialised 1×1
up-projection, **942 to 7,536** parameters, against SD 1.5's 10-50 MB adapter on an
860M U-Net. Identity is scored as eps-MSE improvement on **12 held-out images** of
the subject the adapter never saw, which is generation's own training objective and
cannot be inflated by memorising the inputs.

**FINDING: every setting works, so identity alone cannot pick one.** All 12 grid
cells improve held-out eps-MSE, by **11.1% to 33.7%** — monotone in both rank and
steps, so ranking by identity just selects the biggest, longest-trained adapter.

| forgetting / held-out gain | 100 steps | 400 steps | 1200 steps |
|---|---:|---:|---:|
| rank 1 | 5.5% / 0.111 | 9.5% / 0.148 | 13.7% / 0.178 |
| rank 2 | 6.9% / 0.117 | 12.1% / 0.172 | 28.4% / 0.236 |
| rank 4 | 7.7% / 0.139 | 17.0% / 0.210 | 33.9% / 0.286 |
| rank 8 | 7.5% / 0.154 | **18.7% / 0.243** | 38.6% / **0.337** |

**MECHANISM: both curves climb with steps at every rank and neither plateaus**, so
a stopping rule has to come from their ratio. Forgetting over gain, by rank 1/2/4/8:

| | rank 1 | rank 2 | rank 4 | rank 8 |
|---|---:|---:|---:|---:|
| at 100 steps | 0.50 | 0.59 | 0.56 | 0.49 |
| at 1200 steps | **0.77** | 1.21 | 1.19 | 1.14 |

The ratio crosses 1 only at high rank **and** high steps. Rank 1 peaks at **0.77**
and never crosses at all: the product binds, not either knob.

**ANSWER: rank 8 at 400 steps.** Call a setting affordable when it forgets less of
the base distribution than it gains on unseen images of the subject. Raw identity
picks **r8@1200** (gain 0.337) and that is not affordable; the affordable frontier's
argmax is **r8@400** at gain **0.243**, one of **9** affordable settings in 12.

**FINDING: "without overfitting to the input images" names the wrong quantity.**
The train-minus-held-out gap the exercise asks about peaks at **0.007** across the
entire grid, while forgetting spans **5.5% to 38.6%**. It is forgetting, not
memorisation of the 12 inputs, that sets the budget — and the two knobs are not
symmetric: **8x the rank costs 1.35x** the forgetting where **12x the steps costs
2.48x**. Rank is cheap; steps are dear.

**CONTROL: the zero-initialised adapter is an exact no-op.** Injecting rank 8 into
the frozen model moves `eps` by **0.0** because `up` starts at zero, so every number
above is the adapter's own doing.

**Not measured here.** Two things the exercise asks for are missing on purpose, both
paid for by D14's line ceiling. There is no picture: the honest readout of a
37k-parameter stand-in is not a novel scene, so identity is the held-out denoising
loss instead. And the sweep runs one base-model seed, so the replication behind the
monotonicity claims is the four ranks — four independent adapter initialisations —
not a second foundation model. On real weights this is
`accelerate launch train_dreambooth_lora.py --rank=8 --max_train_steps=1200`
against SD 1.5 at batch 1: roughly 15-25 minutes on one RTX 4090, an estimate
rather than a run.
