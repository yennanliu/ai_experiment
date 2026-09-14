<!-- generated:start -->
# 08-generative-ai / 07-latent-diffusion-stable-diffusion

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/07-latent-diffusion-stable-diffusion/) · upstream spec
`phases/08-generative-ai/07-latent-diffusion-stable-diffusion/docs/en.md`

```bash
uv run demo practice run 07-latent-diffusion-stable-diffusion --ex 1
uv run demo explain 07-latent-diffusion-stable-diffusion --ex 1
uv run pytest demos/phases/08-generative-ai/07-latent-diffusion-stable-diffusion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` with guidance `w ∈ {0, 1, 3, 7, 15}`. Record mean sample by class. A… | code | T0 | `ex01_class_one_crosses_at_w_one_and_class_zero_never_does.py` |
| 2 | Medium. Swap the toy linear encoder for a tanh-MLP encoder/decoder pair with a reconstruction… | code | T0 | `ex02_the_tanh_latent_is_worse_and_the_reason_is_the_decoder.py` |
| 3 | Hard. Set up a real Stable Diffusion inference with diffusers: load `sdxl-base`, run 30 Euler… | code | T0 | `ex03_four_steps_costs_ten_times_less_and_the_schedule_is_why.py` |
<!-- generated:end -->

## Answers

The lesson is a 184-line class-conditional latent diffusion with CFG dropout. All
three exercises are **T0**. The thing to notice before any of them:

```python
def encode(x):  return x * 0.5
def decode(z):  return z * 2.0
```

`decode(encode(x)) - x` is **0.0** — not small, zero. The latent is the data in
different units. Nothing is compressed, nothing is discarded, and the lossy
learned lower-dimensional code that makes latent diffusion worth doing is not
present.

### 1 — class 1 crosses at w=1, class 0 never does

| w | 0 | 1 | 3 | 7 | 15 |
|---|---:|---:|---:|---:|---:|
| class 0 | −1.45 | −1.71 | −1.70 | −1.65 | −1.60 |
| class 1 | +1.73 | **+2.28** | **+2.37** | **+2.36** | **+2.36** |

**ANSWER: class 1 from w=1 onward; class 0 at no w tested.** The question expects
one crossing point and the two classes of the same model do not share one.

**FINDING: the means are not monotone in w.** Both peak around w=1–3 and retreat;
the last two columns differ by 0.043. "At what w do the means diverge past" reads
as though guidance keeps pushing outward — past w=3 it does not push at all.

**CONTROL:** `sample_data` uses the literals −2.0 and +2.0, so the threshold is
the generating parameter, not a sample statistic.

### 2 — the tanh latent is worse, and the decoder's slope says why

**ANSWER: yes, and it gets worse** — on the same diffusion budget and seeds.

**FINDING: the autoencoder is the better reconstructor and the worse latent.** It
reconstructs to under 0.05 mse, so it has learned the data — and diffusion on top
of it still loses to a rescaling that learned nothing. Reconstruction quality and
latent quality are not the same axis, which a linear `encode` cannot show.

**FINDING: `tanh` saturates, so the decoder amplifies.** The data sits at ±2, deep
in the flat region: latents span only **1.490** while the decoder's slope is
**3.48** against the linear decoder's exact **2.0**. Every error diffusion makes
in latent space is multiplied by that slope on the way out.

### 3 — 4 steps costs ~10× less, and the schedule is why

`diffusers`, `torch`, `safetensors` are absent and SDXL is gigabytes this repo
does not ship. Both arms therefore sample from **one** trained network, which
isolates sampling — something the exercise's two-checkpoint version cannot do,
since those models differ in training too.

| arm | in-modes | time (200 draws) | network calls per draw |
|---|---:|---:|---:|
| 30 steps, w=7 | **1.000** | 0.59 s | **60** |
| 4 steps, w=0 | 0.525 | **0.04 s** | 4 |

**FINDING: the cost ratio is the call ratio and nothing else.** Guidance doubles
the calls per step, so 30 steps at w=7 is 60 forwards against 4 — a **15×**
arithmetic prediction against **14.6×** measured.

**FINDING: the two arms do not visit the same noise levels.** The short arm
touches 4 of 40 steps — 10% of the schedule — landing on `alpha_bar` values
`[0.6671, 0.8331, 0.9532, 0.9999]` rather than the range the model trained
across. That, not the missing guidance, is what a distilled model like
`sdxl-turbo` is retrained to fix: it is trained *for* a short schedule, where this
one is merely run on one.
