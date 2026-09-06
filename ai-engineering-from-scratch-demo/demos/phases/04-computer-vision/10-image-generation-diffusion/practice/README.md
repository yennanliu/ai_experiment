<!-- generated:start -->
# 04-computer-vision / 10-image-generation-diffusion

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/10-image-generation-diffusion/) · upstream spec
`phases/04-computer-vision/10-image-generation-diffusion/docs/en.md`

```bash
uv run demo practice run 10-image-generation-diffusion --ex 1
uv run demo explain 10-image-generation-diffusion --ex 1
uv run pytest demos/phases/04-computer-vision/10-image-generation-diffusion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Visualise the forward process: take one image and plot `x_t` at `t in [0, 100, 250, 50… | code | T1 | `ex01_forward_process_snr_decay.py` |
| 2 | (Medium) Train the TinyUNet on the synthetic-circles dataset for 20 epochs and sample 16 circ… | code | T1 | `ex02_ddpm_ddim_same_seed.py` |
| 3 | (Hard) Implement a cosine noise schedule (Nichol & Dhariwal, 2021): `alpha_bar_t = cos^2((t/T… | code | T1 | `ex03_cosine_schedule_low_steps.py` |
<!-- generated:end -->

## Answers

The lesson's `main()` is honest about its own limits — it prints sample ranges of
`[-19, 23]` for data that lives in `[-1, 1]` — and all three exercises land on the
same thing from different directions: **the forward process is exact arithmetic
you can check to five decimals, and the reverse process at this budget is not**.
Two of the three exercises contain a claim that is false as written, and the
number that explains both is the same closed form.

**1 — the forward process matches `q(x_t | x_0)` exactly, and "pure noise" is a sample-size claim.**

The exercise asks for `t in [0, 100, 250, 500, 750, 1000]`. There is no `t = 1000`:
`linear_beta_schedule(T=1000)` returns 1000 betas indexed `0..999`, and `q_sample`
raises `IndexError: index 1000 is out of bounds for dimension 0 with size 1000`.
Measured at 999 instead, over all 200 circles (153,600 pixels):

| t | 0 | 100 | 250 | 500 | 750 | 999 |
|---|---:|---:|---:|---:|---:|---:|
| alpha_bar | 0.9999 | 0.8951 | 0.5214 | 0.0778 | 0.0033 | 4.04e-05 |
| mean | −0.7215 | −0.6845 | −0.5218 | −0.2008 | −0.0405 | −0.0079 |
| predicted mean | −0.7215 | −0.6827 | −0.5210 | −0.2013 | −0.0415 | −0.0046 |
| std | 0.5720 | 0.6307 | 0.8061 | 0.9720 | 0.9977 | 1.0004 |
| predicted std | 0.5720 | 0.6306 | 0.8057 | 0.9735 | 0.9989 | 1.0000 |
| corr(x_t, x_0) | 0.9998 | 0.8579 | 0.5113 | 0.1665 | 0.0323 | 0.0038 |
| predicted corr | 0.9998 | 0.8581 | 0.5126 | 0.1639 | 0.0329 | 0.0036 |

**ANSWER: every number a reader would take off the plot is the closed form.** With
`Var(x_0) = 0.3271`, `q(x_t|x_0) = N(sqrt(a_bar) x_0, (1 − a_bar) I)` predicts the
mean as `sqrt(a_bar)·mean(x_0)`, the standard deviation as
`sqrt(a_bar·Var + 1 − a_bar)`, and the correlation as
`sqrt(a_bar·Var / (a_bar·Var + 1 − a_bar))`. Worst gaps **0.0033 / 0.0014 / 0.0026**
against a **0.00255** standard error at this sample size.

**FINDING: the marginal is Gaussian-shaped at t = 500, 500 steps before the image
is gone.** `x_0` is strongly non-normal — skew **+1.757**, excess kurtosis
**+1.491**. By `t = 500` those are **+0.011** and **−0.020**, indistinguishable
from a Gaussian at 153,600 samples. But `corr(x_500, x_0)` is still **0.166** and
the mean is **−0.201**, not 0. The shape statistics say "noise" long before the
content does; a normality eye-test has to cover location and scale too.

**MECHANISM: what decays is `alpha_bar`, and the correlation is its square root in
SNR.** The signal's share of the variance runs 0.9997 → 0.7363 → 0.2628 → 0.0269 →
0.0011 → 0.0000, and the plotted correlation is exactly its square root.

**FINDING: at t = 999 "pure Gaussian noise" is a statement about how many pixels
you look at.**

| | pixels | predicted corr | z |
|---|---:|---:|---:|
| one image | 768 | 0.0036 | **0.10** (measured +0.018 → +0.49) |
| 40 draws × 200 images | 6,144,000 | 0.0036 | **11.0** (measured 0.004431) |

Seeing the residual at 3 sigma needs **~682,000 pixels**. `x_999` is not noise —
it is *undetectable*, which is a different claim. **CONTROL:** on all four moments
it passes for `N(0, 1)`: mean **−0.0079** (closed form −0.0046), std **1.0004**,
skew **−0.020**, excess kurtosis **+0.014**.

**2 — from the identical `x_T`, DDPM-1000 and DDIM-50 land in different places, and neither lands on a circle.**

Both samplers open with `torch.randn(shape)`, so reseeding to 7 before each call
hands them an `x_T` that differs by **exactly 0.0** — the comparison is about the
samplers, not the seeding.

| after 20 epochs (MSE 0.1838) | DDPM (1000 steps) | DDIM (50 steps) | data |
|---|---:|---:|---:|
| pooled correlation with the other | **0.460** | **0.460** | — |
| per-image correlation | 0.306 … 0.695 | | |
| std | 8.06 | 13.53 | **0.572** |
| range | [−75.7, 103.1] | [−115.9, 156.0] | [−1, 1] |
| pixels inside [−1, 1] | 21.9% | 9.4% | 100% |

**ANSWER: no — same noise, same weights, same schedule, and the 20× cheaper
sampler lands somewhere else.** **FINDING: but neither output is a circle.** Both
leave the data's range by roughly two orders of magnitude, which is what the
lesson's own `main()` already shows at `T = 200`. At 20 epochs, "do they produce
similar images?" is a question about two failures.

**MECHANISM: the reverse chain multiplies `x` by a fixed, closed-form gain, and
that gain is 157.**

| | measured | closed form |
|---|---:|---:|
| DDPM: `prod_t 1/sqrt(alpha_t)` | 157.4105 | `1/sqrt(alpha_bar_999)` = 157.4104 |
| DDIM (eta=0): `prod sqrt(a_prev/a_t)` | 157.4025 | `sqrt(a_0/a_999)` = 157.4026 |

DDPM's mean is `sqrt(1/alpha_t)·(x − c·eps)`, so each step scales `x` by
`1/sqrt(alpha_t)`; the product telescopes to `1/sqrt(alpha_bar_T)`. DDIM at
`eta = 0` keeps `sqrt(a_prev/a_t)` per step and telescopes to the same thing over
its subsampled grid. Any systematic bias the eps-network has at high `t` is
amplified **157-fold** before it reaches `x_0`.

**FINDING: train 5× longer and the answer flips.**

| after 100 epochs (MSE 0.1196) | DDPM | DDIM |
|---|---:|---:|
| std | **0.672** (data 0.572) | 3.75 |
| range | [−5.2, 8.0] | [−48.4, 55.2] |
| in range | 63.4% | 37.3% |
| pooled correlation | **0.126** (was 0.460) | |

DDPM's per-step noise injection re-randomises the accumulated error; DDIM's 50
deterministic steps carry it. **CONTROL: "similar?" has no stable answer at this
budget** — a 5× change in training moves it from "roughly half" to "not at all",
so the 20-epoch number is not a property of the two samplers.

**3 — the cosine schedule as written is 200× worse, and the exercise's claim only survives an SNR-matched comparison.**

**ANSWER: implemented the paper's way, checked back against the closed form.** The
exercise quotes only `alpha_bar_t = cos^2(...)`; `precompute_schedule` wants
*betas*, so the other half — `beta_t = 1 − a_bar(t)/a_bar(t−1)` clipped at 0.999,
and the normalisation `a_bar(t)/a_bar(0)` — has to be supplied. Round-tripped
through the lesson's own `precompute_schedule` it lands within **3.3e-07** of the
cosine; without the normalisation the chain would start at **0.99984**, not 1. The
0.999 clip binds on **exactly one** step, the last.

DDIM Wasserstein-1 to the data's pixel distribution, same model, 20 epochs:

| steps | 10 | 20 | 50 |
|---|---:|---:|---:|
| linear | 10.44 | 9.36 | 8.75 |
| cosine (paper, clip 0.999) | **2144** | **1866** | **1593** |
| cosine (clip 0.1, SNR-matched) | **9.56** | **8.40** | **7.36** |

**FINDING: run literally, "cosine gives better samples" is false by 200×.**

**MECHANISM: the clip is the whole story.** The cosine sends `alpha_bar` to 0 at
`t = T` by definition, and the 0.999 clip is all that stops it:

| | terminal alpha_bar | reverse gain `1/sqrt(alpha_bar_T)` |
|---|---:|---:|
| linear | 4.04e-05 | 157.4 |
| cosine (clip 0.999) | **2.43e-09** | **20,291** |
| cosine (clip 0.1) | 1.18e-04 | 91.9 |

That is exercise 2's gain again. One last step, not the schedule's shape, is the
200×.

**CONTROL: match the terminal SNR and the exercise's claim comes back, on every
point.** Clipping beta at 0.1 leaves the cosine's shape untouched but puts its
gain *below* linear's, and it then beats linear at **6/6** step-count × seed
points, by up to **16%** (seed 1: 10.16/8.95/7.94 against 11.19/10.05/9.43).

**MECHANISM: linear burns 30% of its chain on steps with nothing to undo.**
Linear's `alpha_bar` crosses 0.5 at **t = 259** and spends **29.8%** of the chain
below log-SNR −5, where `x_t` carries no image at all. The cosine crosses at
**t = 496** — the closed form `T·(1−s)/2 = 496` — and spends **5.3%**. At 10 DDIM
steps, a third of linear's budget lands where there is nothing left to reverse.

**CONTROL: cosine's higher training loss is that same artefact, not a worse
model.** Cosine reports MSE **0.2622** against linear's **0.1838**. Score both at
matched log-SNR instead of matched `t`:

| log-SNR | −8 | −4 | −2 | 0 | +2 | +4 |
|---|---:|---:|---:|---:|---:|---:|
| linear | 0.091 | 0.091 | 0.098 | 0.159 | 0.423 | **0.792** |
| cosine | 0.110 | 0.109 | 0.113 | 0.152 | 0.366 | **0.729** |

They agree to **0.063**, and cosine is the *better* of the two at the informative
end. Predicting eps is nearly free below log-SNR −5, and linear puts six times as
many timesteps there — so its average loss is lower for free.
