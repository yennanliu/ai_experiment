<!-- generated:start -->
# 04-computer-vision / 23-diffusion-transformers-rectified-flow

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/23-diffusion-transformers-rectified-flow/) · upstream spec
`phases/04-computer-vision/23-diffusion-transformers-rectified-flow/docs/en.md`

```bash
uv run demo practice run 23-diffusion-transformers-rectified-flow --ex 1
uv run demo explain 23-diffusion-transformers-rectified-flow --ex 1
uv run pytest demos/phases/04-computer-vision/23-diffusion-transformers-rectified-flow
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Train the TinyDiT above on the synthetic blob dataset for 500 steps. Compare samples p… | code | T1 | `ex01_euler_step_count_error_curve.py` |
| 2 | (Medium) Add text conditioning by concatenating a learned class embedding to the time embeddi… | code | T1 | `ex02_class_conditioning_colour_match.py` |
| 3 | (Hard) Compute the Fréchet distance (FID proxy) between generated samples from rectified-flow… | code | T1 | `ex03_frechet_rf_vs_ddpm_convergence.py` |
<!-- generated:end -->

## Answers

Rectified flow is the rare topic where the exact answer beats the experiment, and
all three exercises here turn on that. The interpolant `x_t = (1-t)x_0 + t*eps`
has velocity `eps - x_0` with no `t` in it, so the regression target is
t-independent by identity rather than by approximation, and on a one-point
dataset the exact velocity field is *constant along every trajectory* — which
makes an Euler solve exact in one step. Every claim the lesson makes about step
counts can therefore be measured against a field that is known in closed form
rather than against a short training run, and two of the three exercises ask for
a statistic that cannot rank what they are asking about.

### 1 — 10, 20 and 50 Euler steps, and the field underneath them

Trained exactly as the exercise says: the lesson's own `TinyDiT` (dim 96, depth
4), 500 steps of `rectified_flow_train_step`, loss 1.4394 → 0.1332. Every solve
is re-seeded to the same starting noise, so the step counts are compared on one
trajectory; the error is a distance to a converged 400-step solve, as a fraction
of the 25.38 that solve travels.

| Euler steps | trained net | exact marginal field (128 blobs) | sample range |
|---|---:|---:|---|
| 1 | 16.3% | 16.3% | — |
| 2 | 12.4% | 12.7% | — |
| 4 | 8.1% | 6.6% | — |
| 10 | **3.9%** | 4.4% | [-2.45, 1.82] |
| 20 | **2.1%** | 4.4% | [-2.48, 1.87] |
| 50 | **0.8%** | 2.9% | [-2.50, 1.90] |

**ANSWER: the three land a few percent apart, and the error is first-order.**
3.9% / 2.1% / 0.8%, falling ~2× per doubling (1.32×, 1.53×, 1.87×). Going from 10
steps to 50 buys one digit of ODE accuracy, not a different picture.

**FINDING: the statistic `main()` prints cannot rank them.** It reports
`samples range [min, max]` and the shape. Across 10, 20 and 50 steps that range
moves **0.043** at the bottom and **0.080** at the top, while the images behind
it differ by **5.1×**. A saturated metric ranks nothing, so every number here is
a distance to a converged solve instead.

**MECHANISM: the regression target is exactly t-independent.**
`rectified_flow_train_step` writes `target_v = epsilon - x0` and never touches
its own `t`. Recovered from the noised sample alone as `(x_t - x_0)/t` at
t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, it returns `eps - x_0` to **1.1e-15** in float64 —
rounding, not agreement. Every training step regresses one function of
(x_0, eps) whatever t the batch drew, which a schedule-scaled DDPM target cannot.

**MECHANISM: on a one-point dataset the lesson's own sampler is exact in ONE
step.** The exact marginal field `E[eps - x_0 | x_t]` is constant along every
trajectory when the dataset is a single image, and `rectified_flow_sample` at 1
step matches its own 400-step solve to **9e-07** of the travel — float32
rounding, which is the tolerance an exact identity gets from a sampler that
allocates in float32. Add one more image and straightness is gone: 1 step is
**17.9%** off, 2 steps 4.1%, exact again only from 4.

**FINDING: the trained model's step curve *is* the true field's curve.** Running
the closed-form marginal field of the same 128 blobs — what a perfect fit would
be — through the same sampler gives the middle column above: the two agree to
**1.5%** over 1–10 steps. The residual at 10 steps is curvature in the target
field, not undertraining; more training would not remove it, only reflowing the
paths would. Past 10 steps the exact field stops improving because it memorises
(its endpoints sit **4.9e-05** of the travel from a real training image) and all
that is left is which image it lands on.

**CONTROL: straight end to end, curved pointwise.** Read on the straight chord
joining a solve's own endpoints, the trained velocity deviates from that chord's
single constant direction by **29.1% / 15.9% / 9.6% / 12.5% / 15.8%** at
t = 0.1…0.9 — worst at the noise end — and yet 50 steps land 0.8% from 400. That
gap between "straight in aggregate" and "constant pointwise" is the whole reason
one step is not enough here and is enough on one point.

### 2 — Class conditioning, and what adaLN-Zero does to the first step

`synthetic_blobs` draws `rng.uniform(-1, 1, 3)` per image, so the exercise's "10
classes by colour" do not exist; the labels are built by repainting the lesson's
own blob masks from a fixed 10-entry palette. And "concatenating" cannot be done
as written — `TinyDiT.__init__` hardcodes `DiTBlock(dim, heads, cond_dim=dim)`,
so a 2×64 concatenation does not fit the `AdaLNZero` linear it feeds and has to
be projected back to 64. Both routes are wired in by *replacing* `time_mlp`,
which is the only conditioning call `TinyDiT.forward` makes.

| arm | colour accuracy (10 classes) | parameters | final loss |
|---|---:|---:|---:|
| concat + projection | **0.983** | 178,764 | 0.2005 |
| plain addition (the DiT paper) | 0.258 | 170,508 | 0.2187 |
| chance | 0.100 | — | — |

**ANSWER: classes 0, 5 and 9 come out the colour they were asked for** — 1.00,
1.00, 1.00, with all ten averaging **0.983** against a 0.100 chance rate, judged
by the hue of each sample's 8 brightest pixels against the palette.

**FINDING: the projection the concatenation forces is worth every parameter.**
It costs **8,256** on top of addition's 170,508 and buys 0.983 against **0.258**
— on losses that barely differ (0.2005 vs 0.2187), so the training loss cannot
tell you which route worked. The class fixes only the blob pixels, a small slice
of the MSE.

**CONTROL: the colour is not coming from the marginal.** The palette sums to a
mean colour of norm **0.0081** over the whole training set, so an unconditional
model has no colour to prefer — while the same nearest-palette classifier reads
the *data's* own labels at **1.000**. The metric is exact on real blobs and blind
on the marginal.

**MECHANISM: adaLN-Zero makes every DiT block exactly the identity at
initialisation.** `AdaLNZero.__init__` zeroes its modulation linear, so
scale = shift = gate = 0 and `x + gate * attn(...)` returns x untouched. Block
output minus input is **0.0** — exactly, not approximately — and the whole model
inherits it: at init its output is bit-identical for t=0 vs t=1 (**0.0**) and for
class 0 vs class 9 (**0.0**). A fresh DiT is a t-blind, class-blind map.

**MECHANISM: so the class embedding this exercise adds gets exactly zero gradient
on the first step.** One `rectified_flow_train_step` leaves **124,928 of 178,764**
parameters — **23 of 36** tensors — at gradient exactly 0.0: every attention and
MLP weight, the wrapped `time_mlp`, and the new `embed`. Inside
`self.mlp(cond).chunk(3)` only the gate third is alive (**0.0, 0.0, 9.025e-03**),
because scale and shift reach the loss only through a branch the gate multiplies
by zero. Step 1 trains the patch embed, the head and the gates and cannot use the
class at all — which is the warm start adaLN-Zero is designed to give.

### 3 — Which converges faster, and why the asked-for number cannot say

Lesson 23 ships no DDPM code, so the second arm is lesson 10's own `train_step`,
linear beta schedule and `sample_ddim`, driving the **same** TinyDiT — both
lesson-10 functions call `model(x_t, t)` and nothing else, so the "same-size
network" is literally the same network and only the objective and sampler change.

| training steps | rectified flow | DDPM (lesson 10 objective) |
|---|---:|---:|
| 200 | **0.678** | 1.22e+03 |
| 400 | **0.567** | 706 |
| 600 | **0.561** | 581 |
| floor (two halves of the real data) | 0.1175 | 0.1175 |

| sampling steps | 1 | 2 | 4 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| rectified flow | 0.653 | 0.634 | 0.585 | 0.564 | **0.561** |
| DDIM | 788 | 606 | **504** | 550 | 581 |

**ANSWER: rectified flow converges faster by two orders of magnitude, at every
checkpoint.** Its *first* checkpoint (200 steps) beats DDPM's last (600) by
**857×**, on one dataset, one optimiser, one seed and 171,660 shared parameters.

**FINDING: the distance the exercise literally asks for ranks nothing.** Fréchet
between the two *generators* is **584** — within **0.4%** of DDPM's own distance
to the data (581), because the rectified-flow arm has already arrived (0.561) and
is standing where the data is. A distance between two answers cannot say which is
closer to the truth; each arm's distance to the data can, read against the 0.1175
floor.

**MECHANISM: DDIM divides by `sqrt(alpha_bar)` and the Euler step has no
analogue.** `sample_ddim` forms `x0_pred = (x - sqrt(1-a)*eps) / sqrt(a)`, and
`1/sqrt(alpha_bar)` on the linear schedule reaches **157.4** at t=999 — so
whatever the eps-head still gets wrong is multiplied by that. DDPM samples come
out at std **10.258** against the data's **0.196**; rectified flow's Euler step
adds `v*dt` with dt ≤ 1 and lands at **0.205**.

**FINDING: RF is flat in the few-step regime while DDIM is not even monotone.**
RF's 1-step sample is within **1.2×** of its 20-step one and beats DDIM's *best*
by **771×**; DDIM peaks at 4 steps and gets worse at 10 and 20, because more
steps of a badly-scaled field is more error, not less. That is the lesson's
"fewer steps" claim, tested rather than asserted.

**CONTROL: it is the objective and the budget, not the transformer.** Lesson 10's
own TinyUNet (37,395 parameters) under the same objective, schedule, data and 600
steps scores **176** at std 6.150 — **1,495×** the floor, the same failure mode as
the DiT's 581. DDPM is not broken; it is a 1000-step chain being asked to
converge in 600, which is exactly the comparison the exercise's "same number of
steps" sets up.

### A note on file lengths

The three files run 146 / 145 / 137 lines of code, over D14's 120-line target and
under its 150-line ceiling. Exercise 1 carries the exact marginal velocity field
and a second step-count sweep through it; exercise 2 carries two training arms
plus a first-step gradient audit; exercise 3 carries three training runs and a
Fréchet distance whose matrix square root is taken through `torch.linalg.eigh`,
so the whole file stays inside the `vision` dependency group.
