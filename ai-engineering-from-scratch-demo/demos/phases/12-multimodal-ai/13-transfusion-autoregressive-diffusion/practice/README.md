<!-- generated:start -->
# 12-multimodal-ai / 13-transfusion-autoregressive-diffusion

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/13-transfusion-autoregressive-diffusion/) · upstream spec
`phases/12-multimodal-ai/13-transfusion-autoregressive-diffusion/docs/en.md`

```bash
uv run demo practice run 13-transfusion-autoregressive-diffusion --ex 1
uv run demo explain 13-transfusion-autoregressive-diffusion --ex 1
uv run pytest demos/phases/12-multimodal-ai/13-transfusion-autoregressive-diffusion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | A Transfusion-style model trains 70% text tokens and 30% image patches. The image diffusion l… | code | T0 | `ex01_the_lesson_balances_magnitudes_and_the_mix_balances_nothing.py` |
| 2 | Implement the block-triangular mask for a sequence: `[T, T, <image>, P, P, P, P, </image>, T]… | code | T0 | `ex02_the_closing_separator_cannot_see_the_image_it_closes.py` |
| 3 | MMDiT has modality-specific QKV weights. What parameter count overhead does this add vs Trans… | code | T0 | `ex03_twenty_five_percent_of_parameters_for_a_five_percent_axis.py` |
| 4 | Generation: given a text prompt, the model runs NTP for 50 tokens, then hits `<image>`, then… | code | T0 | `ex04_the_image_is_twenty_nine_percent_of_the_passes_and_ninety_nine_of_the_work.py` |
| 5 | Read SD3 paper Section 3. Describe rectified flow and why it converges in fewer inference ste… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Four code solutions and one prose one. Exercises 1 and 2 both find the lesson's
own implementation not doing what the surrounding prose claims: the trainer's
losses are closed-form countdowns rather than functions of the data, and the
block-triangular mask leaves its own separators unable to see the block they
delimit.

### 1 — the lesson balances magnitudes; the mix balances nothing

**ANSWER: two different numbers for two readings of "balance".**

| reading | `w_img / w_text` | resulting contributions (text, image) |
|---|---:|---|
| equal per-token magnitude | **0.1** | (0.7, 0.3) — text carries 70% |
| equal contribution after the mix | **0.2333** | (0.7, 0.7) |

**FINDING: the lesson ships the first and never asks the second.** `train` sets
`img_w = 0.1` exactly, and `two_loss_step` computes
`text_w * text_loss + img_w * img_loss` with **no token-count term anywhere** —
so the 70/30 mix the exercise is built on cannot enter the calculation.

**FINDING: neither loss in the toy depends on the data.**

| step k | `text_loss` | `img_loss` factor |
|---:|---:|---:|
| 0 | 1.2040 | 0.0400 |
| 5 | 0.5978 | 0.0100 |
| **10** | 0.2231 | **0.0000** |

`text_loss` is `-log(0.3 + 0.05k)` and `img_loss` is `(0.8 + 0.02k - 1)²` times
the noise magnitude — closed-form countdowns in the step index. The image loss
reaches exactly zero at step 10, which is the last step the demo runs.

**FINDING: the text loss goes negative at step 14.** `0.3 + 0.05k` passes 1.0
there, and `cross_entropy_toy` clamps only the lower end, so a "probability" of
1.05 yields **−0.0488**. The demo stops four steps short of its own arithmetic
breaking.

### 2 — the closing separator cannot see the image it closes

```
     0 1 2 3 4 5 6 7 8
 0 | 1 . . . . . . . .    T
 1 | 1 1 . . . . . . .    T
 2 | 1 1 . . . . . . .    <image>
 3 | 1 1 . 1 1 1 1 . .    P
 4 | 1 1 . 1 1 1 1 . .    P
 5 | 1 1 . 1 1 1 1 . .    P
 6 | 1 1 . 1 1 1 1 . .    P
 7 | 1 1 . . . . . . .    </image>
 8 | 1 1 1 1 1 1 1 1 1    T
```

**ANSWER: 40 of 81 entries are 1** — row sums 1, 2, 2, 6, 6, 6, 6, 2, 9, 49.4%
dense. Causal over text, fully bidirectional within the patch block, which is the
rule the mask is named for.

**FINDING: `</image>` attends to the two text tokens and to none of its four
patches.** `in_text` excludes both separators by name and `same_img` covers only
the open interval between them, so row 7 falls through every branch but "text at
or before me". The token whose job is to close the block cannot read it.

**FINDING: the patches cannot see their own opening separator either** — column
2 is zero across rows 3–6, for the mirror reason. The delimiters are
structurally invisible to the content they delimit, in **both** directions.

**FINDING: the trailing text token is the only complete row.** Row 8 sees all
nine positions, because a text query takes preceding text by one branch and
everything non-text by another, with no exclusion for the separators.

### 3 — 25% of parameters for a 5% axis

**ANSWER: +25% of block parameters — 7B becomes 8.75B.** Duplicating Q, K and V
is 3d² per layer against a 12d² block. The attention *operation* is still
shared: the two streams concatenate before the softmax, which is what makes
MMDiT one model rather than two.

**FINDING: duplicating the whole block instead would be +100%** — 12d² against
12d², which is what SD3's MMDiT actually does. The exercise names the cheaper
half of the design, and the two readings differ by **4×**.

**ANSWER: by this phase's own numbers, not obviously.**

| | |
|---|---:|
| parameter ratio | 1.25× |
| in decades | 0.097 |
| expected gain at Lesson 12.07's +7 MMMU/decade | **+0.68** |
| architecture axis (Lesson 12.07) | **5%** of variance |
| visual-token-count axis | **60%** |

The extrapolation assumes the extra parameters behave like scale, which is a
stated assumption and probably generous to MMDiT. Spending a quarter of the
parameter budget on the smallest of six axes is defensible only once the token
count — the axis worth twelve times as much — is already right.

**FINDING: the overhead is paid on every text token too.** The duplicated
weights are per-modality, so **3.3 of 16.3 GiB** of bf16 weights sit resident for
a modality a text-only request never touches.

### 4 — the image is 29% of the passes and 99% of the work

**ANSWER: 70 forward passes** — 50 text (one per token, with a KV cache) and
**20** image (one per denoise step, *not* one per patch).

| | forward passes | positions processed |
|---|---:|---:|
| text | 50 | 50 |
| image | **20** | **5,120** |
| share of total | 28.6% / **99.0%** | |

**FINDING: counting passes predicts latency backwards.** A text pass processes
one new position; a denoise pass processes all 256.

**FINDING: 20× the work in 12.8× fewer passes.** The same 256 patches generated
one token at a time is 256 passes and 256 positions. The trade is parallelism,
not total compute — and the factor is the denoise-step count in both directions.

**FINDING: the breakeven is 256 steps.** Below it Transfusion wins on passes; at
it the two arms match on both counts; above it loses on both. The shipped 20
steps is 12.8 patches per step, and that number — which the exercise treats as a
given — is what decides whether the architecture is cheaper than Chameleon's at
all.

### 5 — why rectified flow needs fewer steps

Drawing on **MMDiT: Stable Diffusion 3's variant**, which is where the lesson
places SD3 in this family, and on `two_loss_step`, which already implements the
objective.

**The lesson ships rectified flow and does not name it.** Look at the three lines
inside `two_loss_step`:

```python
xt         = (1 - t) * x + t * noise      # straight-line interpolation
target_vel = noise - x                    # constant velocity along it
img_loss   = mse(predicted_vel, target_vel)
```

That is the whole method. The path from data to noise is a **straight line**, the
target is its **constant** velocity, and the loss is a plain MSE. Nothing else.

**Why the straight line is the point.** DDPM defines a stochastic forward process
that adds Gaussian noise over T steps and learns to reverse it one step at a
time. The reverse trajectory in data space is **curved**, so a coarse
discretisation cuts corners and the error accumulates — which is why DDPM at 20
steps looks noticeably worse than at 250, and why an entire literature of
higher-order solvers (DDIM, DPM-Solver) exists to follow the curve with fewer
evaluations.

Rectified flow removes the curve from the target. Along any single
`(x₀, x₁)` coupling the velocity is constant, so an Euler step of size 1 would be
**exact** if the network returned that coupling's own velocity. The step count is
then not paying for curvature; it is paying for something else.

**What it is actually paying for.** The network cannot return a coupling's own
velocity, because it only sees `x_t` and many couplings pass through it. What it
learns is the conditional *expectation* of the velocity over all of them — and
that marginal field is curved even though every individual path is straight. This
is the honest version of "fewer steps": the target is straight, the learned field
is nearly straight, and a first-order solver is a good approximation *because the
approximation error is small*, not because it is zero.

Reflow is the procedure that closes the remaining gap: generate pairs with the
current model, retrain on those couplings, and the marginal field straightens
further. Each round buys fewer steps. SD3's own contribution on top is the
timestep schedule — sampling `t` from a logit-normal rather than uniformly, which
concentrates training where the velocity is hardest to predict, in the middle of
the trajectory.

**And the failure mode is a scale error, which is easy to see here.** The toy's
`predicted_vel` is the target times `0.8 + 0.02k`. At k = 0 that is **0.8×** the
true velocity — so a single Euler step from noise lands **20% short of the
data**, every time, with no noise and no curvature involved. In a curved
formulation that error would be tangled up with discretisation; in a straight one
it is visible as exactly what it is, which is the other reason the objective is
worth having.
