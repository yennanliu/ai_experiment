<!-- generated:start -->
# 08-generative-ai / 04-conditional-gans-pix2pix

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/04-conditional-gans-pix2pix/) · upstream spec
`phases/08-generative-ai/04-conditional-gans-pix2pix/docs/en.md`

```bash
uv run demo practice run 04-conditional-gans-pix2pix --ex 1
uv run demo explain 04-conditional-gans-pix2pix --ex 1
uv run pytest demos/phases/08-generative-ai/04-conditional-gans-pix2pix
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Modify `code/main.py` to add a third class. Confirm G still maps each class's noise to… | code | T0 | `ex01_a_third_class_is_a_label_not_a_mode.py` |
| 2 | Medium. Replace L1 with a perceptual-style loss in the 1-D setting (e.g. a small frozen D act… | code | T0 | `ex02_both_reconstruction_losses_collapse_the_conditional.py` |
| 3 | Hard. Sketch a CycleGAN in the 1-D setting: two distributions, two generators, cycle loss. Sh… | code | T0 | `ex03_the_cycle_loss_does_not_pick_which_map.py` |
<!-- generated:end -->

## Answers

The lesson is a 199-line conditional GAN on a 1-D two-mode mixture, `G(z, c)`
against `D(x, c)`, with the class as a one-hot appended to both inputs. All three
exercises are **T0**, and all three run into the same thing from different sides:
the *data* is two points and a little noise, so anything the exercise asks the
model to learn about structure was already decided by a one-line sampler.

### 1 — a third class is a label, not a mode

| | class 0 | class 1 | class 2 |
|---|---:|---:|---:|
| real data at `num_classes=3` | −1.99 | **+2.00** | **+2.00** |
| G, trained | −2.23 | +2.03 | +2.18 |
| real data, three centres | −2.00 | 0.00 | +2.00 |
| G, trained on those | −2.10 | +0.05 | +1.96 |

**ANSWER: yes — and for the third class that confirmation means nothing.**
`sample_real_conditional` is `if c == 0 … else`, so every class above zero draws
`gauss(+2, 0.3)`. Classes 1 and 2 land **0.15** apart against **4.25** between
classes 0 and 1. The exercise asks to confirm a mapping the data does not define,
and it passes because being wrong about class 2 is indistinguishable from being
right about class 1.

**FINDING: give the third class a mode and the same network finds it** — closest
pair **1.90** apart. Capacity was never the constraint.

**FINDING: the lesson's own 600 steps cannot confirm anything.** At `main()`'s
budget the stock two-class run is unconverged, putting *both* classes on the same
side of zero on 1 seed of 3 (**−1.27**, **−0.79**). The confirmation needs about
**3,000** steps — 5× what the lesson runs.

**CONTROL: widening the label is one integer.** `one_hot`, `g_forward` and
`d_forward` are already written over `num_classes`.

### 2 — both reconstruction losses collapse the conditional

| arm | mean per-class σ | against real 0.30 |
|---|---:|---|
| the lesson as shipped | 0.146 | already 2.1× too narrow |
| perceptual, weight 0.5 | 0.046 | 6.5× too sharp |
| perceptual, weight 2.0 | 0.025 | 12× too sharp |
| L1, weight 1.0 | **0.023** | **13× too sharp** |

**ANSWER: it changes sharpness sharply, and every arm overshoots.** The spread
was already on the wrong side of the target before any reconstruction term, and
each term moves it further the same way.

**FINDING: there is no L1 in use to replace.** `update_g`'s reconstruction
parameters default to `(0.0, None)` and `main()` calls it with neither, so the
branch never executes. "Replace L1" has to begin by switching on a term the
lesson does not run.

**FINDING: in 1-D the two losses are one family, and only the weight differs.**
Both pull `x_hat` to one point per class, and the four arms lie on a single
monotone curve. A perceptual loss can differ from L1 only where its features fold
two inputs together — and leaky-ReLU of an affine map of a scalar is monotone
unit by unit, so this distance is *provably* a reweighted `|x − t|`. That is
stated as mathematics, not dressed up as a measurement; what is measured is the
curve.

**CONTROL: the frozen extractor is frozen, and is not the live critic.** It still
equals the snapshot taken beside it at warm-up, and no longer equals the critic.

### 3 — the cycle loss does not pick which map

Each generator is `x ↦ w·x + b` — the smallest thing that maps a line to a line,
which is what makes the claim provable instead of merely observed.
`A = N(−2, 0.5)`, `B = N(+2, 0.5)`, nothing ever paired.

| | seeds | w | b | predicted b |
|---|---:|---:|---:|---:|
| order-reversing | **6 of 8** | −0.93 | +0.14 | `μ_B + μ_A` = 0 |
| order-preserving | **2 of 8** | +0.95 | +3.78 | `μ_B − μ_A` = +4 |

**ANSWER: yes — the marginals match and the cycle closes.** G_AB sends A to mean
**+1.99**, σ **0.47** against B's +2.0 and 0.5, and `G_BA(G_AB(a))` returns within
**0.16** of `a`. No sample was ever paired with another.

**FINDING: which map it learns is not determined, and both answers appear.** Same
objective, same data, same code — the seed decides whether the largest `a` goes to
the largest `b` or the smallest.

**FINDING: the ambiguity is exact.** Matching the marginal forces
`|w| = σ_B/σ_A = 1` and `b = μ_B − w·μ_A`; the cycle forces `w_BA = 1/w_AB`. Both
signs satisfy both, so the objective has **two global optima**, and the two
predicted intercepts are measured at **+3.78** and **+0.14**. Training cannot
break a tie the loss cannot see — which is the standard CycleGAN critique, here
small enough to write down rather than assert.

**CONTROL: both arms match the marginal.** Every seed lands within **0.15** of
B's mean and **0.06** of its σ, whichever sign it chose — so these are two
converged solutions the objective scores identically, not one solution and one
failure.
