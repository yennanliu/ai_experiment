<!-- generated:start -->
# 03-deep-learning-core / 06-optimizers

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/06-optimizers/) · upstream spec
`phases/03-deep-learning-core/06-optimizers/docs/en.md`

```bash
uv run demo practice run 06-optimizers --ex 1
uv run demo explain 06-optimizers --ex 1
uv run pytest demos/phases/03-deep-learning-core/06-optimizers
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement Nesterov momentum, where you compute the gradient at the "lookahead" position (w -… | code | T0 | `ex01_nesterov_lookahead.py` |
| 2 | Implement a learning rate warmup schedule: linear ramp from 0 to max_lr over the first 10% of… | code | T0 | `ex02_warmup_cosine.py` |
| 3 | Track the effective learning rate for each parameter during Adam training. The effective rate… | code | T0 | `ex03_effective_lr_spread.py` |
| 4 | Implement gradient clipping (clip by global norm). Set the max gradient norm to 1.0. Train wi… | code | T0 | `ex04_clip_by_global_norm.py` |
| 5 | Compare Adam vs AdamW on a network with large weights. Initialize all weights to random value… | code | T0 | `ex05_adam_vs_adamw.py` |
<!-- generated:end -->

## Answers

Five exercises, five comparisons between optimisers, and the same result each
time: **the comparison the exercise asks for cannot be won on this fixture, and
the reason is always that the thing being varied is not what decides the
outcome.** Adam normalises the gradient away, so clipping it is a no-op;
schedules only change how the learning-rate budget is spent; and the one arm that
does shrink weights fastest does it by turning the classifier into a constant.

Two of the five also find defects in the lesson's own code: `Adam` accepts no
`weight_decay`, so the exercise's headline comparison cannot be run as written,
and STEP 4 prints a ratio of endpoints where it means a ratio of shrinkages.

**1 — Nesterov wins on the circle data, and loses to plain SGD on the bowl.**

At the lesson's own `lr = 0.05`:

| | 90% at epoch | 95% at epoch | after 300 epochs |
|---|---:|---:|---|
| momentum | 6 | never (best 94.0%) | 0.2069 / 93.0% |
| Nesterov | **3** | 16 | **0.0647 / 97.5%** |

**MECHANISM.** On `f(x) = (x−3)²` — the quadratic the lesson's own `__main__`
minimises — both methods are exact two-term recurrences in the error `e = x − 3`:

| | recurrence (lr = 0.1, β = 0.9) | per-step rate |
|---|---|---:|
| SGD | `e[t+1] = 0.8·e[t]` | **0.8000** |
| momentum | `e[t+1] = 1.70·e[t] − 0.90·e[t−1]` | 0.9487 = √β |
| Nesterov | `e[t+1] = 0.8·(1.90·e[t] − 0.90·e[t−1])` | 0.8485 |

Worst residual against those hand-derived coefficients over 40 steps: **1.78e-15**.
The `(1 − 2·lr)` factor is what separates the two — Nesterov is momentum with the
whole bracket damped.

**FINDING: on that quadratic plain SGD beats both.** `|e₄₀|` = 9.30e-04 (SGD),
8.56e-03 (Nesterov), 8.12e-01 (momentum) — momentum is **873× behind**. A 1-D bowl
has no narrow valley, so there is nothing for momentum to smooth and its
overshoot is pure cost.

**CONTROL: the circle-data win is a step-size effect, not a free lunch.** Standard
momentum at `lr = 0.01` ends at **0.0041 / 100.0%** — better than Nesterov at
`lr = 0.05`. `0.05` is past momentum's stability edge, and what the lookahead buys
there is damping, not acceleration.

**2 — warmup + cosine is slower than constant Adam at every learning rate tried.**

Epochs to reach 90% on the circle data:

| max_lr | constant | warmup + cosine |
|---|---:|---:|
| 0.001 (the default) | **39** | 54 |
| 0.003 | **13** | 28 |
| 0.01 | **5** | 16 |

**MECHANISM: the schedule spends exactly half the budget.** Mean lr over the run
is **0.000500** — `max_lr/2` to six places — so the run integrates to 30.0
lr-steps against constant's 60.0. Both halves cost the same: a linear ramp to
`max_lr` averages `max_lr/2`, and so does the cosine tail.

**CONTROL: halving alone does not explain it — the shape helps.** A *constant*
`lr = 0.00050`, the schedule's own mean, needs **76** epochs — 22 *later* than the
schedule — and lands on the same 99.0% endpoint. So the shape buys speed and the
budget sets the finish.

**CONTROL: the decay half does earn its keep once max_lr is aggressive.** At
`max_lr = 0.01` the scheduled run reaches 100% training accuracy at epoch 157 and
the constant run never does (best 99.5%). Annealing to zero is what the cosine
tail is for. The *ramp* is the part this problem does not need: warmup insures
against early instability, and a 33-parameter net trained one sample at a time has
none to insure against.

**3 — no, Adam's parameters are not updated at the same speed, and the ones with zero gradient move anyway.**

`|lr · m̂ / (√v̂ + ε)| / lr` across all 33 parameters:

| step | min | median | max | spread | grad == 0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.0316 | 0.4379 | 0.6924 | 21.9× | 16 of 33 |
| 50 | 0.0042 | 0.3860 | 0.6365 | **152.9×** | 24 of 33 |
| 200 | 0.0068 | 0.0637 | 0.4106 | 60.2× | 20 of 33 |

**FINDING: 24 of 33 gradients are exactly zero at step 50, and 18 of those still
move by over 0.1·lr.** ReLU zeroes any unit with `z ≤ 0`; Adam steps from the
momentum buffer `m`, which remembers.

**MECHANISM: the ratio measures agreement, not gradient size.**

| gradient sequence | m̂/√v̂ | measured |
|---|---|---|
| constant | exactly 1 | 1.0 for 10 steps running (worst gap 4.0e-09, the ε) |
| sign-flipping | `(1−β₁)/(1+β₁)` = 1/19 | 0.052632 |
| gone to zero | decays toward `β₁/√β₂` | 0.9032 per step → 0.9005; 46 steps to 0.01·lr |

**FINDING: `lr` is not a ceiling on Adam's step.** The largest effective rate over
4,000 steps is **1.2102·lr**, against a Cauchy–Schwarz bound of
`(1−β₁)/√((1−β₂)(1−β₁²/β₂))` = 7.27·lr.

**CONTROL.** Recomputing the tracked quantity from the optimiser's own `m`, `v`
and `t` matches the actual parameter delta to **1.1e-16** over 4,000 steps — it is
the update, not a parallel model of it.

**4 — 0 runs diverge with clipping, 0 without, and NaN is unreachable in this lesson.**

The exercise asks for a count of NaN divergences. Adam at `lr = 0.01`, 40 epochs,
seeds 0-9: **0 and 0**. Mean final loss 0.0555 clipped against 0.0523 plain,
accuracy 97.7% against 98.2%, with 14.8% of steps clipped.

Two reasons the metric cannot separate the arms.

**FINDING: nothing in this lesson can reach NaN.** SGD at `lr = 1e30` ends with a
weight of 6.79e+29 and a final loss of **10.188939036499** — bit-identical to the
same run at `lr = 100`. `sigmoid()` clamps its input to `[-500, 500]` before `exp`
and the loss clamps `p`, so the loss is bounded whatever the weights do.

**MECHANISM: an 8× cut in the gradient moves Adam's largest step under 20%.**
Unclipped, the global norm reaches 8.12 — 8.1× the clip — yet the largest
parameter move is 1.73·lr unclipped and 1.58·lr clipped. Adam divides by `√v̂`:
the gradient's magnitude is already gone from the step.

**CONTROL: Adam is invariant to a global gradient rescale.** Multiplying every
gradient by 1000 for 20 epochs reproduces all 33 parameters to **1.2e-07**
absolute (3.3e-08 relative — the ε). A clip *is* a rescale, so it can only act
through *which* steps it selects.

**CONTROL: the same clip is decisive for SGD**, whose step *is* the gradient. At
`lr = 5.0` over the same ten seeds: largest move 345.1 unclipped against 5.00
clipped, mean final loss 14.20 against 0.91. Still 0 NaNs, because there are none
to be had.

*A note on "10 random seeds":* the lesson's `OptimizerTestNetwork` calls
`random.seed(0)` inside its own constructor, so a caller cannot vary the
initialisation. All ten runs start from weights differing by **0.0**; the only
seed that varies is the data seed.

**5 — AdamW shrinks weights faster only because the lesson's Adam has no weight decay.**

Weight L2 norm from a `uniform[-5, 5]` start, seeds (0, 1, 42):

| arm | epoch 0 | epoch 50 | epoch 200 | final accuracy |
|---|---|---|---|---|
| Adam (as shipped, no decay) | 15.62, 17.51, 17.25 | — | 16.49, 18.21, 16.17 | 95.5, 99.5, 96.0 |
| AdamW (`weight_decay=0.1`) | same | 5.94, 7.19, 6.30 | 4.02, 4.68, 4.14 | 89.5, 93.5, 91.5 |
| Adam + L2 in the gradient | same | — | **0.80, 0.77, 0.88** | **71.5, 71.5, 71.5** |

The exercise says "train for 200 epochs with `weight_decay=0.1`" for both
optimizers — but the lesson's `Adam.__init__` takes no `weight_decay` argument, so
that arm cannot be run as written. Its Adam is decay-free, which is the entire
reason AdamW wins.

**FINDING: at a matched weight decay the ordering reverses.** Folding `wd · w`
into the gradient before `Adam.step` — the pre-AdamW reading of "Adam with weight
decay" — shrinks the norm to 0.80/0.77/0.88 against AdamW's 4.02/4.68/4.14, **on
every seed**. "AdamW should show faster weight shrinkage" is false.

**MECHANISM: decoupled decay is proportional; coupled decay is normalised.** With
zero gradients, AdamW multiplies every weight by `(1 − lr·wd)` per step — 100
steps land on `n₀·(1 − lr·wd)¹⁰⁰` to **1.8e-16** relative. That is a pull of
5e-04 at `|w| = 5.0`, and it *vanishes with w*. The coupled term instead enters
the numerator and is divided by `√v̂`, so it survives as a pull of order
`lr = 0.001` however small `w` gets — which is why it wins the shrinkage race.

**CONTROL: the extra shrinkage is bought by destroying the classifier.** The L2
arm's final accuracy is **71.50% on all three seeds** — exactly the majority-class
rate (71.5% of the circle data is labelled 0). It predicts class 1 for **0 of 200**
points. It is a constant function, not a fit.

**FINDING: the lesson's STEP 4 prints the wrong ratio.** Its 100-step demo shrinks
the norm by 0.0117 under Adam and 0.1117 under AdamW — a factor of **9.6**. What
it reports is `norm_adam / norm_adamw = 1.0x`, a ratio of *endpoints* rather than
of shrinkages, which reads as "no difference" for a 9.6× effect.
