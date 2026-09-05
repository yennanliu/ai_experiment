<!-- generated:start -->
# 03-deep-learning-core / 07-regularization

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/) · upstream spec
`phases/03-deep-learning-core/07-regularization/docs/en.md`

```bash
uv run demo practice run 07-regularization --ex 1
uv run demo explain 07-regularization --ex 1
uv run pytest demos/phases/03-deep-learning-core/07-regularization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement spatial dropout for 2D data: instead of dropping individual neurons, drop entire fe… | code | T0 | `ex01_spatial_dropout.py` |
| 2 | Implement label smoothing from lesson 05 combined with dropout from this lesson. Train with f… | code | T0 | `ex02_smoothing_plus_dropout.py` |
| 3 | Add a BatchNorm layer between the hidden layer and the activation in your circle-dataset netw… | code | T0 | `ex03_batchnorm_before_relu.py` |
| 4 | Implement early stopping: track test loss each epoch, save the best weights, and stop if test… | code | T0 | `ex04_early_stopping.py` |
| 5 | Compare LayerNorm vs RMSNorm on a 4-layer network (not just 2). Initialize both with the same… | code | T0 | `ex05_layernorm_vs_rmsnorm.py` |
<!-- generated:end -->

## Answers

Two facts about the lesson's code decide four of the five exercises:

1. **`RegularizedNetwork` trains one sample at a time.** BatchNorm on a batch of
   one returns a constant zero, so exercise 3 as written removes the hidden layer.
2. **`BatchNorm`, `LayerNorm` and `RMSNorm` have no `backward`.** Only `Dropout`
   does. So `gamma` and `beta` never move, and any use of them back-propagates as
   if the layer were the identity.

The third fact is the fixture: 150 training points and 16 hidden units, on data
separable enough that a plain network reaches **100%** test accuracy. There is no
overfitting for dropout, smoothing or early stopping to remove, and every
regularizer in this lesson costs accuracy instead of buying it.

**1 — spatial dropout does not beat standard dropout, and only the widest setting moves at all.**

Train−test gap over 8 mask seeds:

| grouping | gap | seed sd |
|---|---:|---:|
| none | +2.00 pp | — |
| width 1 (= standard dropout) | +1.25 pp | ±1.08 |
| width 4 | +1.17 pp | ±1.94 |
| width 16 | **−0.67 pp** | ±1.60 |

Widths 1 and 4 differ by **0.08 pp** against a seed sd of **1.94 pp** — under half
the noise the answer is read through. On 150 test points one flipped label is
0.67 pp.

**CONTROL: at width 1 this *is* the lesson's own `Dropout`,** value for value —
same global RNG stream, same `1/(1−p)` scaling. Every width is the lesson's code
regrouped, not a reimplementation.

**MECHANISM: grouping holds the mean and scales the sd by √width.** Over 6000
masks the summed activation is 31.98 ± 3.68 at width 1 and 32.12 ± 14.71 at width
16, against a flat 32.00 (worst drift 0.4%) — and **every sd over √width is 3.68**.
`1/(1−p)` stays per-channel; `C` draws replace 32.

**FINDING: width 16 underfits rather than regularizes.** Its gap is *negative* —
test above train — at 96.75% accuracy. With 2 channels both are off in 8.7% of
passes (`p^C = 9.00%`), erasing the hidden layer that often and leaving the output
on `b2` alone.

**2 — every regularizer widens the gap, and the lesson's own printout says otherwise.**

Eval-mode train − test accuracy, and test accuracy:

| | gap | test accuracy |
|---|---:|---:|
| neither | **+2.00 pp** | **97.33%** |
| dropout | +2.89 pp (±0.92) | 96.22% |
| smoothing | +2.67 pp | 95.33% |
| both | +2.11 pp (±0.25) | 95.78% |

**FINDING: regularization does not close the gap here, it costs test accuracy.** A
plain net already at +2.00 pp has no overfitting to remove, so every constraint
added is capacity lost.

**FINDING: read off the lesson's own printout the answer reverses.**
`train_model` grades the *training* set through the dropout mask, so its printed
`gap=` line reads neither +2.00, dropout +0.89, smoothing +2.67, **both −0.56 pp**
— and picks "both" as the winner. That is an artefact of measuring train accuracy
in training mode.

**CONTROL: two-class smoothing is BCE on a softened target** — lesson 05's
`label_smoothed_cce(alpha=0.1)` is BCE at target 0.95 to **2.8e-16**, and lesson
07's own `backward` differentiates it to 3.5e-10.

**MECHANISM: smoothing buys the gap with confidence, and log-loss bills it.** Top
test prediction 0.9999999999 unsmoothed against 0.9927 smoothed — a 0.95 ceiling
it still overshoots — and test loss on hard labels 0.1658 against 0.0592.

**3 — the BatchNorm the exercise asks for cannot be inserted where it says.**

`BatchNorm.forward([[1.0, -2.0, 3.0, 0.5]])` returns **`[0.0, 0.0, 0.0, 0.0]`**. On
a batch of one the variance is 0, so `x̂` is 0 and the output is `β`. Inserted
between the hidden layer and the activation of a network that trains one sample at
a time, it makes the hidden layer independent of its input.

`hasattr(cls, 'backward')`: BatchNorm **False**, LayerNorm **False**, RMSNorm
**False**, Dropout True.

**ANSWER: at the three rates named, the vanilla network never diverges.** The
lesson's own network and loop, unmodified, over 200 epochs:

| lr | 0.01 | 0.05 | 0.1 |
|---|---:|---:|---:|
| test accuracy | **100%** | **100%** | **100%** |
| test loss | 0.0361 | 0.0069 | 0.0067 |

There is nothing to rescue.

**ANSWER: batched, so the layer can run at all, it is worse at all three rates** —
plain 80 / 96 / 100% against BatchNorm's 72 / 92 / 96%. Normalising 16
pre-activations over 16 samples throws away the scale the one output neuron reads,
and the `γ` that would learn it back never moves.

**FINDING: BatchNorm lowers the rate at which training collapses.** Past the named
rates, plain holds 100% up to `lr = 4.0` and only fails at 8.0; BatchNorm reaches
100% at none of 1.0, 2.0, 3.0, 4.0 or 8.0.

**4 — early stopping saves 889 epochs on one arm and 0 on the other, and keeps the worse model.**

| | best-accuracy epoch | epochs tied at it | patience-20 stop | saved |
|---|---:|---:|---:|---:|
| plain | 8 | **992 / 1000** | never fires | **0** |
| regularized | 11 | 976 / 1000 | epoch 111 | **889** |

**ANSWER (question 1): "which epoch had the best test accuracy" names the first of
a 992-way tie.** On 50 held-out points at 100% there is no best epoch to report.

**ANSWER (question 2): 889 epochs saved — on the arm that has one.** The plain
network's test loss is still setting records at epoch 999 (0.0005, the best of the
run), so patience 20 never fires and saves nothing.

**FINDING: the rule keeps a worse model than the one it throws away.** It restores
epoch 90 at test loss **0.0427**, while the run's best is **0.0292 at epoch 640** —
529 epochs *after* the stop — and 0.0364 at the end. Patience 20 fired on a
plateau, not on overfitting.

**MECHANISM: there is no overfitting here to stop.** Train loss falls 0.5644 →
0.0013 (plain) and 0.5780 → 0.1130 (regularized), and test loss falls with it —
rising on 216 and 483 of 999 epoch-to-epoch steps and never turning.

**CONTROL: the number being stopped on is the number being reported.** The same 50
points choose the stopping epoch, select the weights and score the result — and
the 100.0% they report does not survive a fresh 200-point draw, where the finished
nets score **98.0%** and **97.5%**.

**5 — RMSNorm is faster; the accuracies are not the same; and no norm is better than either.**

200 epochs on 2-16-16-16-1 from one shared init:

| | test accuracy | ms/epoch |
|---|---:|---:|
| no normalisation | **100.0%** | **11.30** |
| RMSNorm | 100.0% | 12.15 |
| LayerNorm | 94.0% | 12.85 |

**Half of "faster with the same accuracy" holds.** RMSNorm is 5.4% faster per
epoch — and **6 points more accurate**, so the accuracies are not the same either.

**MECHANISM: the gap is large per call and small per epoch.** One call on a
16-vector costs ≈1.8 µs for LayerNorm against ≈0.9 µs for RMSNorm — LayerNorm
computes a mean, a variance and a `β` where RMSNorm computes one sum of squares.
But an epoch is 150 samples × 4 layers of pure-Python matrix work, so the norm
calls carry only ~5% of it.

**MECHANISM: the two are not the same transform here.** RMSNorm is LayerNorm
without the mean subtraction, so they coincide only at zero mean. On this net's
first hidden pre-activation `|mean(z)| / rms(z)` averages **0.19**.

**FINDING: `γ` never moves, so "training" the norms is not what is being
compared.** After 200 epochs every `γ` in both arms is still exactly 1.0. First-layer
`|grad|` is 0.5163 with no norm, 0.5886 with LayerNorm and 0.5857 with RMSNorm —
the norms raise it ~14% and differ from each other by 0.5%.
