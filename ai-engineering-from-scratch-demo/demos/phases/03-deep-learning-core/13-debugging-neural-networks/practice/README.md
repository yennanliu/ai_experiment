<!-- generated:start -->
# 03-deep-learning-core / 13-debugging-neural-networks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/13-debugging-neural-networks/) · upstream spec
`phases/03-deep-learning-core/13-debugging-neural-networks/docs/en.md`

```bash
uv run demo practice run 13-debugging-neural-networks --ex 1
uv run demo explain 13-debugging-neural-networks --ex 1
uv run pytest demos/phases/03-deep-learning-core/13-debugging-neural-networks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add an exploding gradient detector. Modify the `NetworkDebugger` to detect when gradients exc… | code | T0 | `ex01_exploding_gradient_detector.py` |
| 2 | Build a dead neuron resurrector. Write a function that identifies dead ReLU neurons (always o… | code | T0 | `ex02_dead_neuron_resurrector.py` |
| 3 | Implement the learning rate finder with plotting. Extend `find_learning_rate` to save results… | code | T0 | `ex03_lr_finder_csv.py` |
| 4 | Create a data pipeline validator. Write a function that checks for: duplicate samples across… | code | T0 | `ex04_data_pipeline_validator.py` |
| 5 | Debug a real failure. Take the mini-framework from Lesson 10, introduce a subtle bug (e.g., t… | code | T0 | `ex05_locate_the_transpose.py` |
<!-- generated:end -->

## Answers

The lesson ships four diagnostics — `check_activations`, `check_gradients`,
`check_loss_health`, `gradient_check` — plus `overfit_one_batch` and
`find_learning_rate`. Between them the five exercises find that **three of the six
are silent on exactly the failure they are named for**, and that the two remaining
exercises ask for tools the lesson has no place to put.

| tool | what it misses |
|---|---|
| `check_gradients` | the exploding run it is named for — NaN fails both threshold tests |
| `find_learning_rate` | its suggestion is set by the sweep's step count, not by the curve |
| `gradient_check` | tests 5 entries per parameter, and names the layer *before* the bug |
| `NetworkDebugger` | nothing at all that looks at the data (ex 4) |

**1 — `check_gradients()` calls an exploding run HEALTHY.**

A gain-2.8 initialisation drives the loss to NaN in 60 steps. `check_gradients()`
returns `['HEALTHY']`.

**MECHANISM: NaN fails every comparison, so both branches skip.** All 41 recorded
`grad_output` abs-means are NaN, and the guards `< 1e-7` and `> 100` are
`(False, False)` for NaN.

**MECHANISM: and before any NaN the `> 100` bound is out of reach anyway.** At gain
2.8 the probe norm is 63.9 while the largest `grad_output` abs-mean is **1.34e-02**
— **7473× under the bound** — because the hook averages `|grad|` over a 32×32
tensor rather than taking a norm.

**FINDING: `check_loss_health()` catches what `check_gradients()` misses** —
`NAN_OR_INF` on the exploding run and `NOT_DECREASING` on the control. It is the
only check that tests `math.isnan`.

**2 — resurrecting incoming weights alone does not recover the network.**

75% of hidden units are killed with the lesson's own BUG 2 (a −6.0 bias).

| repair | dead after | `overfit_one_batch` |
|---|---:|---|
| Kaiming on **incoming** weights (as written) | 36 / 48 | fails — loss 0.5644, acc 68.0% |
| incoming **and** outgoing | 1 / 48 | loss 0.0008, acc 100.0% |
| never damaged (control) | 1 / 48 | loss 0.0008, acc 100.0% |

Recovery is a **match**, not merely an improvement.

**MECHANISM: Kaiming scales from `fan_in` and knows nothing about the bias.** A
resurrected unit has to push `w·x` past 6.0 to clear the −6.0 bias; per layer the
best it manages is 6.93, 1.12, 0.65 — only layer 1, fed by raw inputs, gets there.

**MECHANISM: a dead unit stays dead because its gradient is exactly 0.** Summed
`|grad|` over every dead unit's weight row and bias is **0.0**, and 200 Adam steps
move the dead count from 38 to 38. `ReLU'(z) = 0` below zero.

**3 — the LR finder's suggestion is a function of its step count.**

| steps | suggested | minimum | ratio | closed form `(end/start)^(10/steps)` |
|---:|---:|---:|---:|---:|
| 100 | 1.318 | 8.318 | 6.31× | 6.31 |
| 200 | 3.631 | 9.120 | 2.51× | 2.51 |
| 400 | 6.026 | 9.550 | 1.58× | 1.58 |

Matched to **0.00%**. `results[min_idx − 10]` is a fixed *step* offset, so the
suggestion moves **4.6×** across the three sweeps while the curve does not move at
all.

**FINDING: "just before the loss starts climbing" points at nothing here.** Across
1e-07 to 10 the loss never rises **once** — 0 rises in 99, 199 and 399 transitions
— and the minimum is the **last** sample of every sweep. The divergence guard never
fires; all three sweeps run their full length.

**MECHANISM: the sweep is full-batch.** It calls `model(x_data)` on all 512 rows at
once, so the rate it suggests is calibrated to a batch size the training loop will
not use. Held at one rate for the same 100 full-batch steps: 2.2461 at 1e-3, 0.0008
at the suggested 1.318, 0.0011 at 1.0.

**CONTROL: the model really is restored** — `deepcopy(state_dict())` and
`load_state_dict` leave every parameter bit-identical (**0.0** worst difference).
What it does *not* restore is the caller's optimizer.

The exercise's real target is ResNet-18 on CIFAR-10: this file with `blobs`
replaced by `torchvision.datasets.CIFAR10` and `net` by
`torchvision.models.resnet18(num_classes=10)` — about 6 GPU-min on an A100-40GB at
$1.29/h, ~$0.13.

**4 — the validator's four rules, and what three of them leave undefined.**

Clean split: no findings. Corrupted one fault at a time: 12 duplicate rows;
imbalance 594:1; unnormalised (mean 2.94, std 3.99); 2 non-finite. One rule each.

The lesson's own `NetworkDebugger` offers `check_activations`, `check_gradients`
and `check_loss_health` — and **nothing that looks at the data**.

**FINDING: "duplicate samples" needs an equality rule, and `==` is a lower bound.**
Copying 12 training rows into the test set is caught; adding **1e-05** to each of
the same rows first hides **all 12**. Any resize, re-encode or augmentation between
the split and the check makes a leak invisible.

**FINDING: ">10:1" does not say ratio of what.** The imbalanced split
`[594, 6, 0, 0]` is **594:1** as max/min — which fires — and **4.0:1** as max/mean —
which does not. And the rule cannot be evaluated at all when a class is absent,
since min is 0.

**FINDING: "mean near 0, std near 1" passes a dataset with no feature near
either.** Offsetting alternate features by ±0.6 and scaling by 0.8
(`offset² + spread² = 1`) leaves the pooled mean at **−0.012** and the pooled std at
**0.993** — all four rules silent — while the worst *per-feature* mean is **0.67**.
Two pooled scalars cannot see an offset that cancels across columns.

**CONTROL: NaN is the only unambiguous rule, and it has to run first.** Two
poisoned entries are counted exactly — but the same tensor's mean and std come back
NaN, so the normalisation rule evaluates `abs(nan) > 0.1` as **False** and stays
silent. A validator that runs the rules in the order the exercise lists them
reports the NaNs last and the normalisation not at all.

**5 — the transpose is in module 2, and the gradient check names module 0.**

| model | worst relative difference |
|---|---:|
| clean | **1.389e-08** |
| one line of module 2's `backward` transposed | **1.946e+00** |

Per module, on the bugged model:

| module 0 | module 2 (the bug) | module 4 |
|---:|---:|---:|
| **1.95e+00** | 1.39e-08 | 6.20e-10 |

**FINDING: it names the layer *before* the bug.** The transpose corrupts
`input_grad` — the next layer down's error — while module 2's own weight gradients
stay right. **Gradient checking localises the symptom, not the cause.**

**FINDING: the exercise's two halves do not compose.** This lesson's
`gradient_check` on the lesson-10 framework raises
`AttributeError: 'list' object has no attribute 'double'`. It calls `x.double()`
and `model.named_parameters()`; lesson 10 has Python lists and a `parameters()` of
its own. The checker used here is written to the same formula — central
differences, relative difference, a 1e-5 threshold — over all **369** parameters.

**MECHANISM: the bug is only insertable where the layer is square.**
`self.weights[j][i]` needs a row `j`, so on the 4→16 layer it raises
`list index out of range` on the first backward pass. Only the 16→16 layer accepts
it in silence — **a transpose is a no-op on shape exactly where it is undetectable
by shape.**

**CONTROL: the lesson's own sampling rule would have missed it.** `gradient_check`
tests `min(5, param.numel())` entries per parameter, in flat order. The first 5 of
module 0 already disagree here, so it is caught — but the entries at 5..100
disagree just as much, and a bug confined to those would have been reported **OK**.
