<!-- generated:start -->
# 03-deep-learning-core / 10-mini-framework

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/10-mini-framework/) · upstream spec
`phases/03-deep-learning-core/10-mini-framework/docs/en.md`

```bash
uv run demo practice run 10-mini-framework --ex 1
uv run demo explain 10-mini-framework --ex 1
uv run pytest demos/phases/03-deep-learning-core/10-mini-framework
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `SoftmaxCrossEntropyLoss` class for multi-class classification. Softmax the predictions… | code | T0 | `ex01_softmax_cross_entropy.py` |
| 2 | Implement learning rate scheduling in the optimizer: add a `set_lr()` method and wire in the… | code | T0 | `ex02_set_lr_warmup_cosine.py` |
| 3 | Add a `save()` and `load()` method to Sequential that serializes all weights to a JSON file a… | code | T0 | `ex03_save_load_json.py` |
| 4 | Implement weight decay (L2 regularization) in the Adam optimizer. Add a `weight_decay` parame… | code | T0 | `ex04_adam_weight_decay.py` |
| 5 | Replace the per-sample training loop with proper mini-batch gradient accumulation: accumulate… | code | T0 | `ex05_minibatch_accumulation.py` |
<!-- generated:end -->

## Answers

The framework works. What the five exercises find is that three of its
abstractions leak in ways the lesson's own `train_framework` walks straight into:

| leak | who trips on it |
|---|---|
| `parameters()` returns raw `(list, i, j, grads)` tuples — not a state dict, and no weight/bias distinction | ex 3, ex 4 |
| the DataLoader's batching is decorative — `train_framework` steps per *sample* inside it | ex 5 |
| `Adam` holds `t` and the moments, so any "change the LR" that rebuilds it resets them | ex 2 |

**1 — `p − y` is the gradient, and computing it the other way underflows to zero.**

Worst |analytic − central difference| over 200 random logit triples × 3 logits at
`h = 1e-06`: **1.866e-09** — the quotient's own `O(h²)` error.

The 3-class spiral is learned with it: 144/36 split,
`Linear(2,16)-ReLU-Linear(16,16)-ReLU-Linear(16,3)`, 30 epochs of Adam(lr=0.02) →
**train 100.0%, test 100.0%** against 33.3% chance, mean loss 0.9410 → 0.000292.
**CONTROL:** the identical run on *permuted* labels lands at chance — train 42.4%,
test 30.6%, loss stalled at 1.0708 against `ln 3 = 1.0986`.

**FINDING: the un-combined backward underflows to zero on confident logits.**
Composing the softmax Jacobian with `dCE/dp` reproduces `p − y` exactly at
`(5, 0, −5)`, returns **8.757e-12** at `(30, 0, −30)` where the true gradient is
1.000000, and returns **exactly 0.0** at `(400, 0, −400)`.

**MECHANISM.** `dCE/dp = −1/p` reaches ~1e+26 while the Jacobian row carries a
factor `p ~ 1e-26`. The product survives only if neither end is rounded away
first — and `p − y` never forms either factor.

**2 — the schedule buys optimization, not generalization; and rebuilding Adam turns it into signSGD.**

Final mean training loss over seeds (1, 2, 3), 20 epochs of Adam:

| | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| constant lr = 0.01 | 0.0230 | 0.0368 | 0.0216 |
| warmup(240) + cosine | **0.0125** | **0.0121** | **0.0105** |
| the same 4800 rates, shuffled | 0.0237 | 0.0359 | 0.0247 |

Lower on every seed — but the 60 held-out rows give 98.3 / 98.3 / 100.0 against
100.0 / 98.3 / 100.0, within 1.7 points.

**CONTROL: the *ordering* does the work, not the mean.** The shuffled arm is an
identical multiset at an identical mean of 0.005005, and it lands with the
constant arm. Only a monotone tail spends the **end** of training at a small rate.

**CONTROL: the `set_lr` plumbing itself changes nothing** — on a constant schedule
it reproduces the reference loop bit for bit: 0.022959050444682435 against
0.022959050444682435 over 4800 steps.

**FINDING: changing the LR by rebuilding Adam silently turns it into signSGD.** A
fresh `Adam` per step resets `t`, pinning bias correction at `t = 1`, where
`m̂/(√v̂ + ε) = g/(|g| + ε)`. All **43 of the 193** parameters with `|g| > 1e-5` moved
by 0.999998–1.000000 × lr although their gradients span 5.20e-03 to 9.87e-01.
After 4800 steps it still reports `t = 1`, at loss 1.6800 / 76.7% against
0.0230 / 98.3%.

**3 — the round trip is exact, and `parameters()` is not a state dict.**

**81 of 81** grid predictions equal *to the last bit* after an 833-byte JSON round
trip — max |difference| exactly **0.0**, all 33 scalars equal. **CONTROL:** before
the load, the differently-seeded target differed by up to **0.9181** on that grid.

**FINDING: BatchNorm's buffers are missing.** A save built from exactly what
`Sequential.parameters()` enumerates — `weights`, `biases`, `gamma`, `beta` —
reloads a BatchNorm model to predictions off by **0.9913**, against 0.0 once
`running_mean` and `running_var` travel too. (`BatchNorm.forward` moves
`running_mean` to 2.1175 and leaves `running_var` at `[1.0]`.) Buffers are not
parameters.

**FINDING: load must write *through* the lists.** Both loads restore the
checkpoint exactly (gap 0.0 either way) — but `parameters()` hands the optimizer
the list objects themselves, so rebinding `module.weights = [...]` leaves it on
the old ones. After 200 further Adam steps the predictions have moved **0.0**,
against 0.4407 when loaded in place. **A silently frozen model that loads
correctly.**

**4 — decay 0.01 shrinks the weights 6%, and the other reading of "L2" shrinks them 55%.**

40 epochs, 1000 Adam steps:

| | test loss / accuracy | ‖θ‖ |
|---|---|---:|
| decay 0 | 0.0619 / 96.0% | 11.975 |
| decay 0.01, decoupled | 0.0596 / 97.0% | 11.235 |
| decay 0.01, **coupled** | 0.0798 / 98.0% | **5.347** |
| decay 0.1, decoupled | 0.0543 / 97.0% | 7.753 |

**MECHANISM: the decoupled rule is exactly a geometric shrink.** With every
gradient zeroed, 1000 steps of `w -= lr·wd·w` land on `‖θ‖·(1 − lr·wd)ⁿ` to
**1.1e-14** relative — predicting a 9.5% shrink where 6.2% is measured. The
gradient pushes the rest back.

**FINDING: the two readings differ by 2.1×.** Folding `wd·w` into the *gradient*
shrinks to 5.347 against the decoupled 11.235, because Adam divides the coupled
term by `√v̂` — it survives as a pull of order `lr` however small `w` gets, while
the decoupled term vanishes with `w`. **The exercise's own wording ("shrinks
weights toward zero each step") picks the weaker one**, while its title says L2.

**FINDING: `parameters()` does not distinguish weights from biases.** 465 entries,
**41 of them biases**, told apart only by `j is None`. A loop over the list decays
them too; excluding them gives ‖θ‖ 11.371 against 11.235.

**CONTROL: ten times the decay moves the norm and not the accuracy** — 35% below
the undecayed run at 0.0543 / 97.0%. A 465-parameter net on 400 training points
has nothing to over-fit.

**5 — mini-batching changes convergence in opposite directions on the two clocks.**

| | epochs to loss < 0.10 | after 40 epochs | optimizer steps |
|---|---:|---|---:|
| per sample (the lesson's loop) | **3** | 0.2612 / 94.0% | 16,000 |
| per batch (the exercise's) | 6 | **0.0619 / 96.0%** | **1,000** |

The per-sample loop arrives first and then walks away.

**MECHANISM: 16× fewer steps for the same data.** `train_framework` iterates a
DataLoader of batch 16 and then calls `zero_grad`/`step` **inside** the inner loop,
so its batching only decides the shuffling order.

**MECHANISM: the framework was already built for this.** `Linear.backward` writes
`self.weight_grads[i][j] += ...` — gradients already accumulate across calls. The
whole change is moving `zero_grad` and `step` out of the inner loop and scaling
once. No new state, no new method.

**MECHANISM: "divide by batch size" is a no-op under Adam** — 0.0619 / 96.0%
divided against 0.0623 / 96.0% undivided. Adam divides the update by `√v̂`, so a
constant factor on every gradient cancels.

**CONTROL: under the lesson's SGD the same division decides the run** —
0.3864 / 82.0% divided against 0.1232 / 94.0% undivided. Dividing by 16 is a 16×
learning-rate cut, and nothing downstream undoes it.
