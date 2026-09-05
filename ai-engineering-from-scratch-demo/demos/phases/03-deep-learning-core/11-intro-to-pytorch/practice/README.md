<!-- generated:start -->
# 03-deep-learning-core / 11-intro-to-pytorch

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/11-intro-to-pytorch/) · upstream spec
`phases/03-deep-learning-core/11-intro-to-pytorch/docs/en.md`

```bash
uv run demo practice run 11-intro-to-pytorch --ex 1
uv run demo explain 11-intro-to-pytorch --ex 1
uv run pytest demos/phases/03-deep-learning-core/11-intro-to-pytorch
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add batch normalization. Insert `nn.BatchNorm1d` after each linear layer (before the activati… | code | T0 | `ex01_batch_norm_layers.py` |
| 2 | Implement a learning rate finder. Train for one epoch with exponentially increasing learning… | code | T0 | `ex02_lr_range_finder.py` |
| 3 | Port to GPU with mixed precision. Add `torch.amp.autocast` and `GradScaler` to the training l… | code | T3 | `ex03_gpu_mixed_precision.py` |
| 4 | Build a custom Dataset. Download Fashion-MNIST (same format as MNIST but with clothing items)… | code | T0 | `ex04_custom_dataset.py` |
| 5 | Replace Adam with SGD + momentum. Train with `SGD(params, lr=0.01, momentum=0.9)`. Compare co… | code | T0 | `ex05_sgd_momentum_cosine.py` |
<!-- generated:end -->

## Answers

Every fixture here is a **seeded 784-D, 10-class Gaussian blob** — the lesson's
`download_mnist` fetches from a remote server, and these solutions run offline and
reproducibly. MNIST's shapes, none of MNIST's pixels. The lesson's own
`MNISTModel`, `create_loaders`, `train_one_epoch` and `evaluate` run over it
unchanged, so what is compared is always the lesson's code.

Three of the five exercises make a claim that the measurement reverses: batch norm
converges *later*, the LR finder's answer is a 15-point regression, and mixed
precision is 5× slower on the hardware that is actually here. The fourth and fifth
find the confound instead of the effect.

**1 — batch norm costs accuracy and reaches 98% later.**

| model | peak accuracy | ms/epoch | epochs to 98% |
|---|---:|---:|---:|
| dropout (the lesson's, with its dropout) | **0.9890** | 11.9 | **3** |
| + batch norm | 0.9845 | 13.5 (1.14×) | 5 |
| the lesson's `MNISTModelWithBatchNorm` | 0.9800 | 11.7 | 11 |

Bayes rate on this fixture is 0.9990. "Fewer epochs" holds only for epoch 1, where
batch norm leads 0.8940 to 0.8550.

**FINDING: `parameters()` undercounts the batch-norm checkpoint by 770.** The
dropout model's 235,146 parameters are 235,146 `state_dict` numbers; the batch-norm
model reports **235,914 against 236,684**. `+768` is `2·(256+128)` scales and shifts;
the other 770 are running statistics — buffers `parameters()` misses and
`torch.save` writes.

**FINDING: a size-1 last batch kills batch norm inside `train_one_epoch`.** 1281
rows at `batch_size=64` ends on a batch of 1:

```
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])
```

The dropout model runs the same loader fine. `60000 % 64 = 32`, so MNIST hides it —
and the missing `drop_last` in `create_loaders` stays a landmine.

**2 — the LR finder's answer is 20× too high, and stably so.**

128 steps from 1e-7 to 1.0: a plateau at 2.307 until `lr = 3.8e-03` (**64% of the
epoch**), a minimum of 1.309 at **2.22e-02**, and 40.4 at `lr = 1`.

The matched-budget ladder — same init, same epoch, one fixed LR each:

| LR | 1e-3 | 3e-3 | 2.22e-02 (the finder's) |
|---|---:|---:|---:|
| test accuracy | **0.9870** | 0.9450 | **0.7750** |

A **15–21 point** regression.

**MECHANISM: the curve at LR *x* is a model trained at every LR below *x*.** Only
18 of the 128 steps land within 3× of 1e-3, so the sweep reads **2.286** there — its
opening plateau — while a run *held* at 1e-3 for the same 128 steps reaches 0.9870.

**CONTROL: biased, not noisy.** Three seeds put the minimum in 1.96e-02–2.22e-02, a
spread of 1.14×, and name 1e-3 the ladder winner every time. fastai's `lr/10` lands
at 2.2e-03, whose rung tops out at 0.9450.

**3 — mixed precision is 5–6× slower here, and the accuracy claim is the one that holds.**

2048 samples/epoch on CPU: fp32 **128,319/s**, bf16 22,493/s (0.18×), fp16+scaler
18,333/s (0.14×). The real run is this file unchanged on a CUDA host —
`uv sync --extra llm && uv run python <this path>` on an A100-40GB at $1.29/h list,
about **$0.0018** for the 10 epochs the lesson quotes at 0.5 s each.

**MECHANISM: the 2× is a tensor-core property, not a float16 property.** Halving
the bit width buys nothing where there is no half-precision matmul unit — this CPU
runs bf16 GEMMs by widening back to fp32, so `autocast` only adds casts: **5.7× of
pure overhead**.

**FINDING: "mixed" is literal and asymmetric.** `autocast` returns bf16 and fp16
*logits* but an **fp32 loss** — `cross_entropy` is on autocast's fp32 list — and
after a full fp16 epoch with `GradScaler` every parameter is still fp32. On one
batch with a logit span of 0.4725: bf16 off by 0.3330, fp16 by 0.3409, and 30
argmax flips.

**CONTROL: accuracy is untouched.** One epoch each from the same seed: fp32 0.9920,
bf16 0.9920, fp16+GradScaler 0.9910. 3.30% of gradients fall below fp16's smallest
normal (6.10e-05); 65536× scaling cuts that to 0.197%. bf16's is 1.2e-38, so 0.191%
and no scaler. Headroom holds: 7.2e+03 < 65504.

**4 — the custom Dataset is a drop-in, and its whole contract is two methods that nothing validates.**

The same 2048 rows through a `DataLoader` of batch 64 differ by **exactly 0.0** from
`TensorDataset` — unshuffled, shuffled from the same generator seed, and against the
loader `create_loaders` itself builds.

**ANSWER: the harder task reproduces the gap the exercise predicts** — **99.2%** at
class separation 0.18 against **86.8%** at 0.12, same MLP, same optimizer, same 5
epochs. The real run is this file with `blobs` replaced by the lesson's own
`load_images`/`load_labels` over the Fashion-MNIST IDX files (4 files, ~30 MB) — the
only part of the exercise that needs the network.

**MECHANISM: `__len__` decides how many indices are drawn, and nothing checks it.**

| `__len__` returns | what happens |
|---|---|
| `n + 1` | `IndexError: index 2048 is out of bounds for dimension 0 with size 2048` |
| `n − 4` | yields **2044 of 2048 rows, silently** |

The sampler trusts `__len__`; `__getitem__` can object only by raising.

**FINDING: `__getitem__` may return the wrong dtype and the collate will take it.**
Returning each row as a numpy `float64` array collates to a `torch.float64` batch —
`default_collate` converts whatever it is handed. The model is the first thing to
object: *"mat1 and mat2 must have the same dtype, but got Double and Float."* A
Dataset that returns the wrong precision is a runtime error one layer away, not a
load error.

**CONTROL: shuffling is the sampler's job, not the dataset's.** `__getitem__` never
sees an epoch.

**5 — SGD catches up, the cosine does not do it, and the lesson's own run changes two things at once.**

Test accuracy per epoch:

| | e1 | e3 | e10 |
|---|---:|---:|---:|
| Adam(1e-3) | **0.976** | 0.992 | **0.992** |
| SGD(0.01, 0.9) | 0.314 | 0.948 | 0.989 |
| SGD(0.01, 0.9) + cosine | 0.314 | 0.940 | 0.988 |
| SGD(0.05, 0.9) | 0.934 | 0.986 | 0.979 |
| SGD(0.05, 0.9) + cosine | 0.934 | 0.986 | **0.988** |

**ANSWER: it catches up — 0.3 points apart at epoch 10** — but takes 3 epochs to
pass 0.948 where Adam is at 0.976 after **one**. Adam's whole advantage is the start.

**ANSWER: adding `CosineAnnealingLR` at lr = 0.01 does not help** — at or below the
unscheduled arm at every epoch after the first. A cosine only lowers the rate, and
0.01 was never too high.

**FINDING: the lesson's own `experiment_sgd_cosine` uses lr = 0.05** where
`experiment_sgd` and the exercise use 0.01, so its comparison confounds a **5× rate
increase** with the schedule. Separated: the rate buys the early epochs (0.934
against 0.314 at epoch 1); the cosine buys the late ones.

**FINDING: at the rate that *is* too high, the cosine earns its keep.** SGD(0.05)
peaks at 0.989 and **falls to 0.979** by epoch 10; with the cosine it holds 0.988.
What the schedule fixes is late-training decay at a rate too high to sit at — a
different claim from "SGD catches up".

**CONTROL: the schedule steps once per epoch**, so the scheduled and unscheduled
arms are the same run until the first `sched.step()` — 0.9340 and 0.3140 in both,
exactly. Any first-epoch difference would have been a seeding bug.
