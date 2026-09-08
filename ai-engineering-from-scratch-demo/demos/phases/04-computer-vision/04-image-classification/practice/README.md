<!-- generated:start -->
# 04-computer-vision / 04-image-classification

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/04-image-classification/) · upstream spec
`phases/04-computer-vision/04-image-classification/docs/en.md`

```bash
uv run demo practice run 04-image-classification --ex 1
uv run demo explain 04-image-classification --ex 1
uv run pytest demos/phases/04-computer-vision/04-image-classification
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Train the same model with and without mixup for five epochs on the synthetic dataset.… | code | T1 | `ex01_mixup_loss_floors.py` |
| 2 | (Medium) Implement Cutout — zero out a random 8x8 square in each training image — and run an… | code | T1 | `ex02_cutout_ablation.py` |
| 3 | (Hard) Build a CIFAR-100 pipeline (100 classes, same input size) and reproduce a ResNet-34 tr… | code | T1 | `ex03_cifar100_resnet34_sweep.py` |
<!-- generated:end -->

## Answers

### 1 — Why mixup's train loss is higher

Both arms are the lesson's own `train_one_epoch`, five epochs, the lesson's own
optimiser and schedule. The premise holds — but only just, and for a reason the
raw numbers hide.

| arm | train loss (5 epochs) | val accuracy | final train loss |
|---|---|---|---:|
| mixup | 1.184 0.630 0.597 0.395 0.561 | 1.000 | **0.561** |
| no mixup | 1.109 0.654 0.591 0.556 0.546 | 1.000 | **0.546** |

**FINDING: the two numbers are not the same objective.** The mixup arm minimises
`soft_cross_entropy` against a mixed target; the other minimises
`cross_entropy(label_smoothing=0.1)`. Each has an irreducible floor, and they are
not the same floor:

| arm | floor | final loss | above its own floor |
|---|---:|---:|---:|
| mixup | 0.2050 (measured) | 0.561 | **0.356** |
| label smoothing 0.1 | 0.5003 (closed form) | 0.546 | **0.045** |

So mixup sits **7.8×** further from its own optimum, against a raw gap of 0.015.
The comparison the exercise invites understates the effect by a factor of 24 —
and would reverse if the no-mixup arm dropped its label smoothing.

**MECHANISM: the floor is the entropy of the label being predicted.** One λ is
drawn per batch from Beta(0.2, 0.2); over 200,000 draws E[H(λ)] = **0.2278**, and
nine pairs in ten cross classes, giving **0.2050**. A *perfect* model scores
exactly that, because the target really is a mixture. The loss is higher without
the fit being worse.

**CONTROL: score the same weights on clean data and the gap disappears** — the
mixup-trained model gives **0.031** at accuracy **1.000** on un-mixed,
un-augmented training images. The elevated number describes what it was scored
on, not the fit.

**CONTROL: the reference's mixup train accuracy is a coin flip.**
`train_one_epoch` argmaxes the *mixed* image against the *un-permuted* `y`, so it
reports 0.514 0.537 0.467 0.473 0.777 — averaging **0.554**, near the 0.500 of
guessing which of the two labels dominates — while those same weights are at
1.000 on clean data.

**CONTROL: this dataset cannot test "val accuracy similar or better".** Both arms
end at 1.000 on 200 held-out images. `synthetic_cifar` gives class *c* the
frequency 2 + *c*, so there is no generalisation gap for mixup to close.

### 2 — Cutout, and a four-way ablation that cannot rank anything on accuracy

| arm | val accuracy | final val loss | epochs to 1.000 |
|---|---:|---:|---:|
| none | 1.000 | 0.078 | 4 |
| hflip+crop | 1.000 | 0.067 | 3 |
| hflip+crop+cutout | 1.000 | 0.057 | 4 |
| hflip+crop+mixup | 1.000 | 0.032 | 4 |

**ANSWER: all four reach val accuracy 1.000**, which is the metric the exercise
asks for and the one that separates nothing.

**FINDING: val loss separates what val accuracy ties.** The four final val losses
fall in augmentation order across a span of **0.046**, while re-running
`hflip+crop+cutout` under a second seed moves its own by only **0.001** — the
ordering is **72×** the seed noise. The tie at 1.000 is the metric's resolution,
not the arms being equal.

**MECHANISM: the hole is exactly 8×8×3 = 192 entries, and it moves.** Composed
after `standardize`, cutout zeroes exactly 192 entries, its top-left corner taking
**177** distinct positions over 200 draws out of the 625 a wholly-inside placement
allows.

**CONTROL: composed before `standardize`, "zero" is not zero.** The hole is filled
with raw 0.0, which (0 − 0.5)/0.25 turns into **−2.0** — two standard deviations
below the mean, an out-of-distribution patch rather than the neutral one the
method wants.

**CONTROL: composed before `random_crop`, it is no longer an 8×8 square.** Traced
with a NaN fill, the surviving hole ran from **60 to 324** entries over 200 draws,
reaching the intended 192 only **62%** of the time. The crop slides part of the
square off the image, and reflect padding copies a border-touching hole back in
twice.

### 3 — A CIFAR-100 pipeline, a 3×2 sweep, and a target that does not exist

`ResNet-34` here is the [3, 4, 6, 3] BasicBlock stack at widths (64, 128, 256,
512) on a CIFAR stem — **21,328,292** parameters, mapping (2, 3, 32, 32) → (2,
100). The sweep runs the lesson's own `MiniClassifier` at 100 classes so the whole
exercise finishes on a CI core.

| lr | weight decay | val accuracy |
|---:|---:|---:|
| 0.01 | 5e-4 | 0.2325 |
| 0.01 | 5e-2 | 0.1925 |
| 0.05 | 5e-4 | 0.5262 |
| 0.05 | 5e-2 | 0.0312 |
| 0.2 | 5e-4 | **0.8738** |
| 0.2 | 5e-2 | 0.0037 |

Six rows written to a real CSV and read back identically.

**MECHANISM: the two knobs are not separable — weight decay flips the sign of the
learning-rate effect.** At wd 5e-4 accuracy climbs 0.233 → 0.526 → 0.874 with the
learning rate; at wd 5e-2 it *falls* 0.193 → 0.031 → 0.004. Sweeping one at a time
would have read the lr effect backwards at the larger decay: SGD shrinks weights
by `lr·wd` a step, so the two multiply.

**FINDING: the errors are neighbours, not a scatter.** Top confusions (true,
predicted, count): (15, 13, 12), (29, 26, 11), (68, 62, 10), (22, 23, 10), (96,
94, 10), (20, 21, 9). Over the whole matrix the mean |true − predicted| class
distance is **2.84** against **33.67** for uniformly spread errors — 12× tighter.
`synthetic_cifar` gives class *c* the frequency 2 + *c*, so adjacent labels are
adjacent patterns.

**CONTROL: 1% is below this recipe's own reproducibility.** He et al. report
ResNet-34 on ImageNet and their CIFAR results on CIFAR-10; ResNet-34 on CIFAR-100
is not in the paper, and the ~76–78% figures in circulation come from third-party
recipes. What *can* be measured: re-running the best configuration with nothing
changed but the seed gives **0.8050** against 0.8738 — a move of **0.0687**, seven
times the tolerance the exercise asks to match.

At full scale: `python train.py --data cifar100 --arch resnet34 --epochs 200` is
~2.5 h on one A10G (~$0.40/h spot), so ~$1 a run and ~$6 for this six-point sweep.

### A note on `ex03`'s length

At 148 lines of code `ex03` is over D14's 120-line target and under its 150-line
hard ceiling. The overrun is a ResNet-34 builder, a six-point sweep, a CSV
round-trip and a confusion analysis — four deliverables the exercise names
separately.
