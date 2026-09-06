<!-- generated:start -->
# 04-computer-vision / 05-transfer-learning

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/05-transfer-learning/) · upstream spec
`phases/04-computer-vision/05-transfer-learning/docs/en.md`

```bash
uv run demo practice run 05-transfer-learning --ex 1
uv run demo explain 05-transfer-learning --ex 1
uv run pytest demos/phases/04-computer-vision/05-transfer-learning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Train a `ResNet18` as a linear probe (backbone frozen) and as a full fine-tune on the… | code | T1 | `ex01_probe_vs_finetune_gap.py` |
| 2 | (Medium) Introduce a bug on purpose: set `base_lr = 1e-1` on the backbone stage instead of th… | code | T1 | `ex02_stage_lr_divergence.py` |
| 3 | (Hard) Take a medical imaging dataset (e.g. CheXpert-small, PatchCamelyon, or HAM10000) and c… | code | T1 | `ex03_pretrain_regimes_crossover.py` |
<!-- generated:end -->

## Answers

One thing about this lesson cannot be run as written, and saying so is most of
the work. The lesson's own `make_feature_extractor` and `make_fine_tune` ask
torchvision for `ResNet18_Weights.IMAGENET1K_V1`, and that is a **46 MB fetch of
`resnet18-f37072fd.pth`** which a hermetic test has no network for. Each solution
therefore rebinds one name — `ref.resnet18` — to build the same torchvision
ResNet-18 with `weights=None`, and everything else is the lesson's own code,
imported: the freeze loop, the head swap, `discriminative_param_groups`,
`freeze_bn_stats`, `train_and_eval`, `ArrayDataset`, `synthetic_dataset`. So the
graph and its initialisation are torchvision's real ones (11,689,512 parameters
at 1,000 classes, 11,181,642 with the lesson's 10-class head), and the weights
are random. **There are no pretrained features here** — which is the honest core
of exercises 1 and 3.

### 1 — probe vs finetune gap

Three epochs of the lesson's own `train_and_eval` on 300 synthetic-CIFAR images
at 32×32, 100 held out, at the lesson's own learning rates.

| arm | trainable | val accuracy | backbone weights moved | BatchNorm stats moved | head moved |
|---|---:|---:|---:|---:|---:|
| linear probe (`base_lr` 0.03) | 5,130 | **1.000** | **0.000** | 0.880 | 1.598 |
| full fine-tune (`base_lr` 0.001) | 11,181,642 | **1.000** | 2.70e-03 | — | 0.271 |
| probe + `freeze_bn_stats` | 5,130 | **1.000** | 0.000 | 0.000 | 2.123 |

(Movement is relative L2, ‖ΔW‖ / ‖W₀‖, over the named slice of the state dict.)

**ANSWER: 1.000 and 1.000.** The two arms are indistinguishable, on 2,180× the
trainable parameters. The exercise asks which gap tells you the features transfer
well and which tells you they do not; with a gap of **+0.000** neither does.

**FINDING: the question cannot be asked here at all.** Both arms start from
random weights, because the checkpoint is a download a test cannot make. A
probe-vs-fine-tune gap is a statement about *pretrained* features; exercise 3
pretrains a checkpoint of its own and answers it there.

**FINDING: "backbone frozen" froze the weights and nothing else — and here it did
not matter.** After the probe run the backbone parameters have moved **exactly
0.0** (`requires_grad = False` held) while the **9,600 BatchNorm running
statistics moved 0.880**. Pinning those with the lesson's own `freeze_bn_stats`
holds them at 0.0 and the arm **still scores 1.000**, its head moving 2.123
instead of 1.598. The adaptation channel is real and it is invisible to
`requires_grad`; this task simply does not need it, and the linear head absorbs
the difference in feature scale.

**MECHANISM: the LR ladder makes "full fine-tune" a probe with a slower head.**

| group | `conv1_bn1` | `layer1` | `layer2` | `layer3` | `layer4` | `fc` |
|---|---:|---:|---:|---:|---:|---:|
| LR at `base_lr=1e-3` | 2.43e-06 | 8.10e-06 | 2.70e-05 | 9.00e-05 | 3.00e-04 | 1.00e-03 |

`discriminative_param_groups` decays by 0.3 per stage, so the stem runs at
`0.3⁵ = 0.00243` of the head. Measured over the run, the head moves **0.271** and
the whole backbone **2.70e-03** — **100× less**. The 11.2M-parameter arm is a
5,130-parameter arm with a slower head.

**CONTROL: three numbers per image get most of the way.** A logistic regression
on nothing but each image's three per-channel means scores **0.850** against the
ResNet's 1.000. `synthetic_dataset` gives class *c* its own colour centre, so the
label is largely in the mean colour. On this dataset no probe-vs-fine-tune
comparison can separate good features from an easy task.

### 2 — stage lr divergence

Sixteen-step bursts on 200 images at 24×24, then a per-stage sweep of six steps
with one group in the optimiser at a time. Divergence is defined up front: a peak
loss above **3× the first step's**, or a non-finite loss.

| arm | first | last | peak |
|---|---:|---:|---:|
| the bug as written — ladder reversed, `conv1_bn1` at 1e-1, `fc` at 2.43e-04 | 2.76 | **0.61** | 2.76 (1.00×) |
| `discriminative_param_groups(base_lr=1e-1)` — `fc` at 1e-1 | 2.76 | 21.74 | **26.6 (10×)** |
| `discriminative_param_groups(base_lr=1e-3)` — the lesson's own setting | 2.76 | **0.82** | 2.99 |

**ANSWER: the bug does not explode. It trains.** Putting 1e-1 on the backbone
stage takes the loss **2.76 → 0.61**, and its peak *is* its first step. The
exercise's premise is backwards for this architecture. (A flat 1e-1 on *every*
parameter, run as the second reading of the instruction while these files were
written, does explode — peak **31.0**, 11× its first step — but that is the head
being included, not the backbone.)

**CONTROL: the helper is not the recovery.** `discriminative_param_groups` is
what the exercise says to apply to recover, and at `base_lr = 1e-1` it **peaks at
26.6**, because its ladder hands the *largest* LR to `fc`. The same helper at the
lesson's own `base_lr = 1e-3` takes the loss to 0.82. The recovery is the
`base_lr`, not the grouping — the helper redistributes a budget, it does not
shrink one.

**ANSWER: the LR at which each stage starts diverging.**

| stage | `conv1_bn1` | `layer1` | `layer2` | `layer3` | `layer4` | `fc` |
|---|---|---|---|---|---|---|
| diverges at | never | never | never | never | never | **0.1** |

Never means at no LR on the grid 0.01, 0.1, 1, 10 — a hundred times past the
threshold that destroys the head.

**MECHANISM: BatchNorm makes conv weights scale-free; the head has no BatchNorm.**

| LR on `fc` | 0.1 | 1 | 10 |
|---|---:|---:|---:|
| peak loss | 11.3 | 121.2 | 1213.4 |

That is **10.7× then 10.0× per decade** — the signature of a loss that is linear
in the size of the step, not one that has left its basin. The logits are linear
in `fc`'s weights, so a large step passes straight into the loss. Scaling the
weights of a convolution that is followed by BatchNorm changes nothing the next
layer sees; the step only rotates them. At LR 10 the worst backbone peak is
**2.88**, 0.24% of `fc`'s.

**CONTROL: tolerance is not indifference.** With only that stage in the optimiser
at LR 10, the worst of `layer2`, `layer3` and `layer4` ends at **1.23**, down
from 2.76. They absorb an LR a hundred times past the one that destroys `fc`
*and still learn*.

### 3 — pretrain regimes crossover

No medical dataset and no ImageNet checkpoint can be reached from a hermetic
test, so both are substituted and named as substitutes. The source domain is the
lesson's own `synthetic_dataset` (200 images, three epochs, **1.000** source
validation accuracy), standing in for ImageNet; the target is scikit-learn's
bundled 8×8 handwritten digits upsampled to 24×24 RGB — low-resolution,
low-texture and domain-far, of the sort a medical modality is. Two epochs per
arm, 128 held out, and the **step count is identical across regimes** at each
size, so an accuracy difference cannot be a compute difference.

| n | steps | (a) frozen probe | (b) pretrained fine-tune | (c) scratch | (b) − (c) |
|---:|---:|---:|---:|---:|---:|
| 60 | 8 | 0.352 (1.2 s) | **0.383** (1.5 s) | 0.289 (1.5 s) | **+0.094** |
| 240 | 30 | 0.727 (3.1 s) | **0.836** (4.3 s) | 0.812 (4.4 s) | +0.023 |
| 480 | 60 | 0.750 (5.6 s) | **0.906** (7.8 s) | 0.883 (8.0 s) | +0.023 |

Trainable parameters: 5,130 for (a), 11,181,642 for (b) and (c). Wall clocks are
from one run on two CPU threads and move between runs; the step counts, parameter
counts and accuracies do not.

**ANSWER: scratch is competitive almost immediately — the crossover is below
n = 240, not above it.** Taking "competitive" to mean a gap under 0.05, the
pretrained fine-tune is ahead only at n = 60 (**+0.094**); by n = 240 the gap is
**+0.023**, which is *three* of the 128 held-out images at a resolution of 0.008
per image. The exercise's framing expects a large-N crossover; on a domain-far
target there is barely a gap to close.

**FINDING: the frozen probe plateaus and is overtaken.** It gains only **+0.023**
over the last doubling of data and ends **0.133 behind scratch**. A frozen
backbone caps the hypothesis class at whatever a linear map of its features can
express; the other two regimes keep buying capacity with data. Every regime gains
at least **+0.398** from n = 60 to n = 480 — the target set, not the checkpoint,
is what all three of them want.

**MECHANISM: a domain-far source buys conditioning, and ResNet-18 already starts
conditioned.** The checkpoint was trained on sinusoidal colour textures and moved
onto handwritten digits; nothing in it recognises a digit. What can carry over is
BatchNorm statistics and a sensibly scaled initialisation — and torchvision's
Kaiming init already supplies the second, which is why the benefit is +0.094 at
n = 60 and ~0.02 thereafter. This is the same shape as the medical-transfer
literature's finding for real ImageNet weights on real scans.

**A retraction worth recording.** An earlier draft of these files could not
import torchvision at all (it was not yet in the `vision` dependency group) and
rebuilt the ResNet-18 graph from the curriculum's own lesson-03 `BasicBlock`.
That graph is identical in shape and parameter count, but it inherits PyTorch's
*default* conv initialisation rather than torchvision's Kaiming-normal
`fan_out`, and against that badly-conditioned start the numbers told a very
different story: scratch scored **0.109** at n = 60 and the pretrained arm
**0.578**, a textbook transfer curve with a crossover between n = 240 and n = 480,
and in exercise 1 `freeze_bn_stats` collapsed the probe from 1.000 to **0.220**.
Both of those "findings" were artefacts of the initialisation, not of transfer.
Running against the real torchvision graph removed them.

**CONTROL: the run the exercise actually asks for, priced.**

```
train.py --data ham10000 --arch resnet18 --pretrained --epochs 20 --img 224
```

HAM10000 is 10,015 images. This ResNet-18 measures **≈25 images/s at 224×224 on
two CPU threads** here (`solve()` times three real forward-backward steps at that
resolution), so 20 epochs is **≈2.2 CPU-hours** — and one to two orders of
magnitude less on a single data-centre GPU, i.e. minutes and a few dollars of
spot time. Nothing above used ImageNet weights or a medical image, and the
substitution is the finding, not a footnote.
