<!-- generated:start -->
# 04-computer-vision / 03-cnns-lenet-to-resnet

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/03-cnns-lenet-to-resnet/) · upstream spec
`phases/04-computer-vision/03-cnns-lenet-to-resnet/docs/en.md`

```bash
uv run demo practice run 03-cnns-lenet-to-resnet --ex 1
uv run demo explain 03-cnns-lenet-to-resnet --ex 1
uv run pytest demos/phases/04-computer-vision/03-cnns-lenet-to-resnet
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Count parameters by hand for `TinyResNet` layer by layer. Compare against `sum(p.numel… | code | T1 | `ex01_param_budget_by_layer.py` |
| 2 | (Medium) Implement the Bottleneck block (1x1 -> 3x3 -> 1x1 with skip) and use it to build a R… | code | T1 | `ex02_bottleneck_resnet50.py` |
| 3 | (Hard) Remove the skip connection from `BasicBlock`, train a 34-block "plain" network and a 3… | code | T1 | `ex03_plain_vs_residual_deep.py` |
<!-- generated:end -->

## Answers

### 1 — Where TinyResNet's 2.8M parameters go

Written as a closed-form predictor and checked group by group, not just on the
total — a matching grand total can hide two cancelling mistakes.

| group | hand count | `sum(p.numel())` | off by |
|---|---:|---:|---:|
| `stem` | 928 | 928 | 0 |
| `layer1` | 37,120 | 37,120 | 0 |
| `layer2` | 131,712 | 131,712 | 0 |
| `layer3` | 525,568 | 525,568 | 0 |
| `layer4` | 2,099,712 | 2,099,712 | 0 |
| `head` | 2,570 | 2,570 | 0 |
| **total** | **2,797,610** | **2,797,610** | **0** |

**ANSWER: the convs, by three orders of magnitude** — 2,790,240 (**99.74%**)
against BatchNorm's 4,800 (0.17%) and the classifier head's 2,570 (0.09%).

**FINDING: "the convs" really means "the last group".** `layer4` alone is
**75.1%** of the network — more than the other five children combined.

**MECHANISM: parameters quadruple per group while the compute stays flat.**
`layer1` → `layer4` is **57×** the parameters but **0.89×** the
multiply-accumulates (37.7M → 33.6M): doubling the channels costs 4× the weights,
and the stride-2 that comes with it divides H·W by 4. The parameter budget is not
where the compute is.

**CONTROL: LeNet-5 gives the opposite answer**, putting **95.8%** of its 61,706
parameters in its fully-connected head. So "convs dominate" is a property of
`AdaptiveAvgPool2d(1)`, not of CNNs.

**CONTROL: `sum(p.numel())` is not the checkpoint size.** The BatchNorms carry a
further **4,820** non-learnable numbers — more than their learnable ones — that no
parameter count sees and every checkpoint still ships.

### 2 — A bottleneck ResNet-50 for CIFAR

ResNet-50's `[3, 4, 6, 3]` layout and 4× expansion, but on a CIFAR stem: the
ImageNet 7×7/stride-2 conv and max-pool would discard three quarters of a 32×32
image before the first block, so the stem is 3×3/stride-1 and the first group
keeps stride 1.

| network | parameters | conv layers |
|---|---:|---:|
| `TinyResNet` | 2,797,610 | 20 |
| bottleneck `[3, 4, 6, 3]` | **23,520,842** (8.41×) | 53 |
| the same, widths ÷ 3 | 2,605,193 (0.93×) | 53 |

**ANSWER: it is a residual block, proved rather than asserted.** Zero the final
BatchNorm's weight and bias and the whole 1×1 → 3×3 → 1×1 branch vanishes; the
block output then equals `relu(shortcut(x))` to **exactly 0.0**.

**FINDING: at the same input and output width the bottleneck is 16.8× cheaper** —
`BasicBlock(256 → 256)` costs 1,180,672 parameters, `Bottleneck(256 → 64 → 256)`
costs 70,400 for the same interface.

**MECHANISM: the only 3×3 runs at a quarter width.** Inside that block the weights
split 1×1 **16,384** / 3×3 **36,864** / 1×1 **16,384**. The 3×3 is 9·(C/4)² rather
than 9·C² = 589,824, and that 16× saving pays for both projections many times over.

**CONTROL: most of the 8.41× is width, not the block.** Divide every width by 3
and the same 53-layer network fits inside TinyResNet's budget — so "ResNet-50 is
bigger" is a statement about channels.

### 3 — Plain-34 against ResNet-34, and against its shallower twin

The exercise's last sentence names a different comparison from the one before it.
He et al.'s **Figure 1** is plain-20 against plain-**56** — deep plain against
*shallow plain*; plain-34 against ResNet-34 is Figure 4. Reproducing "higher loss
than its shallower twin" therefore needs a third network, so all three are trained
from one seed on one schedule. CIFAR-10 is replaced by a deterministic 384-image
teacher-labelled set at 12×12 so the run finishes on a CI core; the claim is about
optimisation, not about CIFAR, and it survives the swap.

Training loss, 16 epochs of SGD (lr 0.05, momentum 0.9, batch 64):

```
plain4    2.34  2.27  2.21  2.09  1.97  1.87  1.78  1.71  1.59  1.48  1.40  1.30  1.17  1.04  0.91  0.77
plain34   2.32  2.30  2.28  2.28  2.28  2.28  2.28  2.27  2.28  2.28  2.28  2.28  2.28  2.28  2.27  2.28
res34     3.69  3.11  2.48  2.35  2.29  2.27  2.25  2.22  2.20  2.16  2.15  2.04  2.00  2.05  1.91  1.77
```

**ANSWER: Figure 1 reproduced.** The 34-block plain net ends at **2.275** against
the 4-block plain net's **0.77** — 3× worse with 8.3× the parameters.

**ANSWER: the skip is the whole difference.** Restoring `+ x` — same depth, same
159,482 parameters, same seed, same schedule — ends at **1.766**, recovering
**34%** of the gap the depth opened. It also *starts* worse (3.69 against 2.32),
because the skip adds the branch's variance to the identity path, so the gain is in
the optimisation and not the initialisation.

**FINDING: the plain net is stalled, not merely slow.** Over the whole run
plain-34 moved **0.043** while plain-4 moved **1.571**. More epochs is not what it
is missing.

**MECHANISM: without the skip the two ends of the net see different scales.** At
initialisation the stem's gradient norm divided by the head's is **3.1e+04** for
plain-34, against 0.70 for plain-4 and 0.25 for residual-34. One learning rate
cannot suit both ends of the plain network.

**CONTROL: capacity is not the problem — the deep net can express the shallow
one.** Load trained plain-4 into a 34-block plain net and set the other 30 blocks
to identity (delta kernels, unit BatchNorm) and the loss reproduces to **2.1e-06**.
That solution exists in the 34-block weight space and SGD does not find it. This is
He et al.'s argument, run rather than quoted.

At full scale the exercise is `python train.py --arch plain34 --epochs 10` on
CIFAR-10 with 34 blocks at width 64: ~35 min on one A10G, about **$0.25** for the
pair at spot pricing.

### A note on `ex03`'s length

At 146 lines of code `ex03` is over D14's 120-line target and under its 150-line
hard ceiling. The overrun is three trained networks, a gradient probe and the
identity graft — each carrying a claim of its own, and none reducible to prose.
