<!-- generated:start -->
# 04-computer-vision / 14-vision-transformers

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/14-vision-transformers/) · upstream spec
`phases/04-computer-vision/14-vision-transformers/docs/en.md`

```bash
uv run demo practice run 14-vision-transformers --ex 1
uv run demo explain 14-vision-transformers --ex 1
uv run pytest demos/phases/04-computer-vision/14-vision-transformers
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Print the shapes of every intermediate tensor for a forward pass through the tiny ViT… | code | T1 | `ex01_vit_token_shape_ledger.py` |
| 2 | (Medium) Fine-tune a pretrained `timm` ViT-S/16 on the synthetic-CIFAR dataset from Lesson 4.… | code | T1 | `ex02_vit_vs_resnet_small_data.py` |
| 3 | (Hard) Implement MAE pretraining for the tiny ViT: mask 75% of patches, train the encoder + a… | code | T1 | `ex03_mae_masking_linear_probe.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is correct — every shape it prints is the shape it
claims, and its "About 2.8M parameters" is 2,822,602. What the three exercises
find is that **the ViT's interesting properties are exact identities, while the
two experiments the lesson proposes cannot be run as written**: `timm` is not
installed and no weights may be downloaded, and the synthetic dataset both
experiments are pointed at is linearly separable in raw pixel space, so the
accuracy each one asks you to report is pinned at 1.000 before any network exists.

### 1 — the ledger is right, and every piece of it is welded to the next

`ex01_vit_token_shape_ledger.py` — 137 code lines, 5 checks.

**ANSWER: the shapes hold exactly, at every batch size the exercise leaves open.**
Run at `N` in 1, 2, 3 and 8 through the lesson's own `ViT`:

| stage | shape |
|---|---|
| input | `(N, 3, 64, 64)` |
| patches | `(N, 16, 192)` |
| with CLS | `(N, 17, 192)` |
| classifier input | `(N, 192)` |
| logits | `(N, 10)` |

All four rows match. The 16 is not a free choice — it is `(64 // 16) ** 2` — and
the model holds **2,822,602** parameters, of which **2,669,184** are the six
blocks and only **3,456** are the CLS token and the position table together.

**MECHANISM: the "first conv" is exactly a per-patch linear map.**
`F.unfold(x, 16, stride=16)` gives 16 columns of `3*16*16 = 768` pixels;
multiplying them by `proj.weight.reshape(192, -1).T` plus the bias reproduces
`vit.patch(x)` to **1.1e-06**. That identity is why the patch block holds
**147,648 = 192 × (768 + 1)** parameters, and why kernel size and stride are the
same number.

**MECHANISM: position lives entirely in `pos_embed`.** Attention and the MLP are
both permutation-equivariant, so with the position table zeroed the CLS logits are
a set function of the patch tokens:

| patch tokens shuffled | change in CLS logits |
|---|---:|
| `pos_embed` zeroed | **< 1e-06** |
| `pos_embed` as initialised | **0.0218** |
| (logit std for scale) | 0.460 |

The real table moves the logits by only **4.7%** of their own spread, because
`trunc_normal_(std=0.02)` starts positions at std **0.0201**. At initialisation a
ViT barely knows where anything is; that has to be learned.

**FINDING: the ledger is welded to 64×64, and interpolation repairs it.** A 32×32
image is legal for every layer the model contains, and still raises
`RuntimeError: The size of tensor a (5) must match the size of tensor b (17)` — the
17-row `nn.Parameter` cannot broadcast onto 5 tokens. Bicubic-resizing its 4×4
patch grid to 2×2 with the CLS row set aside restores a coherent ledger,
`(2, 4, 192) → (2, 192) → (2, 10)`, and resizing 4×4 to itself is the identity to
**0.0**, so the repair costs nothing at the native resolution.

**CONTROL: `.eval()` is not load-bearing here.** The same batch through
`vit.eval()` and `vit.train()` differs by exactly **0.0**. `Block` defaults to
`dropout=0.0` and `ViT` never overrides it, and there is no BatchNorm anywhere —
unlike every CNN lesson in this phase, where forgetting `.eval()` changes the
answer.

### 2 — the comparison the exercise asks for cannot be made, and the one that can points at time

`ex02_vit_vs_resnet_small_data.py` — 138 code lines, 6 checks.

**CONTROL: "pretrained" is not available.** `importlib.import_module("timm")`
gives `ModuleNotFoundError: No module named 'timm'`; timm is in no dependency
group in `pyproject.toml`, and downloading ImageNet weights is out of scope, so
ResNet-18 is built with `weights=None`. **Both arms start from random init**, and
what is compared is architecture plus optimiser, not transfer.

**ANSWER: both reach the same accuracy, and the ViT gets there roughly 6× faster.**
32 AdamW steps (lr 3e-4, batch 48) on 480 of Lesson 4's `synthetic_cifar` images,
scored against 120 held out, two seeds each:

| arm | parameters | optimiser time (seed 0 / 1) | final accuracy | first step at 1.000 |
|---|---:|---:|---:|---:|
| ViT (`image_size=32, patch_size=8`) | 2,712,010 | **1.14s / 1.13s** | 1.000 / 1.000 | 12 / 20 |
| ResNet-18 (`weights=None`) | 11,181,642 | **7.53s / 7.43s** | 1.000 / 1.000 | 20 / 8 |

Wall-clock figures are from one machine and will move, so the **6.5×** above is
reported rather than required: the check asserts only that the ratio exceeds
**2×**, which is the widest bound that still carries the finding.

**FINDING: "final accuracy" ranks nothing.** All four runs end at 1.000, a spread
of **0.000**. `synthetic_cifar` gives class `c` the spatial frequency `2 + c` on a
fixed 32×32 grid, so the classes are separated before any network sees them. The
comparison needs a dataset both models can still get wrong.

**FINDING: the obvious fallback statistic is noise.** Sampling validation accuracy
every 4 steps, the first step at 1.000 is ViT **12 / 20** against ResNet-18
**20 / 8** — the winner changes with the seed. Both curves are non-monotone (ViT
`0.73 0.88 1.00 0.92 1.00 1.00 1.00 1.00`, ResNet-18
`0.76 0.89 0.87 0.93 1.00 1.00 1.00 1.00`), so any "converges faster" claim taken
from one seed is a coin flip.

**MECHANISM: wall clock does not follow parameter count.** ResNet-18 carries
**4.1×** the ViT's parameters and costs **6.5×** the time per step — the ratio has
the wrong sign for a parameter explanation. ResNet-18 is 20 sequentially dependent
convolutions over feature maps that start at 32×32; the tiny ViT is 19
`nn.Linear` modules over 17 tokens of width 192, which is the shape BLAS likes.
Parameters price memory, not latency.

**CONTROL: no network is under test at all.** A multinomial logistic regression on
the 3072 un-featurised pixels of the same 480 training images scores **1.000** on
the same validation split. Whatever either network learned, the task did not
require it.

### 3 — MAE trains, and the metric the exercise grades it with cannot see it

`ex03_mae_masking_linear_probe.py` — 146 code lines, 5 checks. Neither
`code/main.py` nor `docs/en.md` ships an MAE, so one is built by **subclassing the
lesson's own `ViT`** — its `patch`, `cls_token`, `pos_embed`, `blocks` and `ln` are
the encoder unchanged, re-entered through an `encode` that gathers the surviving
tokens first; the inherited `head` is dead weight and is never called. The
additions are a 96-wide one-block decoder fed a shared mask token, with the loss on
the hidden patches only.

**ANSWER: linear-probe accuracy before and after is the same number on every seed.**

| measurement (3 seeds) | before | after |
|---|---:|---:|
| frozen-CLS probe, 480 fit / 120 held out | **1.000** | **1.000** |
| masked-patch MSE | 2.354 – 2.606 | **0.185 – 0.217** |

The probe resolves nothing; the pretraining it was meant to grade plainly happened.

**FINDING: reconstruction is genuinely learned.** The loss covers the 12 hidden
patches only, and predicting the training set's mean patch for each of them costs
**1.3375** (pixel std 1.401). The trained model reaches **0.185 – 0.217**, at least
**6.2× below that floor**, on all three seeds.

**MECHANISM: the encoder provably never sees a masked pixel.** Masking 75% of 16
patches keeps 4, so the encoder runs **5 tokens (CLS + 4), not 17**, and because
the mask is an argsort permutation exactly **12** hide every time. Overwriting all
12 hidden patches with Gaussian noise and re-encoding moves the latent by
**0.0e+00** against a latent scale of 3.28 — there is no information path, so the
saving is real rather than cosmetic.

**CONTROL: the probe was saturated before any encoder existed.** The same logistic
regression on the 3072 raw pixels also scores **1.000** on the same split.
`synthetic_cifar` writes class `c` as the spatial frequency `2 + c`, which a linear
map reads off directly — so no pretraining recipe could show a gain on this
benchmark, and the exercise's before/after question needs a harder dataset rather
than a better MAE.

**CONTROL: only the hidden patches are supervised.** The identical forward pass,
scored on the 4 patches the encoder *was* handed instead of the 12 it was not,
gives **1.888 – 1.943** against **0.185 – 0.217** — at least **8.7× worse**. Those
outputs never enter the loss, so they stay untrained, which is also why a decoder
that simply copied its input could not cheat the objective.
