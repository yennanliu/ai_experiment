<!-- generated:start -->
# 04-computer-vision / 18-open-vocab-clip

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/18-open-vocab-clip/) · upstream spec
`phases/04-computer-vision/18-open-vocab-clip/docs/en.md`

```bash
uv run demo practice run 18-open-vocab-clip --ex 1
uv run demo explain 18-open-vocab-clip --ex 1
uv run pytest demos/phases/04-computer-vision/18-open-vocab-clip
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Use a pretrained OpenCLIP ViT-B/32 and do zero-shot classification on CIFAR-10 with th… | code | T1 | `ex01_openclip_zero_shot_chance_floor.py` |
| 2 | (Medium) Compare single-template ("a photo of a {}") vs 80-template averaged embeddings on th… | code | T1 | `ex02_template_averaging_gap.py` |
| 3 | (Hard) Build a zero-shot image retrieval index: embed 1,000 images with CLIP, build a FAISS i… | code | T1 | `ex03_retrieval_recall_at_k_null.py` |
<!-- generated:end -->

## Answers

All three exercises are written against a pretrained checkpoint and a downloaded
dataset, and **this repo has neither**. `open_clip`, `clip`, `transformers` and
`faiss` are in no dependency group, `torchvision.datasets.CIFAR10(download=False)`
raises, and no weights may be fetched. So the first thing each solution does is
record the absence as a measurement rather than route around it, and the rest is
built on the lesson's own `TwoTower`, `clip_loss` and `zero_shot_classify` over a
CIFAR-10-shaped stand-in. What comes back is more interesting than the numbers the
exercises predict: **two of the three headline metrics cannot be read at all** —
one because it saturates, one because it measures an identity the data does not
carry — and the lesson's own sanity check contains a claim that is false at the
lesson's own initialisation.

### 1 — `ex01_openclip_zero_shot_chance_floor.py`

**ANSWER: 85-90% is not reproducible here, because neither half of the setup
exists.** Every CLIP package the exercise implies is missing, and the dataset
refuses to appear without a download:

| probe | result |
|---|---|
| `import open_clip` | `ModuleNotFoundError: No module named 'open_clip'` |
| `import clip` | `ModuleNotFoundError: No module named 'clip'` |
| `import transformers` | `ModuleNotFoundError: No module named 'transformers'` |
| `CIFAR10(download=False)` | `RuntimeError: Dataset not found or corrupted` |

The 85-90% band is reported for OpenCLIP ViT-B/32 on LAION-2B; it is not measured
here. The stand-in is 10 classes of Lesson 4's `synthetic_cifar`, standardised and
pushed through a frozen seeded random projection into the 128-d pre-extracted
image features the lesson's image tower expects, with one 64-d unit vector per
class where a text tower would sit.

**FINDING: an untrained zero-shot head sits at 1/C, and half of the inits land
*under* it.** Eight fresh `TwoTower` inits, scored by the lesson's own
`zero_shot_classify` on 200 held-out images:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top-1 | 0.085 | 0.065 | **0.005** | 0.155 | 0.095 | 0.120 | 0.115 | 0.110 | **0.094** |
| classes ever named | 5 | 4 | 6 | 7 | 4 | 7 | 4 | 6 | of 10 |

**MECHANISM:** chance is 1/10 = 0.100, and 4 of the 8 fall below it. Both towers
start as near-arbitrary linear maps, so one or two class prompts win almost every
image — and a head that never names a class cannot score that class's images.
That is why the worst init reads 0.005 rather than 0.100.

**FINDING: trained, top-1 saturates at 1.000 and ranks nothing.** Top-1 and the
mean top-1-minus-runner-up cosine margin along the training ladder (seed 0):

| steps | 0 | 2 | 5 | 10 | 25 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| top-1 | 0.085 | 0.760 | **1.000** | 1.000 | 1.000 | 1.000 |
| margin | 0.026 | 0.067 | 0.119 | 0.257 | 0.479 | **0.588** |

All 8 seeds finish at 1.000 for a spread of **0.000**, overshooting the exercise's
own band — the stand-in task is far easier than CIFAR-10. So the discriminating
quantity is the margin, which climbs from 0.019-0.041 at step 0 to **0.574-0.608**
by step 100, long after accuracy stopped moving.

**MECHANISM: top-1 cannot see the temperature at all.** Scaling the similarity
matrix by 0.01, 1.0, 14.285 and 100.0 — a span bracketing the lesson's own
`logit_scale.exp()` ≈ 14.3 — leaves all 200 predictions bit-identical, because a
positive scalar cannot reorder a row. `logit_scale` reaches zero-shot accuracy
only through the loss, which ends at 1.886-2.037.

**CONTROL: the chance floor is exactly 1/C.** Relabelling the trained head's
predictions through 2,000 random permutations of the 10 prompts scores
**0.10075** against 1/10 = 0.10000. Every image has probability 1/C of being
relabelled correctly whatever the class balance, so this is the null that every
zero-shot number in the lesson has to be read against.

### 2 — `ex02_template_averaging_gap.py`

**ANSWER: 80 templates beat one by 0.095-0.295 top-1 — and the real templates
cannot be built.** `open_clip` is absent, so the literal string `"a photo of a
{}"` cannot be tokenised, and the lesson's text tower consumes 64-d features
rather than tokens. A template is therefore modelled as what a template *is* in
feature space: a per-template direction shared across all classes (style 0.4)
plus per-(class, template) content noise (0.3), added to the class prototype
before L2-normalising.

| K templates | 1 | 2 | 4 | 8 | 20 | 80 |
|---|---:|---:|---:|---:|---:|---:|
| top-1, bank 11 | 0.840 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| top-1, bank 12 | **0.705** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| top-1, bank 13 | 0.905 | 0.840 | 1.000 | 1.000 | 1.000 | 1.000 |
| margin, bank 11 | 0.174 | 0.156 | 0.243 | 0.288 | 0.313 | **0.332** |

OpenAI report 1-3% top-1 for the real 80 on ImageNet; that figure is not measured
here.

**FINDING: the accuracy gap closes at K = 4 — 76 of the 80 templates buy nothing
measurable.** All three banks reach 1.000 by the K=4 rung and stay there, so top-1
ranks nothing past it. The margin keeps moving, ending at **0.324-0.332** against
**0.637** for a single noiseless prompt: averaging recovers all of the accuracy
and about half of the margin.

**MECHANISM: it is 1/sqrt(K) variance reduction and nothing else.** Per-dimension
RMS of the mean of K content-noise draws, against the prediction:

| K | 1 | 2 | 4 | 8 | 20 | 80 |
|---|---:|---:|---:|---:|---:|---:|
| measured | 0.30651 | 0.21800 | 0.15095 | 0.10923 | 0.06742 | 0.03475 |
| 0.3/sqrt(K) | 0.30000 | 0.21213 | 0.15000 | 0.10607 | 0.06708 | 0.03354 |
| ratio | 1.022 | 1.028 | 1.006 | 1.030 | 1.005 | **1.036** |

Every ratio is within 3.6% of 1, and 80 draws cut the nuisance component **8.9x**,
leaving the class prototype behind.

**CONTROL: the gain is per-class noise cancellation, not style removal.** With
content noise switched off and only a shared style direction of 1.2 — three times
the main arm's — the single-template arm still scores **0.935-0.995**, a cost of at
most **0.065** against the **0.295** that per-class noise costs. Adding the same
vector to every class barely reorders an argmax. And at style and noise both zero
the 80 bank rows are **bitwise identical** and the K=1 to K=80 gap is exactly
**0.000** at 1.000 top-1 — so no part of the gain is extra information from extra
wording.

**CONTROL: averaging embeddings is not averaging features, and top-1 cannot tell
them apart.** CLIP averages *after* the tower. Doing it before instead gives class
tables only **0.9155-0.9623** cosine apart per class, because `text_proj` is a ReLU
MLP rather than a linear map. Both score **1.000** top-1, so accuracy cannot rank
the two recipes; their margins differ, 0.332 after the tower against 0.378 before.
(`emb` and `txt_in` are both 64 in the lesson's `TwoTower`, so feeding an
already-averaged embedding table back through `encode_text` type-checks and
silently answers the wrong question — the trap this exercise had to step around.)

### 3 — `ex03_retrieval_recall_at_k_null.py`

**ANSWER: recall@5 comes back at 0.000-0.050 — and there is no FAISS to build the
index with.** `faiss` and `open_clip` are both `ModuleNotFoundError`, so there is
no index library, no checkpoint, and no way to "write a query by hand". The index
half loses nothing: a FAISS `IndexFlatIP` over L2-normalised vectors is exhaustive
inner-product search, and this is asserted rather than assumed.

| structural claim | measured |
|---|---|
| `topk(5)` == full `argsort` prefix | `True`, bitwise |
| embedding norms | 0.9999999 - 1.0000001 |
| towers trained on `clip_loss`, final loss | 1.918 - 1.970 |
| instance recall@5, 3 seeds | 0.05, **0.00**, 0.05 (mean **0.0333**) |

Zero or one query out of 20 finds its own image.

**FINDING: that number blames the metric, not the tower.** The same retrieval puts
the query's *class* in the top 5 for **20 of 20** queries on all three seeds, while
the paired image sits at median rank **56-63 of 1,000**. Every gallery image shares
its class — and therefore its caption — with 99 others, and `clip_loss` never asked
the tower to separate them: instance recall is measuring an identity the data does
not carry. Twenty queries also quantise recall@5 to multiples of 0.050, so the
exercise's own protocol cannot resolve anything finer.

**CONTROL: under random embeddings recall@k is exactly k/N.** Two hundred
independent draws of 1,000 random unit vectors and 20 random queries, 4,000
queries in all:

| k | 1 | 5 | 10 | 50 |
|---|---:|---:|---:|---:|
| measured recall@k | 0.00125 | 0.00600 | 0.01000 | 0.04950 |
| k/N | 0.00100 | 0.00500 | 0.01000 | 0.05000 |

Worst deviation **0.00100**. The trained tower's mean 0.0333 at k=5 is 7x that
floor and still **96.7% misses** — which is the honest way to read any retrieval
score over a gallery of near-duplicates, and the right null for every zero-shot
number the lesson quotes.

**MECHANISM: the temperature scales logits exactly and moves no retrieval result.**
Top-5 is bit-identical across scales 0.01, 1.0, 14.285 and 100.0, which bracket
this tower's `logit_scale.exp()` of 13.838. The scaled matrix stays a
distribution — softmax rows sum to **0.9999998-1.0000005** — and swapping the two
towers transposes the similarity matrix to exactly **0.0**, so `clip_loss` agrees
both ways at **1.010219**. The symmetry the lesson claims for its objective is a
structural identity, not an approximation.

**CONTROL: the lesson's own "loss should be close to log(N)" is false at the
lesson's own initialisation.** `docs/en.md` says the untrained loss should be near
`log(N)`. It is — but only at temperature 1:

| N | 8 | 32 | 64 | 256 |
|---|---:|---:|---:|---:|
| `clip_loss` at `logit_scale` = e^2.6592 | **3.167** | 4.011 | 4.974 | 6.480 |
| same embeddings at scale 1.0 | 2.110 | 3.456 | 4.157 | 5.547 |
| log(N) | 2.079 | 3.466 | 4.159 | 5.545 |

Scale 1.0 matches log(N) to **0.031**, while the shipped `ln(1/0.07)` init runs
**1.16-1.52x** above it. Sharpening the softmax raises a random model's loss above
chance rather than leaving it there, so the lesson's sanity check will not read
2.08 for the model the lesson itself builds — it reads 3.167.
