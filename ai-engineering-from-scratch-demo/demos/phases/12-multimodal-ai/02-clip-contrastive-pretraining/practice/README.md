<!-- generated:start -->
# 12-multimodal-ai / 02-clip-contrastive-pretraining

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/02-clip-contrastive-pretraining/) · upstream spec
`phases/12-multimodal-ai/02-clip-contrastive-pretraining/docs/en.md`

```bash
uv run demo practice run 02-clip-contrastive-pretraining --ex 1
uv run demo explain 02-clip-contrastive-pretraining --ex 1
uv run pytest demos/phases/12-multimodal-ai/02-clip-contrastive-pretraining
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement InfoNCE for a batch of 4 pairs by hand. Construct the 4x4 similarity matrix, run so… | code | T0 | `ex01_the_recipe_in_the_exercise_computes_half_the_loss.py` |
| 2 | SigLIP uses a bias parameter `b` in addition to temperature: `S'[i,j] = S[i,j]/tau + b`. What… | code | T0 | `ex02_the_bias_is_the_class_prior_in_log_odds.py` |
| 3 | Build a zero-shot classifier for cats vs dogs. Try two prompt templates: `a photo of a {class… | code | T0 | `ex03_the_ensemble_beats_the_mean_template_not_the_best_one.py` |
| 4 | Compute the communication cost of softmax InfoNCE vs sigmoid pairwise for a 512-GPU run at ba… | code | T0 | `ex04_the_wire_cost_is_identical_to_the_byte.py` |
| 5 | Read the OpenCLIP scaling-laws paper (arXiv:2212.07143, Cherti et al.). Reproduce their concl… | code | T0 | `ex05_the_law_is_model_independent_in_error_and_not_in_accuracy.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four of them import the lesson's own `main.py` and
measure it; exercise 5 has no code in the lesson to measure, so its constants
are transcribed from the OpenCLIP checkpoint cards and labelled as such, and
every assertion rests on ratios of those numbers rather than their values.

### 1 — the recipe in the exercise computes half the loss

The 4×4 matrix is written down rather than derived, so the hand calculation is
reproducible on paper. It is built around one generic caption — text 0 scores
4.0 against **every** image, the way "a photo" would:

```
        t0   t1   t2   t3
  i0 [ 4.0  0.0  0.0  0.0 ]
  i1 [ 4.0  3.0  0.0  0.0 ]
  i2 [ 4.0  0.0  3.0  0.0 ]
  i3 [ 4.0  0.0  0.0  3.0 ]
```

**ANSWER: the hand calculation and `infonce_loss` agree to 0.0.** Row softmax
diagonals 0.9479, 0.2619, 0.2619, 0.2619 → cross-entropies 0.0535, 1.3397,
1.3397, 1.3397; averaged with the column direction, **0.734559**.

**FINDING: the exercise's own recipe is 38.6% high.** "Pick out the diagonal,
compute cross-entropy" is the image→text direction alone — **1.0181**.
`infonce_loss` averages it with the column direction's **0.4510**. Half a
symmetric loss is not a symmetric loss.

**FINDING: the two directions disagree about which pairs are right.** At strict
argmax the rows retrieve **1 of 4** and the columns **3 of 4** — rows 1–3 rank
the generic caption above their own text, and column 0's four-way tie retrieves
nothing. The averaged number hides a 1-vs-3 split.

**FINDING: a batch of 4 has 1.386 nats of range, and the lesson uses 0.00045%
of it.** Column 0 is uniform, so its term is exactly log 4 — the most a 4-way
softmax can charge. The lesson's own aligned fixture at `tau=0.07` scores
**6.2e-06**. CLIP's 32,768 batch has **10.397** nats.

### 2 — the bias is the class prior in log-odds

| N | measured argmin `b` | `-log(N-1)` |
|---:|---:|---:|
| 4 | −1.15 | −1.099 |
| 8 | −1.97 | −1.946 |
| 16 | −2.72 | −2.708 |
| 32 | −3.45 | −3.434 |
| 64 | −4.15 | −4.143 |

**ANSWER: `b` is the positive rate in log-odds.** A batch of N has N positives
and N²−N negatives, so the prior is 1/N and its logit is −log(N−1). The
measured optimum tracks it to **0.051** — before any training.

**FINDING: without `b` the loss cannot see the imbalance.** At b = 0 the loss
is 0.696–0.711 for *every* N, all within 3% of log 2, because at initialisation
every pair sits at σ(0) = 0.5. The positive-to-negative ratio moves from 1:3 to
1:63 and the loss does not move; the negatives are **98.4%** of the pairs at
N = 64 and take the same gradient as the positives.

**FINDING: SigLIP's published `b = −10` is the logit of their batch.**
−log(32,767) = **−10.397**, 0.4 nats away. Setting b to its optimum cuts the
N = 64 loss from 0.6964 to 0.0811 — **88%** — without changing an embedding.

**FINDING: `tau` and `b` are not independent.** Sharpening tau from 1.0 to 0.07
moves the optimum a further **1.10–1.41** nats negative, which is why SigLIP
learns both.

### 3 — the ensemble beats the mean template, not the best one

The lesson ships no text encoder, so the text model is stated: a template is
its class prototype plus template-specific noise at ALPHA; a test image is its
prototype plus image noise at BETA. Both prototypes are the lesson's own
`make_fake_embedding(42)` and `(43)` — the seeds its zero-shot demo uses.

**ANSWER: yes, by 8 points — 91% against 83% and 80%** on 100 images at
ALPHA = BETA = 2.0, and 9.5 points above the mean of the two templates.

**FINDING: over the 15-cell sweep the ensemble is at or above the mean of its
singles in 15 of 15 cells, and above the better single in only 6.** Where the
templates disagree it drags the good one down: at ALPHA = 4, BETA = 2 the
singles are 51% and 73% and the ensemble is **64%**.

**FINDING: all of the gain is template variance.** At ALPHA = 0 the two
templates are the same vector and single and ensemble agree exactly — 98/98/98,
96/96/96, 93/93/93. A benchmark that varies only the images cannot measure
prompt ensembling.

**FINDING: in the lesson's own text model, applying a template destroys the
class.** `demo_prompt_ensemble` embeds a prompt as
`make_fake_embedding(sum(ord(c) for c in prompt))`. The resulting "a photo of a
cat" vector scores **−0.1254** against the lesson's cat prototype and
**+0.0046** against its dog prototype — nearer the wrong class.
`demo_zero_shot` sidesteps this by using the prototype *itself* as the text
vector, and `demo_prompt_ensemble` never classifies anything.

**FINDING: that seed is anagram-invariant.** `sum(ord(c))` gives "a photo of a
cat" and "a photo of a **act**" the same seed, **1401**, and byte-identical
embeddings — while "a photo of a dog" is 1403, two away, and independent.

### 4 — the wire cost is identical to the byte

At N = 32,768 on 512 devices, SigLIP SO400m's 1,152-wide bf16 embeddings:

| | per device |
|---|---:|
| InfoNCE all-gather, (P−1)/P · N · D · 2 | **75,350,016 B** |
| SigLIP §4 ring pass, (P−1) · (N/P) · D · 2 | **75,350,016 B** |

**ANSWER: the same number — 71.86 MiB per tower per step.** Those are the same
expression written twice.

**ANSWER: what differs is memory, by exactly P.**

| | logits held per device |
|---|---:|
| InfoNCE, 64 rows × 32,768 | **8 MiB** |
| sigmoid chunk, 64 × 64 | **16 KiB** |

Ratio **512** — the device count, not a property of either loss.

**FINDING: neither cost is O(N) against O(N²) at fixed hardware.** Measured
exponents between N = 8,192 and N = 131,072 at a fixed local batch of 64:

| quantity | exponent |
|---|---:|
| wire bytes | 1.00 |
| InfoNCE logits | 1.00 |
| sigmoid chunk | **0.00** |
| pair count | **2.00** |

The only O(N²) quantity is the pair count, and *both* losses have it. The
sigmoid's advantage is an exponent of 0, not of 1.

**FINDING: the lesson's own functions locate the coupling.** Instrumented,
`sigmoid_loss` calls `sigmoid` exactly N² times and `infonce_loss` calls
`log_sum_exp` exactly 2N times — one per row, one per column. The N² arithmetic
is shared; the 2N normalisers are what must be global, and half of them read
columns held on other devices.

### 5 — the law is model-independent in error, not in accuracy

Transcribed from the OpenCLIP checkpoint cards (ImageNet-1k zero-shot top-1):

| model | LAION-400M | LAION-2B | Δ points | error ratio |
|---|---:|---:|---:|---:|
| ViT-B/32 | 62.9 | 66.6 | +3.7 | **0.90027** |
| ViT-L/14 | 72.8 | 75.3 | +2.5 | **0.90809** |

**ANSWER: +5.29 accuracy points per decade of data at ViT-B/32**, +3.58 at
ViT-L/14 — a 5× data increase is 0.699 decades.

**ANSWER: those slopes disagree by 32.4%; the same step in error agrees to
0.87%.** As a power law, error ~ D^−0.0653 and D^−0.0599. That is why the
paper's conclusion is stated in error and not in accuracy.

**FINDING: the log-linear accuracy form names a finite crossing** — 100% top-1
at **4.1e15** images for B/32 and **1.6e16** for L/14. The power law never
reaches 100%. Two fits to the same two points disagree about what is possible.

**FINDING: and nothing reachable distinguishes them.** At 100× LAION-2B — 200
billion images — the line predicts 77.2% and the power law 75.3% (B/32), 82.5%
and 81.3% (L/14). Gaps of **1.9** and **1.2** points. The choice is settled by
the asymptote, not by a measurement.

**FINDING: the step is not a data-size step.** The checkpoints name their own
budgets: `laion400m_e32` is 32 epochs of 400M = 12.8B samples seen,
`laion2b_s34b_b79k` is 34B. Crediting the 3.7 points to 5× the data credits it
with a **2.66×** compute increase as well.
