<!-- generated:start -->
# 06-speech-and-audio / 03-audio-classification

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/03-audio-classification/) · upstream spec
`phases/06-speech-and-audio/03-audio-classification/docs/en.md`

```bash
uv run demo practice run 03-audio-classification --ex 1
uv run demo explain 03-audio-classification --ex 1
uv run pytest demos/phases/06-speech-and-audio/03-audio-classification
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It trains the k-NN MFCC baseline on a 4-class synthetic dataset (pu… | code | T1 | `ex01_the_matrix_is_the_identity_for_a_decade_of_noise.py` |
| 2 | Medium. Replace `summarize` with [mean, var, skew, kurtosis]. Does 4-moment pooling beat mean… | code | T1 | `ex02_fourteen_frames_do_not_carry_a_fourth_moment.py` |
| 3 | Hard. Using `torchaudio`, train a 2D CNN on ESC-50 fold 1. Report 5-fold cross-validation acc… | code | T1 | `ex03_the_time_mask_is_wider_than_the_clip.py` |
<!-- generated:end -->

## Answers

The lesson's baseline scores **20 of 20** on the dataset it ships with, and all
three exercises are about what that number can and cannot support. It survives a
ten-fold increase in noise unchanged; it is beaten by a one-line classifier; and
the two upgrades the exercises propose — a richer pooling and SpecAugment — are
both bounded by the same parameter, the **14 frames** a 0.25 s clip produces at
`frame_len=256, hop=128`.

All three run at **T1**: each rebuilds the corpus through the lesson's own
pure-Python DFT, which costs ~45 ms per clip.

### 1 — the matrix is the identity across a decade of noise

```text
                low    mid_low   mid_high       high
      low         5          0          0          0
  mid_low         0          5          0          0
 mid_high         0          0          5          0
     high         0          0          0          5
```

**FINDING: the same identity survives a ten-fold increase in noise.** The k-NN
still scores 1.00 at σ=0.5 — **−3.0 dB SNR**, against the lesson's +17.0 dB. A
confusion matrix that cannot move across a decade of its own nuisance parameter
is not reporting anything about the classifier.

**FINDING: one DFT argmax matches the pipeline and then beats it.**

| σ | SNR | k-NN on mean+var MFCC | argmax of one 1024-pt DFT |
|---:|---:|---:|---:|
| 0.05 (the lesson's) | +17.0 dB | 1.00 | 1.00 |
| 0.5 | −3.0 dB | 1.00 | 1.00 |
| 1.0 | −9.0 dB | 0.80 | **1.00** |
| 2.0 | −15.1 dB | 0.40 | **0.85** |

The four classes are 200/400/800/1600 Hz — octaves apart, single tones. The doc's
"surprisingly strong baseline" is true of MFCC k-NN in general and false of this
demonstration of it: `featurize → summarize → cosine → knn` buys a strictly worse
decision boundary here than an argmax.

**CONTROL: 5 test clips per class** means the matrix moves in steps of 1/20.
There is no resolution below 0.05.

### 2 — fourteen frames do not carry a fourth moment

| σ | mean+var (26-d) | 4-moment (52-d) |
|---:|---:|---:|
| 0.05 (the lesson's) | **1.00** | **1.00** |
| 1.0 | 0.80 | 0.90 |
| 4.0 | 0.30 | 0.15 |

**ANSWER: no** — and on the dataset as shipped the question has no answer at all,
since both poolings sit at the ceiling. Where the ceiling lifts the arms differ by
one to three clips of twenty, in either direction: one to three steps of the test
set's own resolution.

**MECHANISM: a clip is 14 frames.** At n=14 the standard error of skewness is
`√(6/n) = 0.655` and of excess kurtosis `√(24/n) = 1.309` — the size of the
quantities being estimated. Measured within-class relative spread agrees:

| block | mean | var | skew | kurtosis |
|---|---:|---:|---:|---:|
| within-class relative sd | **0.085** | 0.379 | **4.13** | 1.52 |

The two new blocks wander up to **49×** further inside a single class than the
mean block does.

**MECHANISM: `cosine` is norm-weighted and the new half holds 11% of the norm.**
MFCC means and variances run to tens; standardised third and fourth moments are
O(1). Twenty-six dimensions carrying `0.11` of the vector length can move a
cosine ranking by about that much, whatever they contain.

### 3 — the time mask is wider than the clip

`torch` and `torchaudio` are absent and ESC-50 is not here — the reference tree
holds no audio file of any kind. This is the `DESIGN D11` scaled-down run: 10
synthetic classes, 100 clips, the lesson's own STFT and mel filterbank, a
multinomial logistic head standing in for the CNN, real 5-fold cross-validation,
and SpecAugment implemented exactly as specified. The corpus and the head are
scaled down; the augmentation, the folds and the metric are the real thing.

| arm | 5-fold accuracy |
|---|---:|
| no augmentation | **0.810** |
| SpecAugment `T=20, F=10`, as specified | **0.790** |
| `T=3, F=10` — a mask that fits the clip | 0.780 |
| `T=3, F=10`, masking before `log_transform` | **0.090** |

**The delta is −0.020** — one clip in fifty, against a fold-to-fold spread of
0.35 (`[0.85, 0.80, 0.65, 0.75, 1.00]`). It is not a measurement, and two
structural facts say why before the number is read.

**FINDING: `T=20` is wider than the time axis.** A 0.25 s clip at
`frame_len=256, hop=128` is **14 frames**, so a 20-frame time mask covers **143%**
of every clip and erases all of it. The masked copies collapse to one constant
vector shared by all ten classes, which a softmax head can only absorb into its
intercepts — hence the near-zero delta. On ESC-50 (5 s at 16 kHz, hop 160) the
same `T=20` covers **4.0%** of 498 frames. The exercise carries the number across
a 36× change in frame count without adjusting it.

**FINDING: where the mask is applied decides the sign of the delta.** SpecAugment
masks the log-mel and fills with the mean. Masking the *linear* mel energy to zero
— the obvious way to write it against this pipeline — writes
`log(1e-10) = −23.026` into every masked cell, an outlier a mean+variance pool
cannot survive: same T, same F, **0.780 against 0.090**, the second below the
0.100 chance rate.

**CONTROL: for the classifier the lesson ships, SpecAugment is a no-op.** All
three k-NN arms return **0.540 fold for fold** (`[0.60, 0.60, 0.45, 0.40, 0.65]`).
`knn` has nothing to train, so augmented entries only join the bank, and they are
never among a query's three nearest. Augmentation is a claim about a model that
fits.

The full run the exercise asks for is
`torchaudio.datasets.ESC50` (2,000 clips, 600 MB) plus a ResNet-18 on log-mels for
5 folds — roughly 25 minutes on one consumer GPU, and the number to compare
against is the doc's own table: 82% for AST, 97.0% for BEATs-iter3.
