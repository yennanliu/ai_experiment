"""Exercise 3 — the time mask is wider than the clip.

    **Hard.** Using `torchaudio`, train a 2D CNN on ESC-50 fold 1. Report 5-fold
    cross-validation accuracy. Add SpecAugment (time mask = 20, freq mask = 10)
    and report the delta.

Reading of the exercise: `torch` and `torchaudio` are absent and ESC-50 is not
here -- the reference tree holds no audio file of any kind -- so this is the D11
scaled-down run: 10 synthetic classes, 100 clips, the lesson's own STFT and mel
filterbank, a multinomial logistic head standing in for the CNN, real 5-fold
cross-validation, and SpecAugment implemented exactly as the exercise specifies
it. What is scaled down is the corpus and the head; the augmentation, the folds
and the metric are the real thing. The command and cost for the full run are in
the lesson README.

| arm | 5-fold accuracy |
|---|---:|
| no augmentation | **0.810** |
| SpecAugment T=20, F=10, as specified | **0.790** |
| T=3, F=10 (a mask that fits the clip) | 0.780 |
| T=3, F=10, masking before `log_transform` | **0.090** |

The delta the exercise asks for is **-0.020**, one clip in fifty, against a
fold-to-fold spread of 0.35. It is not a measurement, and two structural facts
say why before any number is read.

**T=20 is wider than the time axis.** At the lesson's `frame_len=256, hop=128`
over a 0.25 s clip there are **14 frames**, so a 20-frame time mask covers
**143%** of every clip and erases all of it, every time. The masked copies then
collapse to one constant vector shared by all ten classes, which a softmax head
can only absorb into its intercepts -- hence the near-zero delta. On ESC-50 the
same `T=20` covers **4.0%** of 498 frames. The hyperparameter is bound to the
corpus it was published for, and the exercise carries it across a 36x change in
frame count unchanged.

**Where the mask is applied decides the sign of the delta.** SpecAugment masks
the log-mel and fills with the mean. Masking the *linear* mel energy to zero --
the obvious way to write it against this pipeline -- writes `log(1e-10) =
-23.026` into every masked cell, an outlier a mean+variance pool cannot survive:
same T, same F, **0.780 against 0.090**, the second below the 0.100 chance rate.

And for the classifier the lesson actually ships, SpecAugment is a structural
no-op. k-NN has nothing to train, so augmented entries only join the bank, and
they are never among any query's three nearest: **0.540 in every arm, fold for
fold.** Augmentation is a statement about a model that fits, and `knn` does not.

Structure: `clip` synthesises one utterance; `log_mel` runs the lesson's STFT and
filterbank once per clip so masking is cheap; `spec_augment` is the masking, with
the fill value as a parameter; `cross_validate` is the 5-fold loop over either
head.
"""

from __future__ import annotations

import math
import random
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "03-audio-classification"
SR, SECONDS, N_MELS, FRAME, HOP = 8000, 0.25, 40, 256, 128
CLASSES, PER_CLASS, FOLDS, COPIES = 10, 10, 5, 2
TONES = [(220, 1.6), (262, 1.9), (294, 1.35), (330, 2.2), (349, 1.5),
         (392, 1.75), (440, 1.25), (494, 2.0), (523, 1.45), (587, 1.85)]
FLOOR, ESC50_FRAMES = math.log(1e-10), 1 + (5 * 16000 - 400) // 160


def clip(label, rnd):
    """Three partials at a class-specific ratio, jittered F0, heavy additive noise."""
    f0, ratio = TONES[label]
    f0 *= 1 + 0.06 * (rnd.random() - 0.5)
    partials = [(f0, 0.3), (f0 * ratio, 0.18), (f0 * ratio * ratio, 0.09)]
    return [sum(a * math.sin(2 * math.pi * f * i / SR) for f, a in partials)
            + rnd.gauss(0, 0.45) for i in range(int(SR * SECONDS))]


def pooled(ref, frames):
    """The lesson's own fixed-length embedding: mean+var of 13 MFCC."""
    return ref.summarize([ref.dct_ii(frame, 13) for frame in frames])


def spec_augment(frames, time_mask, freq_mask, rnd, fill=None):
    """One time band and one frequency band replaced by `fill` (default: the mean)."""
    out = [row[:] for row in frames]
    n_frames, n_bins = len(out), len(out[0])
    value = (sum(sum(row) for row in frames) / (n_frames * n_bins)) if fill is None else fill
    width = min(time_mask, n_frames)
    start = rnd.randint(0, n_frames - width)
    for i in range(start, start + width):
        out[i] = [value] * n_bins
    band = min(freq_mask, n_bins)
    edge = rnd.randint(0, n_bins - band)
    for row in out:
        row[edge:edge + band] = [value] * band
    return out


def training_bank(ref, corpus, labels, train, augment, rnd):
    """The training side only -- augmenting the held-out fold would leak."""
    bank, gold = [pooled(ref, corpus[i]) for i in train], [labels[i] for i in train]
    for index in train if augment else []:
        for _ in range(COPIES):
            bank.append(pooled(ref, augment(corpus[index], rnd)))
            gold.append(labels[index])
    return bank, gold


def cross_validate(ref, corpus, labels, augment, head):
    """Five folds over a shuffled corpus, scored by whichever head is passed."""
    order = list(range(len(corpus)))
    random.Random(3).shuffle(order)
    folds = [order[i::FOLDS] for i in range(FOLDS)]
    rnd, scores = random.Random(11), []
    for held in range(FOLDS):
        train = [i for f, fold in enumerate(folds) if f != held for i in fold]
        bank, gold = training_bank(ref, corpus, labels, train, augment, rnd)
        scores.append(head(bank, gold, [pooled(ref, corpus[i]) for i in folds[held]],
                           [labels[i] for i in folds[held]]))
    return sum(scores) / FOLDS, scores


def linear_head(bank, gold, queries, truth):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(max_iter=6000).fit(np.array(bank), gold)
    return model.score(np.array(queries), truth)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rnd = random.Random(7)
    bank = ref.mel_filterbank(N_MELS, FRAME, SR)
    corpus = [ref.log_transform(ref.apply_filterbank(
        ref.stft_mag(clip(label, rnd), FRAME, HOP), bank))
        for label in range(CLASSES) for _ in range(PER_CLASS)]
    labels = [label for label in range(CLASSES) for _ in range(PER_CLASS)]
    arms = {"none": None,
            "T=20 F=10": lambda f, r: spec_augment(f, 20, 10, r),
            "T=3 F=10": lambda f, r: spec_augment(f, 3, 10, r),
            "T=3 F=10 pre-log": lambda f, r: spec_augment(f, 3, 10, r, FLOOR)}
    vote = lambda b, g, q, t: sum(ref.knn(x, b, g, 3) == y for x, y in zip(q, t)) / len(t)  # noqa: E731
    return {"frames": len(corpus[0]), "floor": FLOOR,
            "linear": {n: cross_validate(ref, corpus, labels, a, linear_head)
                       for n, a in arms.items()},
            "knn": {n: cross_validate(ref, corpus, labels, a, vote)
                    for n, a in list(arms.items())[:3]}}


def verify(result):
    linear, knn, frames = result["linear"], result["knn"], result["frames"]
    base, spec = linear["none"][0], linear["T=20 F=10"][0]
    fitted, prelog = linear["T=3 F=10"][0], linear["T=3 F=10 pre-log"][0]
    return [
        practice.Check(
            "ANSWER: 5-fold 0.810 without, 0.790 with SpecAugment -- a delta of -0.020",
            abs(spec - base) <= 0.05,
            f"{base:.3f} -> {spec:.3f} over {CLASSES * PER_CLASS} clips, delta {spec - base:+.3f}, "
            f"which is one clip in fifty against a fold-to-fold spread of "
            f"{max(linear['none'][1]) - min(linear['none'][1]):.2f}: {linear['none'][1]}",
        ),
        practice.Check(
            "FINDING: a 20-frame time mask is wider than the 14-frame clip",
            frames == 14 and 20 > frames,
            f"at frame_len={FRAME}, hop={HOP} over {SECONDS} s a clip is {frames} frames, so "
            f"T=20 covers {20 / frames * 100:.0f}% of every one and erases all of it. On ESC-50 "
            f"(5 s at 16 kHz, hop 160) the same T=20 covers {20 / ESC50_FRAMES * 100:.1f}% of "
            f"{ESC50_FRAMES} frames -- the exercise carries the number across a "
            f"{ESC50_FRAMES / frames:.0f}x change in frame count",
        ),
        practice.Check(
            "FINDING: where the mask is applied decides the sign of the delta",
            prelog < 0.15 < fitted,
            f"the same T=3, F=10 scores {fitted:.3f} filling with the mean of the log-mel and "
            f"{prelog:.3f} zeroing the linear mel before `log_transform`, which writes "
            f"{result['floor']:.3f} into each masked cell -- an outlier a mean+variance pool "
            f"cannot survive, below the {1 / CLASSES:.3f} chance rate",
        ),
        practice.Check(
            "CONTROL: for the classifier the lesson ships, SpecAugment is a no-op",
            len({tuple(row[1]) for row in knn.values()}) == 1,
            f"all three k-NN arms return {knn['none'][0]:.3f} fold for fold ({knn['none'][1]}): "
            "augmented entries only join the bank and are never among a query's three nearest. "
            "Augmentation is a claim about a model that fits, and `knn` does not fit",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
