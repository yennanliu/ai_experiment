"""Exercise 3 — the three feature sets differ by one number.

    **Hard.** Implement `power_to_db` and compare ASR accuracy of a tiny CNN
    classifier on AudioMNIST using (a) raw log-mel, (b) dB-mel with `ref=max`,
    (c) MFCC-13 + delta + delta-delta. Report top-1 accuracy.

Reading of the exercise: AudioMNIST is not here and neither is torch, so the
substitutions are a 10-class synthetic spoken-digit stand-in (two formants per
class, jittered F0, seeded) and a multinomial logistic classifier over
mean+std-pooled frames. Both substitutions are shared by all three arms, so the
comparison the exercise asks for survives them; what would not survive is the
claim that a CNN is what makes any of these numbers what they are.

`power_to_db` is implemented as librosa defines it -- `10*log10(S/ref)` with
`ref=max`, `amin=1e-10` and an 80 dB floor -- and the first result is that it
does not produce a new representation:

    power_to_db(S) == (10/ln 10) * log_mel(S) - max(...)      to 2.1e-14

(b) is (a) rescaled by 4.3429 and shifted by **one scalar per utterance**. On
matched material the three arms are therefore indistinguishable -- **1.000,
1.000, 1.000** -- and the comparison measures nothing at all.

What separates them is a test set the training set did not describe. Train on
full-level utterances, test on the same utterances 20-40 dB quieter:

| features | matched | 20-40 dB quieter |
|---|---:|---:|
| (a) log-mel | 1.000 | **0.383** |
| (b) dB-mel, `ref=max` | 1.000 | **1.000** |
| (c) MFCC-13 + delta + delta-delta | 1.000 | **0.150** |
| (c') the same, with c0 dropped | 1.000 | **1.000** |

(c) is the interesting row. Deltas difference out any constant, so 26 of its 39
dimensions are level-invariant by construction, and 12 of the remaining 13 are
too. **One dimension of 39 -- c0 -- carries the level**, and because `c0` is the
sum of all 40 log-mels it moves 40x as far as any single log-mel bin does, which
is why (c) ends up *below* (a). `main()`'s own Step 6 prints "coef 0 encodes
overall energy; typically dropped downstream"; the exercise's feature spec keeps
it, and dropping it recovers 1.000.

Structure: `power_to_db` is the implementation the exercise asks for; `digit`
synthesises one utterance; `mel_of` runs the lesson's STFT and filterbank;
`arms` holds the four feature maps; `pooled` is the mean+std frame pooling;
`score` fits on the loud set and reports accuracy on another.
"""

from __future__ import annotations

import math
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "02-spectrograms-mel-features"
SR, FRAME, HOP, N_MELS, SECONDS = 8000, 256, 128, 40, 0.2
FORMANTS = [(700, 1100), (400, 1900), (500, 2400), (600, 900), (350, 2200),
            (450, 1600), (750, 1300), (300, 2600), (550, 1800), (650, 2000)]
PER_CLASS, QUIET_DB = 6, (-40, -20)


def power_to_db(mel, amin=1e-10, top_db=80.0):
    """librosa's contract: 10*log10(S/max(S)), floored at amin and at peak - top_db."""
    peak = max(max(v for v in frame) for frame in mel)
    db = [[10 * math.log10(max(v, amin) / max(peak, amin)) for v in frame] for frame in mel]
    floor = max(max(frame) for frame in db) - top_db
    return [[max(v, floor) for v in frame] for frame in db]


def digit(rng, label, gain):
    """One utterance: a jittered glottal source through the class's two formants."""
    f1, f2 = FORMANTS[label]
    f0 = 110 * (1 + 0.12 * (rng.random() - 0.5))
    harmonics = [(k * f0, 1.0 / (1 + ((k * f0 - f1) / 120) ** 2)
                  + 0.7 / (1 + ((k * f0 - f2) / 180) ** 2) + 0.05 / k)
                 for k in range(1, int(3000 // f0) + 1)]
    wave = [sum(a * math.sin(2 * math.pi * f * i / SR) for f, a in harmonics)
            for i in range(int(SR * SECONDS))]
    scale = gain / max(abs(x) for x in wave)
    return [x * scale for x in wave]


def mel_of(ref, signal, bank):
    return ref.apply_filterbank(ref.stft_magnitude(signal, FRAME, HOP), bank)


def mfcc_deltas(ref, mel, keep_c0=True):
    """MFCC-13 plus first and second temporal differences, 39 dimensions."""
    base = np.array([ref.dct_ii(frame, 13) for frame in ref.log_transform(mel)])
    first = np.diff(base, axis=0, prepend=base[:1])
    second = np.diff(first, axis=0, prepend=first[:1])
    stack = np.concatenate([base, first, second], axis=1)
    return stack if keep_c0 else stack[:, [i for i in range(39) if i % 13]]


def pooled(frames):
    array = np.asarray(frames, dtype=float)
    return np.concatenate([array.mean(0), array.std(0)])


def dataset(ref, rng, bank, loud):
    """(mel spectrograms, labels) -- `loud` picks full level or 20-40 dB down."""
    pairs = [(digit(rng, label, 1.0 if loud else 10 ** (rng.uniform(*QUIET_DB) / 20)), label)
             for label in range(10) for _ in range(PER_CLASS)]
    return [mel_of(ref, signal, bank) for signal, _ in pairs], [label for _, label in pairs]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(0)
    bank = ref.mel_filterbank(N_MELS, FRAME, SR)
    train, labels = dataset(ref, rng, bank, loud=True)
    matched, matched_y = dataset(ref, rng, bank, loud=True)
    quiet, quiet_y = dataset(ref, rng, bank, loud=False)
    arms = {
        "(a) log-mel": ref.log_transform,
        "(b) dB-mel ref=max": power_to_db,
        "(c) MFCC-13+d+dd": lambda m: mfcc_deltas(ref, m),
        "(c') the same, no c0": lambda m: mfcc_deltas(ref, m, keep_c0=False),
    }
    scores = {}
    for name, feature in arms.items():
        model = LogisticRegression(max_iter=3000).fit(
            np.array([pooled(feature(m)) for m in train]), labels)
        scores[name] = (model.score(np.array([pooled(feature(m)) for m in matched]), matched_y),
                        model.score(np.array([pooled(feature(m)) for m in quiet]), quiet_y))
    natural = np.array(ref.log_transform(train[0])) * 10 / math.log(10)
    return {"scores": scores, "n": len(train),
            "identity": float(np.max(np.abs(np.array(power_to_db(train[0]))
                                            - (natural - natural.max())))),
            "invariant": 39 - 1}


def verify(result):
    scores = result["scores"]
    (log_m, log_q), (db_m, db_q) = scores["(a) log-mel"], scores["(b) dB-mel ref=max"]
    (mf_m, mf_q), (no_m, no_q) = scores["(c) MFCC-13+d+dd"], scores["(c') the same, no c0"]
    return [
        practice.Check(
            "ANSWER: on matched material all three arms score 1.000",
            min(log_m, db_m, mf_m) == 1.0,
            f"(a) {log_m:.3f}, (b) {db_m:.3f}, (c) {mf_m:.3f} top-1 over {result['n']} held-out "
            "utterances trained on as many. The comparison the exercise asks for separates "
            "nothing until the test set stops resembling the training set",
        ),
        practice.Check(
            "MECHANISM: power_to_db is log-mel rescaled and shifted by one scalar",
            result["identity"] < 1e-10,
            f"`10*log10(S/max S)` equals `(10/ln 10) * log_mel(S) - max(...)` to "
            f"{result['identity']:.1e} on a real utterance, so (b) is (a) times 4.3429 minus one "
            "number per utterance. Any gap between them is a gap about overall level",
        ),
        practice.Check(
            "ANSWER: 20-40 dB quieter, (b) holds at 1.000 and (a) falls to 0.383",
            db_q == 1.0 and log_q < 0.5,
            f"trained at full level and tested {-QUIET_DB[1]}-{-QUIET_DB[0]} dB down: "
            f"(a) {log_m:.3f} -> {log_q:.3f}, (b) {db_m:.3f} -> {db_q:.3f}. `ref=max` removes the "
            "level the log-mel carries as an additive offset, and that is all it does",
        ),
        practice.Check(
            "FINDING: one dimension of 39 sinks the MFCC arm, and the lesson names it",
            mf_q < log_q and no_q == 1.0,
            f"(c) falls to {mf_q:.3f}, below (a)'s {log_q:.3f}, while dropping c0 gives "
            f"{no_q:.3f}. Deltas difference out any constant, so {result['invariant']} of 39 "
            "dims are level-invariant already; c0 is the sum of 40 log-mels and moves 40x as far "
            "as one bin. Step 6 prints 'typically dropped downstream'; the spec keeps it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
