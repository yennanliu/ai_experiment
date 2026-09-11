"""Exercise 1 — the matrix is the identity across a decade of noise.

    **Easy.** Run `code/main.py`. It trains the k-NN MFCC baseline on a 4-class
    synthetic dataset (pure tones at different pitches). Report confusion matrix.

Reading of the exercise: reporting the matrix is one line -- it is the identity,
20 of 20, every off-diagonal zero. So the exercise is read as "report it *and*
say what it reports", and the answer is that it reports almost nothing: the same
identity comes back over a **ten-fold range of noise**, from the lesson's
sigma=0.05 down to sigma=0.5, which is +17 dB SNR down to **-3 dB**. A measurement
that cannot move across a decade of its own nuisance parameter is not measuring
the classifier.

What it hides is that the pipeline is **strictly dominated by one line of code**.
The four classes are 200, 400, 800 and 1600 Hz -- octaves apart, single tones --
so taking one 1024-point DFT and snapping the argmax bin to the nearest of the
four scores 20 of 20 as well, and keeps scoring when the k-NN stops:

| sigma | SNR | k-NN on mean+var MFCC | argmax DFT bin |
|---:|---:|---:|---:|
| 0.05 (the lesson's) | +17.0 dB | 1.00 | 1.00 |
| 0.5 | -3.0 dB | 1.00 | 1.00 |
| 1.0 | -9.0 dB | 0.80 | **1.00** |
| 2.0 | -15.1 dB | 0.40 | **0.85** |

`featurize` -> `summarize` -> `cosine` -> `knn` is the lesson's whole machinery,
and on the lesson's own data it buys a strictly worse decision boundary than
`max(range(len(mag)), key=mag.__getitem__)`. The doc's "surprisingly strong
baseline" is true of MFCC k-NN in general and false of this demonstration of it.

The test set also cannot express much: 5 clips per class means the matrix moves
in steps of 1/20, so "0.95" is the nearest expressible number to a single error.

Structure: `dataset` rebuilds `main()`'s corpus at a chosen noise level with the
lesson's own `sine`, `add_noise`, `featurize` and `summarize`; `confusion` and
`accuracy` score the lesson's `knn`; `peak_rule` is the one-line classifier.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "03-audio-classification"
SR, SECONDS, K = 8000, 0.25, 3
CLASSES = {"low": 200, "mid_low": 400, "mid_high": 800, "high": 1600}
SIGMAS = (0.05, 0.5, 1.0, 2.0)
PROBE = 1024


def dataset(ref, sigma, n_train, n_test):
    """`main()`'s corpus at one noise level: pooled features plus the raw test audio."""
    random.seed(42)
    train, labels, audio, gold = [], [], [], []
    for name, freq in CLASSES.items():
        for index in range(n_train + n_test):
            signal = ref.add_noise(ref.sine(freq, SR, SECONDS), sigma)
            if index < n_train:
                train.append(ref.summarize(ref.featurize(signal, SR)))
                labels.append(name)
            else:
                audio.append(signal)
                gold.append(name)
    return train, labels, audio, gold


def peak_rule(ref, signal):
    """One DFT, one argmax, snapped to the nearest class frequency in log-Hz."""
    magnitude = ref.dft_mag(signal[:PROBE])
    bin_index = max(range(1, len(magnitude)), key=lambda i: magnitude[i])
    hertz = max(bin_index * SR / PROBE, 1e-9)
    return min(CLASSES, key=lambda name: abs(math.log(CLASSES[name] / hertz)))


def confusion(ref, train, labels, audio, gold):
    """Rows gold, columns predicted, as `main()` prints it."""
    table = {row: dict.fromkeys(CLASSES, 0) for row in CLASSES}
    for signal, truth in zip(audio, gold):
        table[truth][ref.knn(ref.summarize(ref.featurize(signal, SR)), train, labels, K)] += 1
    return table


def accuracy(table):
    hits = sum(table[name][name] for name in CLASSES)
    return hits / sum(sum(row.values()) for row in table.values()), hits


def snr_db(sigma, amp=0.5):
    return 20 * math.log10(amp / math.sqrt(2) / sigma)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep, table = {}, None
    for sigma in SIGMAS:
        train, labels, audio, gold = dataset(ref, sigma, 12, 5)
        scored = confusion(ref, train, labels, audio, gold)
        table = scored if table is None else table
        sweep[sigma] = {
            "knn": accuracy(scored)[0],
            "peak": sum(peak_rule(ref, s) == g for s, g in zip(audio, gold)) / len(gold),
            "snr": snr_db(sigma),
        }
    rate, hits = accuracy(table)
    return {"table": table, "rate": rate, "hits": hits, "sweep": sweep,
            "off_diagonal": sum(v for row, cells in table.items()
                                for col, v in cells.items() if row != col),
            "per_class": len(table["low"]) and sum(table["low"].values())}


def verify(result):
    sweep, table = result["sweep"], result["table"]
    quiet, loud = sweep[SIGMAS[1]], sweep[SIGMAS[2]]
    dominated = [s for s, row in sweep.items() if row["peak"] > row["knn"]]
    return [
        practice.Check(
            "ANSWER: the confusion matrix is the identity, 20 of 20",
            result["off_diagonal"] == 0 and result["hits"] == 20,
            f"every off-diagonal cell is zero and each of the four classes scores "
            f"{table['low']['low']}/{result['per_class']}: {[table[c][c] for c in CLASSES]} on the "
            f"diagonal, accuracy {result['rate']:.3f}, exactly as `main()` prints it",
        ),
        practice.Check(
            "FINDING: the same identity survives a ten-fold increase in noise",
            quiet["knn"] == 1.0,
            f"at sigma={SIGMAS[1]} -- {SIGMAS[1] / SIGMAS[0]:.0f}x the lesson's noise, "
            f"{quiet['snr']:+.1f} dB SNR against {sweep[SIGMAS[0]]['snr']:+.1f} -- the k-NN still "
            f"scores {quiet['knn']:.2f}. A matrix that cannot move across a decade of its own "
            "nuisance parameter is not reporting anything about the classifier",
        ),
        practice.Check(
            "FINDING: one DFT argmax matches the pipeline and then beats it",
            len(dominated) >= 2 and loud["peak"] > loud["knn"],
            f"snapping the argmax bin of a single {PROBE}-point DFT to the nearest of "
            f"{list(CLASSES.values())} Hz ties the k-NN at every noise level it survives and wins "
            f"at {dominated}: {loud['peak']:.2f} against {loud['knn']:.2f} at sigma={SIGMAS[2]}, "
            f"{sweep[SIGMAS[3]]['peak']:.2f} against {sweep[SIGMAS[3]]['knn']:.2f} at "
            f"sigma={SIGMAS[3]}",
        ),
        practice.Check(
            "CONTROL: five test clips per class means the matrix moves in steps of 0.05",
            result["per_class"] == 5,
            f"{result['per_class']} clips in each of {len(CLASSES)} classes is {result['hits'] and 20} "
            "decisions, so the only reachable accuracies are multiples of 1/20 and the nearest "
            "expressible number to one error is 0.95. There is no resolution below that",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
