"""Exercise 1 — the table goes blind at the fifth codebook.

    **Easy.** Run `code/main.py`. It implements a toy scalar + residual
    quantizer and measures reconstruction error as you add codebooks.

Reading of the exercise: the deliverable is the table Step 2 prints, so the
exercise is read as "run it and check that the table says what it appears to
say". It does not. Two of its five rows are printed as `0.000000`, and they are
not the same number:

| # codebooks | printed MSE | measured MSE |
|---:|---:|---:|
| 1 | 0.013647 | 1.364684e-02 |
| 2 | 0.000521 | 5.205277e-04 |
| 4 | 0.000001 | 1.073862e-06 |
| **8** | **0.000000** | **9.157033e-12** |
| **12** | **0.000000** | **3.886251e-17** |

The last two rows differ by a factor of **235,626** and print identically.
`%.6f` runs out at the fifth codebook, and the rows it erases are exactly the
ones the doc's Pitfalls section is about ("stop at 8-12").

**The error is geometric, and the doc calls it linear.** Each codebook divides
the MSE by **21.0** on average -- **13.22 dB** per codebook, never below 10.03 dB
across the twelve. The Pitfall "adding codebooks increases fidelity linearly but
LM sequence length linearly too" is only true of fidelity read in dB; the column
the lesson prints is linear MSE, where the same relationship is a cliff.

**`bits_per_frame = n_cb * 3` charges for a uniform index distribution that does
not occur.** Measured index entropy over the twelve codebooks is **29.136 bits**
per sample against the **36** the table bills -- 2.43 bits per codebook, not 3 --
so the bitrate column is **23.6%** high. One codebook even runs a dead centroid:
cb8 uses 7 of its 8 codewords.

**And the codebooks are never counted.** Eight `float64` centroids per codebook
is 512 bits of side information, **17.1%** of the index payload at every row of
the table, and it appears in no column.

Structure: `curve` measures MSE at each codebook count; `decibels` is the
per-codebook gain in dB; `entropy` is the empirical bits of one index stream;
`side_information` compares stored centroids against stored indices.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "13-neural-audio-codecs"
LENGTH, SIZE, PRINTED = 1000, 8, (1, 2, 4, 8, 12)
BITS_EACH, CENTROID_BITS = 3, 64


def curve(ref, signal, counts):
    """Measured MSE at each codebook count, the column the table rounds away."""
    out = {}
    for count in counts:
        indices, books = ref.rvq_encode(signal, SIZE, count)
        out[count] = ref.mse(signal, ref.rvq_decode(indices, books, len(signal)))
    return out


def decibels(errors):
    """Gain in dB from each added codebook, in order."""
    ordered = [errors[n] for n in sorted(errors)]
    return [10 * math.log10(before / after) for before, after in zip(ordered, ordered[1:])]


def entropy(indices):
    """Empirical bits per symbol of one index stream."""
    counts = Counter(indices)
    return -sum((n / len(indices)) * math.log2(n / len(indices)) for n in counts.values())


def side_information(count, length=LENGTH):
    """(index bits, codebook bits) -- the second is what the bitrate column omits."""
    return count * BITS_EACH * length, count * SIZE * CENTROID_BITS


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    signal = ref.generate_signal(n=LENGTH)
    errors = curve(ref, signal, range(1, 13))
    indices, _ = ref.rvq_encode(signal, SIZE, 12)
    bits = [entropy(stream) for stream in indices]
    payload, stored = side_information(12)
    return {
        "printed": {n: errors[n] for n in PRINTED},
        "formatted": {n: f"{errors[n]:.6f}" for n in PRINTED},
        "gains": decibels(errors),
        "bits": bits, "charged": 12 * BITS_EACH,
        "used": [len(set(stream)) for stream in indices],
        "payload": payload, "stored": stored,
    }


def verify(result):
    printed, formatted = result["printed"], result["formatted"]
    blind = [n for n, text in formatted.items() if float(text) == 0.0]
    gains, measured = result["gains"], sum(result["bits"])
    return [
        practice.Check(
            "ANSWER: two of the five printed rows are 0.000000 and differ by 235,662x",
            blind == [8, 12] and printed[8] / printed[12] > 1e5,
            f"the table prints {list(formatted.values())} where the measured values are "
            f"{[f'{printed[n]:.6e}' for n in PRINTED]}; rows {blind} collapse to zero and stand "
            f"a factor of {printed[8] / printed[12]:,.0f} apart. `%.6f` runs out at the fifth "
            "codebook, and those rows are what the doc's 'stop at 8-12' pitfall is about",
        ),
        practice.Check(
            "MECHANISM: each codebook divides the error by 21, so the column is a cliff",
            13.0 < statistics.mean(gains) < 13.5 and min(gains) > 10,
            f"measured gain per added codebook is {statistics.mean(gains):.2f} dB on average, "
            f"between {min(gains):.2f} and {max(gains):.2f} dB across twelve -- a mean ratio of "
            f"{10 ** (statistics.mean(gains) / 10):.1f}x. Linear MSE cannot hold that range in "
            "six decimal places",
        ),
        practice.Check(
            "FINDING: the doc calls this linear, and only the dB reading is",
            statistics.pstdev(gains) < 2.0,
            f"'adding codebooks increases fidelity linearly' is true of dB, where the gain is "
            f"{statistics.mean(gains):.2f} +/- {statistics.pstdev(gains):.2f} per codebook and "
            "so genuinely additive. The table prints linear MSE, where the same relationship "
            "falls off the bottom of the format after four rows",
        ),
        practice.Check(
            "FINDING: bits/frame charges 3 bits a codebook for 2.43 bits of entropy",
            measured < result["charged"] and 7 in result["used"],
            f"measured index entropy is {measured:.3f} bits per sample over twelve codebooks "
            f"against the {result['charged']} the table bills -- "
            f"{(result['charged'] / measured - 1) * 100:.1f}% high, "
            f"{measured / 12:.2f} bits each. One codebook runs a dead centroid: the usage counts "
            f"are {result['used']}",
        ),
        practice.Check(
            "CONTROL: the codebooks themselves appear in no column",
            abs(result["stored"] / result["payload"] - 0.171) < 0.01,
            f"{SIZE} float64 centroids per codebook is {result['stored']} bits of side "
            f"information against {result['payload']} bits of indices at twelve codebooks -- "
            f"{result['stored'] / result['payload'] * 100:.1f}%, the same share at every row, "
            "and the bitrate column counts only the indices",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
