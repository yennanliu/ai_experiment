"""Exercise 2 — three of the four points plot at zero.

    **Medium.** Install `encodec` and compare 1, 4, 8, 32 codebooks on a held-out
    speech clip. Plot PESQ or MSE vs bitrate.

Reading of the exercise: `encodec`, `torch`, `torchaudio`, `pesq` and
`soundfile` are all absent, and the reference tree ships no `.wav`, `.flac`,
`.mp3` or `.ogg` anywhere, so there is no held-out speech clip either. Both axes
of the requested plot are still computable -- the bitrate axis from the doc's own
Step 1 numbers, and the MSE axis from the RVQ `code/main.py` already implements
(`DESIGN D11`) -- and computing them is what answers the exercise.

**The bitrate axis is exact and the doc supplies it.** Step 1 states 8 codebooks
at 6 kbps, 10-bit codes, and Step 3's table gives EnCodec-24k a 75 Hz frame rate:
`8 x 75 x 10 = 6000` bps, so one codebook is **0.75 kbps** and the four requested
points sit at **0.75, 3.00, 6.00 and 24.00 kbps**.

**The MSE axis has one usable point.** Run on the lesson's own quantizer:

| codebooks | kbps | measured MSE | as `code/main.py` prints it |
|---:|---:|---:|---:|
| 1 | 0.75 | 1.364684e-02 | 0.013647 |
| 4 | 3.00 | 1.073862e-06 | 0.000001 |
| 8 | 6.00 | 9.157033e-12 | **0.000000** |
| 32 | 24.00 | 4.119887e-32 | **0.000000** |

Three of the four points are `0.000001` or `0.000000` in the format the lesson
prints, and the two zeros stand **2.2e+20** apart. Plotted on a linear MSE axis
the whole curve is one visible point and three on the floor; it only becomes a
plot in dB, where the four points fall at **-18.6, -59.7, -110.4 and -313.9 dB**
and lie on a line.

**And the 32-codebook point is not measuring the codec.** The signal's RMS is
0.7759, so float64 resolves it to about **1.72e-16**. From **codebook 26** onward
every centroid is smaller than that: the quantizer is encoding its own rounding
error, and the 4.12e-32 at 32 codebooks is a property of double precision rather
than of the quantizer. The exercise's widest data point is the one that means
least.

Structure: `points` measures MSE at each requested codebook count; `kbps` is the
bitrate derived from the doc's own constants; `db` converts an MSE to decibels;
`precision_floor` finds the first codebook whose centroids sit below float64's
resolution of the signal.
"""

from __future__ import annotations

import importlib.util
import math

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "13-neural-audio-codecs"
LENGTH, SIZE, COUNTS = 1000, 8, (1, 4, 8, 32)
FRAME_HZ, CODE_BITS, DOC_COUNT, DOC_KBPS = 75, 10, 8, 6.0
EPS = 2.220446049250313e-16
ABSENT = ("encodec", "torch", "torchaudio", "pesq", "soundfile")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")


def kbps(count, frame_hz=FRAME_HZ, code_bits=CODE_BITS):
    """The doc's own arithmetic: 8 x 75 x 10 = 6000 bps, so 0.75 kbps a codebook."""
    return count * frame_hz * code_bits / 1000


def points(ref, signal, counts=COUNTS):
    """MSE at each codebook count the exercise names."""
    out = {}
    for count in counts:
        indices, books = ref.rvq_encode(signal, SIZE, count)
        out[count] = ref.mse(signal, ref.rvq_decode(indices, books, len(signal)))
    return out


def db(value):
    return 10 * math.log10(value)


def precision_floor(ref, signal, count=32):
    """(first codebook below float64's resolution of the signal, that resolution)."""
    rms = math.sqrt(sum(x * x for x in signal) / len(signal))
    floor = EPS * rms
    _, books = ref.rvq_encode(signal, SIZE, count)
    below = [k for k, book in enumerate(books) if max(abs(c) for c in book) < floor]
    return (below[0] if below else count), floor, rms


def clips_present():
    """Any audio file anywhere in the reference checkout -- the held-out clip."""
    root = parity.find_reference_root()
    return [p.name for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES][:3]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    signal = ref.generate_signal(n=LENGTH)
    errors = points(ref, signal)
    first, floor, rms = precision_floor(ref, signal)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "clips": clips_present(),
        "errors": errors, "rates": {n: kbps(n) for n in COUNTS},
        "formatted": {n: f"{errors[n]:.6f}" for n in COUNTS},
        "db": {n: db(errors[n]) for n in COUNTS},
        "doc_bps": DOC_COUNT * FRAME_HZ * CODE_BITS,
        "floor_at": first, "floor": floor, "rms": rms,
    }


def render(result):
    """Pre-formatted columns, so `verify` stays a list of claims rather than a loop."""
    errors, rates = result["errors"], result["rates"]
    return {
        "flat": [n for n, text in result["formatted"].items() if float(text) < 1e-5],
        "measured": [f"{errors[n]:.6e}" for n in COUNTS],
        "kbps": [rates[n] for n in COUNTS],
        "decibels": [f"{result['db'][n]:.1f}" for n in COUNTS],
        "monotone": all(result["db"][a] > result["db"][b] for a, b in zip(COUNTS, COUNTS[1:])),
    }


def verify(result):
    errors, formatted, rates = result["errors"], result["formatted"], result["rates"]
    shown = render(result)
    flat = shown["flat"]
    return [
        practice.Check(
            "CONTROL: no codec, no metric, and no held-out clip anywhere in the tree",
            len(result["absent"]) == len(ABSENT) and not result["clips"],
            f"find_spec is None for {result['absent']}, and a recursive scan of the reference "
            f"checkout finds {len(result['clips'])} files ending in {list(AUDIO_SUFFIXES)}. Both "
            "axes are still computable, which is what the rest of this measures",
        ),
        practice.Check(
            "MECHANISM: the bitrate axis comes out of the doc's own constants",
            result["doc_bps"] == DOC_KBPS * 1000 and rates[1] == 0.75,
            f"Step 1 says {DOC_COUNT} codebooks at {DOC_KBPS} kbps with {CODE_BITS}-bit codes "
            f"and Step 3 gives EnCodec-24k {FRAME_HZ} Hz: {DOC_COUNT} x {FRAME_HZ} x "
            f"{CODE_BITS} = {result['doc_bps']} bps exactly, so one codebook is {rates[1]} kbps "
            f"and the four requested points are {shown['kbps']} kbps",
        ),
        practice.Check(
            "ANSWER: three of the four points are at or below the printed floor",
            len(flat) == 3 and errors[8] / errors[32] > 1e19,
            f"on the lesson's own quantizer the four points measure {shown['measured']} and "
            f"print as {list(formatted.values())}. "
            f"Three land on the floor and the two zeros stand {errors[8] / errors[32]:.1e} "
            "apart -- a linear MSE axis shows one point and three at the bottom",
        ),
        practice.Check(
            "FINDING: it is only a plot in dB, where the four points lie on a line",
            shown["monotone"],
            f"the same four values are {shown['decibels']} dB, falling "
            f"{(result['db'][1] - result['db'][32]) / 31:.1f} dB per codebook across the range. "
            "The axis the exercise asks to plot is the one that hides the result",
        ),
        practice.Check(
            "FINDING: past codebook 26 the quantizer is encoding float64 rounding error",
            result["floor_at"] < 32,
            f"the signal's RMS is {result['rms']:.4f}, so float64 resolves it to "
            f"{result['floor']:.2e}; from codebook {result['floor_at']} onward every centroid is "
            f"smaller than that. The {errors[32]:.2e} at 32 codebooks is a property of double "
            "precision, so the exercise's widest data point is the one that means least",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
