"""Exercise 1 — the Easy exercise is the expensive one.

    **Easy.** Synthesize a 1-second mix of 220 Hz + 440 Hz + 880 Hz at 16 kHz.
    Run DFT. Confirm three peaks at the expected bins.

Reading of the exercise: "Run DFT" means the lesson's own `dft` from Step 3, not
`numpy.fft` -- the exercise is the sequel to that step, and substituting a library
FFT would confirm nothing about the code the lesson wrote. So this runs the real
thing at the size the exercise names, and the size is the finding.

The answer is clean: **three peaks at bins 220, 440 and 880**, each of magnitude
**1333.333**, which is `(0.5/3) * 16000/2` to 13 digits. The bins are the
frequencies in Hz because one second at 16 kHz makes the bin width exactly 1 Hz --
`k * sr/N = k`. That is a property of the exercise's own parameters and of nothing
else, and the lesson's own demo does not have it: `main()` runs 512 samples at
8 kHz, where 220 Hz sits at bin 14.08, and it prints **218.8, 437.5, 875.0** Hz.

The height is `(0.5/3)`, not `0.5`, because `mix` averages -- `sum(s[i] for s in
signals) / len(signals)` -- so mixing three tones at `amp=0.5` attenuates each by
three rather than summing to 1.5. Nothing in the lesson says so.

The cost is the real finding. The doc's own Step 3 says the O(N^2) DFT is "fine
for N=256 to confirm correctness, useless for real audio", and the exercise two
sections later asks for N=16000 -- **977x the work of `main()`'s largest call**.
Measured: **~24 s** against `numpy.fft.rfft`'s **~0.1 ms** on the same array, a
factor of order 1e5, for a spectrum that agrees to **3.7e-13** relative. The Easy
exercise is the most expensive one in the lesson; Exercise 3, labelled Hard, is
16x cheaper (98 frames of 400 against one transform of 16000).

Structure: `mix_tones` builds the signal with the lesson's own `sine` and `mix`;
`timed` runs a callable and returns (result, seconds); `top_bins` reads peaks off
a magnitude list; `lesson_grid` repeats the exercise at `main()`'s own sr and
duration to show where the bins stop being integers.
"""

from __future__ import annotations

import time

import numpy as np

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "01-audio-fundamentals"
SR, SECONDS, TONES = 16000, 1.0, (220.0, 440.0, 880.0)
AMP = 0.5


def mix_tones(ref, sr, seconds):
    """The exercise's signal, built with the lesson's own `sine` and `mix`."""
    return ref.mix(*(ref.sine(f, sr, seconds) for f in TONES))


def timed(fn, *args):
    """(result, wall-clock seconds) -- the exercise's cost is half its answer."""
    start = time.perf_counter()
    return fn(*args), time.perf_counter() - start


def top_bins(mags, count):
    """The `count` largest bins, in ascending bin order."""
    return sorted(sorted(range(len(mags)), key=lambda i: -mags[i])[:count])


def lesson_grid(ref):
    """The same three tones on `main()`'s own grid: 512 samples at 8 kHz."""
    sr, seconds = 8000, 0.064
    signal = mix_tones(ref, sr, seconds)
    half = ref.magnitudes(ref.dft(signal))[: len(signal) // 2]
    bins = top_bins(half, 3)
    return {
        "n": len(signal), "sr": sr,
        "hz": [k * sr / len(signal) for k in bins],
        "exact": [f * len(signal) / sr for f in TONES],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    signal = mix_tones(ref, SR, SECONDS)
    spectrum, lesson_seconds = timed(ref.dft, signal)
    mags = ref.magnitudes(spectrum)
    half = mags[: len(mags) // 2]
    array = np.asarray(signal)
    rfft = lambda x: np.abs(np.fft.rfft(x))  # noqa: E731
    rfft(array)                                   # warm the plan cache before timing it
    fast, numpy_seconds = timed(rfft, array)
    return {
        "n": len(signal), "bins": top_bins(half, 3),
        "heights": [mags[int(f)] for f in TONES],
        "predicted": AMP / len(TONES) * len(signal) / 2,
        "peak_amp": max(abs(s) for s in signal),
        "lesson_seconds": lesson_seconds, "numpy_seconds": numpy_seconds,
        "rel": float(np.max(np.abs(fast - np.asarray(half + [mags[len(mags) // 2]])[: len(fast)]))
                     / fast.max()),
        "mirror": [abs(mags[int(f)] - mags[len(mags) - int(f)]) for f in TONES],
        "grid": lesson_grid(ref),
    }


def verify(result):
    grid, n = result["grid"], result["n"]
    ratio = result["lesson_seconds"] / result["numpy_seconds"]
    worst = max(abs(h - result["predicted"]) for h in result["heights"])
    heights = [round(h, 4) for h in result["heights"]]
    bins, peaks = [round(e, 2) for e in grid["exact"]], [round(h, 1) for h in grid["hz"]]
    off_grid = [e for e in grid["exact"] if abs(e - round(e)) > 0.05]
    return [
        practice.Check(
            "ANSWER: three peaks at bins 220, 440 and 880",
            result["bins"] == [220, 440, 880] and n == 16000,
            f"one second at {SR} Hz is N={n}, so the bin width is {SR / n:g} Hz and bin index "
            f"equals frequency in Hz: the three largest half-spectrum bins are {result['bins']}",
        ),
        practice.Check(
            "ANSWER: every peak stands at 1333.333, the closed form to 13 digits",
            worst < 1e-9,
            f"predicted (amp/3) * N/2 = {result['predicted']:.4f}; measured {heights}, "
            f"worst deviation {worst:.2e}",
        ),
        practice.Check(
            "MECHANISM: `mix` averages, so three tones at amp 0.5 peak below 0.5",
            abs(result["peak_amp"] - AMP) > 0.1,
            f"`mix` divides by `len(signals)`, so each component arrives at "
            f"{AMP / len(TONES):.4f} and the mixed signal peaks at {result['peak_amp']:.4f}, not "
            f"{AMP * len(TONES):.2f}. Read it as a sum and every predicted height is 3x too big",
        ),
        practice.Check(
            "FINDING: the Easy exercise is the lesson's most expensive computation",
            ratio > 1000 and result["rel"] < 1e-9,
            f"the doc calls the O(N^2) DFT 'useless for real audio' at N=256 and then asks for "
            f"N={n}, which is {n * n / 512 / 512:.0f}x the work of `main()`'s largest call: "
            f"{result['lesson_seconds']:.1f} s against numpy.fft.rfft's "
            f"{result['numpy_seconds'] * 1000:.2f} ms ({ratio:.0f}x) for the same spectrum to "
            f"{result['rel']:.1e} relative",
        ),
        practice.Check(
            "FINDING: 'the expected bins' are integers only at the exercise's own parameters",
            len(off_grid) == len(TONES),
            f"on `main()`'s grid ({grid['n']} samples at {grid['sr']} Hz) the same three tones sit "
            f"at bins {bins} -- none of them an integer -- and the peaks land at {peaks} Hz, "
            "low by 1.25, 2.5 and 5.0 Hz",
        ),
        practice.Check(
            "CONTROL: each peak ties exactly with its mirror above Nyquist",
            max(result["mirror"]) < 1e-6,
            f"bins {n - 880}, {n - 440} and {n - 220} carry the same magnitudes to "
            f"{max(result['mirror']):.1e}, so a top-3 taken over the *full* transform returns the "
            f"right answer only because `sorted` breaks the tie by index. `main()` halves first",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
