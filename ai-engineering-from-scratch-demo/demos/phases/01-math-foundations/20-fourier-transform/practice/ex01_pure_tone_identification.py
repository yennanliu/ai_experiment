"""Exercise 1 — find an unknown tone's frequency, then add noise.

    **Pure tone identification.** Create a signal with a single sine wave at an
    unknown frequency (between 1 and 50 Hz), sampled at 128 Hz for 1 second. Use
    your DFT to identify the frequency. Verify the answer matches. Now add
    Gaussian noise with standard deviation 0.5 and repeat. How does noise affect
    the spectrum?

Reading of the exercise: 128 samples over 1 second gives 1 Hz bin spacing, so an
integer frequency is recoverable *exactly*, and the noise question gets a precise
answer rather than "it gets messier".

At σ=0.5 the peak does not move at all — noise raises the **floor**, from 6e-14
to 13.8, cutting the peak-to-floor ratio to 4.2x. The arithmetic predicts that:
a unit sine gives a peak of N/2 = 64, while white noise of standard deviation σ
gives about σ√(N/2) = 8σ per bin, and the largest of 64 such bins is a few times
that. Setting the two equal puts the breaking point near σ ≈ 3, and the measured
sweep loses the tone at **σ = 2** — much sooner than "the DFT averages noise
away" would suggest.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "20-fourier-transform"
SAMPLE_RATE, N, SEED = 128, 128, 20260904
TRUE_FREQ = 17.0                       # "unknown" to the identifier, integer so exact
NOISE_SIGMA = 0.5


def identify(ref, signal):
    freqs, magnitudes = ref.spectral_analysis(signal, SAMPLE_RATE)
    peak = max(range(1, len(magnitudes)), key=lambda k: magnitudes[k])
    floor = max(m for k, m in enumerate(magnitudes) if abs(k - peak) > 2 and k > 0)
    return {"freq": freqs[peak], "peak": magnitudes[peak], "floor": floor,
            "ratio": magnitudes[peak] / floor if floor else float("inf")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "fourier")
    clean = ref.generate_signal([TRUE_FREQ], [1.0], N, SAMPLE_RATE)
    rng = random.Random(SEED)
    noisy = [v + rng.gauss(0, NOISE_SIGMA) for v in clean]
    sweep = {}
    for sigma in (0.5, 2.0, 5.0, 10.0, 20.0):
        rng = random.Random(SEED)
        signal = [v + rng.gauss(0, sigma) for v in clean]
        sweep[sigma] = identify(ref, signal)
    first_failure = next((s for s in sorted(sweep)
                          if abs(sweep[s]["freq"] - TRUE_FREQ) > 0.5), None)
    return {"clean": identify(ref, clean), "noisy": identify(ref, noisy),
            "sweep": sweep, "first_failure": first_failure,
            "bin_width": SAMPLE_RATE / N}


def verify(result):
    clean, noisy, sweep = result["clean"], result["noisy"], result["sweep"]
    return [
        practice.Check(f"the clean tone is identified exactly as {TRUE_FREQ:g} Hz",
                       clean["freq"] == TRUE_FREQ,
                       f"peak at {clean['freq']:g} Hz with bin width "
                       f"{result['bin_width']:g} Hz — an integer frequency lands in one bin, "
                       f"so there is no interpolation to do"),
        practice.Check("the clean spectrum has essentially no floor",
                       clean["ratio"] > 1e10,
                       f"peak {clean['peak']:.2f} against a floor of "
                       f"{clean['floor']:.3g}, ratio {clean['ratio']:.2g}"),
        practice.Check(f"ANSWER: σ={NOISE_SIGMA} noise does not move the peak at all",
                       noisy["freq"] == TRUE_FREQ,
                       f"still {noisy['freq']:g} Hz — noise is spread across all 65 bins "
                       f"while the tone's energy stays concentrated in one"),
        practice.Check("…it raises the floor, and that is what costs the margin",
                       3 < noisy["ratio"] < 8,
                       f"the peak barely changes ({clean['peak']:.2f} -> "
                       f"{noisy['peak']:.2f}) while the floor goes from "
                       f"{clean['floor']:.2g} to {noisy['floor']:.2f}, so the ratio falls "
                       f"from {clean['ratio']:.1g} to {noisy['ratio']:.1f}x"),
        practice.Check(f"…and identification breaks at σ = {result['first_failure']:g}, "
                       f"about where the arithmetic says it should",
                       result["first_failure"] == 2.0,
                       "σ -> ratio @ identified frequency: " + ", ".join(
                           f"{s:g} -> {sweep[s]['ratio']:.1f}x @ {sweep[s]['freq']:g} Hz"
                           for s in sorted(sweep))
                       + f". Per-bin noise magnitude is about σ√(N/2) = {8:.0f}σ against a "
                         f"peak of N/2 = {N // 2}, and the largest of {N // 2} noisy bins "
                         f"is a few times the average — so the crossover lands near σ ≈ 3. "
                         f"The DFT's √N ≈ {math.sqrt(N):.1f} coherent gain buys real margin, "
                         f"but far less than it sounds"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
