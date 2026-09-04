"""Exercise 3 — DFT of sin(2π·3t) + 0.5·sin(2π·7t) at 32 points.

    **DFT of a known signal.** Create a signal that is the sum of sin(2*pi*3*t)
    and 0.5*sin(2*pi*7*t) sampled at 32 points. Run your DFT. Verify that the
    magnitude spectrum has peaks at frequencies 3 and 7, with the peak at 7 being
    half the height of the peak at 3.

Reading of the exercise: the 2:1 ratio only holds because 3 and 7 are **integer**
frequencies over the sampling window, so each sinusoid lands entirely in one bin
with no spectral leakage. Check 4 shows what happens at 3.5 cycles, where the
energy smears across bins and the ratio claim stops being meaningful — which is
the reason windowing exists. Check 5 notes the mirror bins at 32−k, which a
one-sided reading of the spectrum would miss.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "19-complex-numbers"
N = 32
TOL = 1e-9


def make_signal(frequencies):
    return [sum(amp * math.sin(2 * math.pi * freq * t / N)
                for freq, amp in frequencies) for t in range(N)]


def magnitudes(ref, signal):
    return [z.magnitude() for z in ref.dft(signal)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "complex_numbers")
    clean = magnitudes(ref, make_signal([(3, 1.0), (7, 0.5)]))
    leaky = magnitudes(ref, make_signal([(3.5, 1.0)]))
    signal = make_signal([(3, 1.0), (7, 0.5)])
    rebuilt = ref.idft(ref.dft(signal))
    peaks = sorted(range(N // 2 + 1), key=lambda k: clean[k], reverse=True)[:2]
    leak_peak = max(range(N // 2 + 1), key=lambda k: leaky[k])
    return {
        "clean": clean, "leaky": leaky,
        "peaks": sorted(peaks),
        "ratio": clean[3] / clean[7],
        "mirror_gap": max(abs(clean[k] - clean[N - k]) for k in range(1, N // 2)),
        "floor": max(clean[k] for k in range(N // 2 + 1) if k not in (3, 7)),
        "leak_peak": leak_peak,
        "leak_spread": sum(1 for k in range(N // 2 + 1)
                           if leaky[k] > 0.1 * leaky[leak_peak]),
        "roundtrip": max(abs(z.real - s) for z, s in zip(rebuilt, signal)),
    }


def verify(result):
    clean = result["clean"]
    return [
        practice.Check("the magnitude spectrum peaks at bins 3 and 7",
                       result["peaks"] == [3, 7],
                       f"|X[3]| = {clean[3]:.4f}, |X[7]| = {clean[7]:.4f}; every other bin "
                       f"below {result['floor']:.2e}"),
        practice.Check("the peak at 7 is exactly half the peak at 3",
                       abs(result["ratio"] - 2.0) < TOL,
                       f"ratio {result['ratio']:.12f} — exact, because the amplitudes were "
                       f"1.0 and 0.5 and each frequency is an integer number of cycles "
                       f"over the {N}-point window"),
        practice.Check("the DFT round-trips through idft",
                       result["roundtrip"] < 1e-12,
                       f"worst |idft(dft(x)) − x| = {result['roundtrip']:.3g}"),
        practice.Check("CAUSE: at a non-integer frequency the energy leaks across bins",
                       result["leak_spread"] > 4,
                       f"a 3.5-cycle sinusoid puts its peak at bin {result['leak_peak']} but "
                       f"spreads above 10% of it across {result['leak_spread']} bins — the "
                       f"exercise's clean 2:1 ratio depends entirely on 3 and 7 being "
                       f"integers, and windowing exists to manage the case where they are "
                       f"not"),
        practice.Check("…and each peak has a mirror at N−k, since the signal is real",
                       result["mirror_gap"] < TOL,
                       f"|X[k]| = |X[{N}−k]| to {result['mirror_gap']:.3g}, so bins "
                       f"{N - 3} and {N - 7} carry equal magnitude. Half the spectrum is "
                       f"redundant for real input, which a one-sided reading would miss"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
