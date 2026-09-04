"""Exercise 2 — FFT against DFT: identical output, and the timing ratio.

    **FFT vs DFT verification.** Generate a random signal of length 64. Compute
    both DFT (O(N^2)) and FFT. Verify that all coefficients match to within
    1e-10. Time both functions on signals of length 256, 512, 1024, and 2048.
    Plot the ratio of DFT time to FFT time.

Reading of the exercise: "plot the ratio" is printed as a table, and the claim
the ratio encodes is testable — DFT/FFT time should grow like N/log₂N, so
doubling N roughly doubles the ratio. Check 4 fits the growth exponents
separately rather than trusting the ratio, since a ratio can grow correctly while
both terms are wrong. The exercise's 1e-10 tolerance is about
three orders looser than the measured 1.3e-13; the measurement is reported.

Tier T1: the N=2048 DFT is 4.2 million pure-Python complex multiplies.
"""

from __future__ import annotations

import math
import random
import time

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "20-fourier-transform"
SEED = 20260904
SIZES = (256, 512, 1024, 2048)


def timed(fn):
    start = time.perf_counter()
    return fn(), time.perf_counter() - start


def solve():
    ref = parity.load_reference(PHASE, LESSON, "fourier")
    rng = random.Random(SEED)
    small = [rng.uniform(-1, 1) for _ in range(64)]
    dft_small, fft_small = ref.dft(small), ref.fft(small)
    worst = max(max(abs(a.real - b.real), abs(a.imag - b.imag))
                for a, b in zip(dft_small, fft_small))
    rows = {}
    for n in SIZES:
        signal = [rng.uniform(-1, 1) for _ in range(n)]
        _, dft_time = timed(lambda: ref.dft(signal))
        _, fft_time = timed(lambda: ref.fft(signal))
        rows[n] = {"dft": dft_time, "fft": fft_time, "ratio": dft_time / fft_time,
                   "predicted": n / math.log2(n)}
    return {"worst": worst, "rows": rows, "n_small": len(small)}


def _exponent(rows, key):
    big, small = SIZES[-1], SIZES[0]
    return math.log(rows[big][key] / rows[small][key]) / math.log(big / small)


def verify(result):
    rows = result["rows"]
    ratios = [rows[n]["ratio"] for n in SIZES]
    dft_p, fft_p = _exponent(rows, "dft"), _exponent(rows, "fft")
    predicted_growth = rows[SIZES[-1]]["predicted"] / rows[SIZES[0]]["predicted"]
    return [
        practice.Check(f"all {result['n_small']} coefficients match, far inside 1e-10",
                       result["worst"] < 1e-12,
                       f"worst |Δ| = {result['worst']:.3g}, about "
                       f"{1e-10 / result['worst']:.0f}x inside the exercise's 1e-10 bar"),
        practice.Check("FFT is faster at every size, and increasingly so",
                       all(r > 1 for r in ratios) and ratios[0] < ratios[-1],
                       "; ".join(f"N={n}: DFT {rows[n]['dft'] * 1e3:.0f} ms, FFT "
                                 f"{rows[n]['fft'] * 1e3:.1f} ms, {rows[n]['ratio']:.0f}x"
                                 for n in SIZES)),
        practice.Check("DFT scales as N² and FFT as N log N",
                       1.8 < dft_p < 2.2 and 1.0 < fft_p < 1.5,
                       f"fitted exponents over N = {SIZES[0]}..{SIZES[-1]}: DFT "
                       f"N^{dft_p:.2f}, FFT N^{fft_p:.2f}. Fitting them separately matters "
                       f"— a ratio can grow correctly while both terms are wrong"),
        practice.Check("…so the ratio grows like N/log₂N",
                       0.5 < (ratios[-1] / ratios[0]) / predicted_growth < 2.0,
                       f"ratio grows {ratios[-1] / ratios[0]:.1f}x from N={SIZES[0]} to "
                       f"N={SIZES[-1]}, against N/log₂N growing "
                       f"{predicted_growth:.1f}x"),
        practice.Check("the absolute cost is what makes the FFT historically important",
                       rows[SIZES[-1]]["dft"] > 1.0,
                       f"at N={SIZES[-1]} the DFT takes "
                       f"{rows[SIZES[-1]]['dft']:.1f} s against "
                       f"{rows[SIZES[-1]]['fft'] * 1e3:.0f} ms — a {SIZES[-1]}-point "
                       f"transform is small by any modern standard, and the naive form is "
                       f"already unusable in a loop"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
