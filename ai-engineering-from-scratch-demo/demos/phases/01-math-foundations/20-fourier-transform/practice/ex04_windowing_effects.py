"""Exercise 4 — 10 Hz and 12 Hz together: which window resolves them?

    **Windowing effects.** Create a signal that is the sum of two sine waves at
    10 Hz and 12 Hz (very close). Sample at 128 Hz for 1 second. Compute the
    power spectrum with no window, Hann window, and Hamming window. Which window
    makes it easiest to distinguish the two peaks? Why?

Reading of the exercise: the expected answer is a window, and the measurement
says the opposite. "Easiest to distinguish" is scored as **valley depth** — how
far the spectrum falls between the two peaks — and at 1 Hz bin spacing with
integer frequencies the rectangular window puts a numerically-zero valley between
them while **Hann merges them entirely** (valley depth 0.98x: the trough is higher
than the lower peak).

The reason is that a window trades main-lobe width for sidelobe suppression, and
here there are no sidelobes to suppress — each tone already occupies exactly one
bin. The trade is all cost. Check 5 runs 10 vs 11 Hz, one bin apart, where
nothing resolves them and no window can, because only a longer observation
narrows a bin.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "20-fourier-transform"
SAMPLE_RATE, N = 128, 128
PAIRS = {"10 and 12 Hz (2 bins apart)": (10.0, 12.0),
         "10 and 11 Hz (1 bin apart)": (10.0, 11.0)}


def analyse(ref, signal, window=None):
    data = ref.apply_window(signal, window) if window else signal
    spectrum = ref.power_spectrum(ref.fft(data))[: N // 2 + 1]
    return spectrum


def peak_and_valley(spectrum, low, high):
    """Peak heights at the two frequencies, and the lowest point between them."""
    peaks = (spectrum[int(low)], spectrum[int(high)])
    between = spectrum[int(low) + 1:int(high)]
    valley = min(between) if between else min(peaks)
    return {"peaks": peaks, "valley": valley,
            "depth": min(peaks) / valley if valley > 0 else float("inf")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "fourier")
    windows = {"none": None, "Hann": ref.hann_window(N), "Hamming": ref.hamming_window(N)}
    rows = {}
    for label, (low, high) in PAIRS.items():
        signal = ref.generate_signal([low, high], [1.0, 1.0], N, SAMPLE_RATE)
        rows[label] = {name: peak_and_valley(analyse(ref, signal, w), low, high)
                       for name, w in windows.items()}
    return {"rows": rows, "bin_width": SAMPLE_RATE / N}


def verify(result):
    asked = result["rows"]["10 and 12 Hz (2 bins apart)"]
    close = result["rows"]["10 and 11 Hz (1 bin apart)"]
    best_asked = max(asked, key=lambda k: asked[k]["depth"])
    best_close = max(close, key=lambda k: close[k]["depth"])
    return [
        practice.Check("FINDING: only the rectangular window resolves 10 and 12 Hz",
                       asked["none"]["depth"] > 1e10 and asked["Hann"]["depth"] < 1.1,
                       "valley depth (min peak / valley): " + ", ".join(
                           f"{k} {v['depth']:.3g}x" for k, v in asked.items())
                       + f". With no window the valley is numerically zero, so the depth is "
                         f"unbounded. Under Hann it is {asked['Hann']['depth']:.2f}x — the "
                         f"valley is *higher* than the lower peak, so the two tones merge "
                         f"into one blob. Windowing does not help here, it destroys the "
                         f"separation"),
        practice.Check(f"ANSWER on the signal as specified: no window is best "
                       f"({best_asked})",
                       best_asked == "none",
                       f"the rectangular window gives a {asked['none']['depth']:.3g}x "
                       f"valley against {asked['Hann']['depth']:.1f}x for Hann and "
                       f"{asked['Hamming']['depth']:.1f}x for Hamming"),
        practice.Check("…because windowing widens the main lobe",
                       asked["Hann"]["depth"] < asked["none"]["depth"]
                       and asked["Hamming"]["depth"] < asked["none"]["depth"],
                       "a window trades main-lobe width for sidelobe suppression. With "
                       "exactly integer frequencies there are no sidelobes to suppress — "
                       "each tone already occupies one bin — so the trade is all cost"),
        practice.Check("Hamming beats Hann here, as their lobe widths predict",
                       asked["Hamming"]["depth"] > asked["Hann"]["depth"],
                       f"Hamming {asked['Hamming']['depth']:.2f}x against Hann "
                       f"{asked['Hann']['depth']:.2f}x — Hamming's main lobe is narrower, "
                       f"at the price of higher sidelobes"),
        practice.Check(f"at 1 bin apart no treatment resolves them, windows included",
                       all(v["depth"] < 2 for v in close.values()),
                       "valley depth: " + ", ".join(f"{k} {v['depth']:.2f}x"
                                                    for k, v in close.items())
                       + f" (best: {best_close}). One bin of separation is below the "
                         f"rectangular window's own resolution limit, so no amount of "
                         f"windowing recovers it — that needs a longer observation, which "
                         f"is the only thing that narrows a bin"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
