"""Exercise 2 — the alias lands between the harmonics, never on them.

    **Medium.** Record a 3-second WAV of your voice at 48 kHz. Downsample to
    16 kHz using `torchaudio.transforms.Resample` (with anti-aliasing), then to
    16 kHz using naive decimation (every third sample). FFT both. Where does the
    aliasing appear?

Reading of the exercise: neither half of the setup exists here. `torchaudio`,
`torch`, `librosa`, `soundfile` and `sounddevice` are all absent, and the whole
reference tree ships no `.wav`, `.flac` or `.mp3`, so there is nothing to record
with and nothing to record. What survives is the question, which is about
arithmetic and not about a particular voice. So the 3 s at 48 kHz is synthesised
as a voice *would* be -- a 120 Hz glottal comb, 1/k voiced rolloff to 6 kHz, then
a flat sibilant shelf out to 22.9 kHz, 191 harmonics -- and the naive arm is the
lesson's own `downsample_naive(x, 3)` unchanged. The anti-aliased arm is a
193-tap windowed-sinc low-pass built here: **0.00 dB at 4 kHz, -112 dB at 12 kHz**.

Where does the aliasing appear? **Mirrored about the new Nyquist.** Every one of
the 125 components above 8 kHz reappears at `|f - 16000*round(f/16000)|`, all 125
of them, and the loudest land at **7240-7960 Hz** -- the top of the retained band,
because that is where the components just above Nyquist fold to. Aliasing works
downward from 8 kHz, not upward from DC.

The sharper answer is *between* which harmonics. `16000 mod 120 = 40`, so a
component at `120k` folds to a frequency `40 Hz` (from above Nyquist) or `80 Hz`
(from above 16 kHz) off the 120 Hz grid -- the offsets measured across all 125 are
exactly `{40, 80}` and **never 0**. No aliased partial can ever reinforce a real
harmonic; it triples the comb density instead. That is the difference between
sounding bright and sounding metallic, and it is why "it just adds some high-end
crud" is the wrong intuition.

Nothing is lost, either: decimating without a filter is energy-preserving in the
wrong place. The source carries **5.45%** of its energy above 8 kHz; the naive
arm's band carries **6.1%** more energy than the filtered arm's, and the two
signals differ by **24%** relative RMS.

Structure: `voice` builds the 48 kHz signal; `lowpass` is the windowed sinc;
`spectrum` returns (freqs, magnitudes); `fold` is the closed-form alias location;
`gain_db` measures the filter's response at one frequency.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "01-audio-fundamentals"
SR, TARGET, FACTOR, SECONDS = 48000, 16000, 3, 3.0
F0, CUTOFF, TAPS = 120.0, 7600.0, 193
ABSENT = ("torchaudio", "torch", "librosa", "soundfile", "sounddevice")


def voice(sr, seconds):
    """A 120 Hz glottal comb with a 1/k voiced rolloff and a flat sibilant shelf."""
    t = np.arange(int(sr * seconds)) / sr
    signal = np.zeros_like(t)
    for k in range(1, int(23000 // F0) + 1):
        amp = 1.0 / k if k * F0 <= 6000 else 0.30 / math.sqrt(k)
        signal += amp * np.sin(2 * np.pi * F0 * k * t)
    return 0.5 * signal / np.max(np.abs(signal))


def lowpass(x, cutoff, sr, taps=TAPS):
    """The anti-aliasing `torchaudio.transforms.Resample` would have applied."""
    half = (taps - 1) // 2
    k = np.arange(-half, half + 1)
    kernel = 2 * cutoff / sr * np.sinc(2 * cutoff / sr * k) * np.hanning(taps)
    return np.convolve(x, kernel / kernel.sum(), mode="same")


def spectrum(x, sr):
    return np.fft.rfftfreq(len(x), 1 / sr), np.abs(np.fft.rfft(x))


def fold(f, sr):
    """Where a component at `f` lands once sampled at `sr`: mirrored about sr/2."""
    return abs(f - sr * round(f / sr))


def gain_db(sr, f, seconds=0.25):
    """The filter's response at one frequency, edges trimmed."""
    t = np.arange(int(sr * seconds)) / sr
    tone = np.sin(2 * np.pi * f * t)
    edge = TAPS
    ratio = np.sqrt(np.mean(lowpass(tone, CUTOFF, sr)[edge:-edge] ** 2)) / np.sqrt(0.5)
    return 20 * math.log10(max(ratio, 1e-12))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = voice(SR, SECONDS)
    naive = np.asarray(ref.downsample_naive(list(source), FACTOR))
    clean = lowpass(source, CUTOFF, SR)[::FACTOR]
    freqs, naive_mag = spectrum(naive, TARGET)
    _, clean_mag = spectrum(clean, TARGET)
    src_f, src_mag = spectrum(source, SR)
    step = TARGET / len(naive)
    above = [k * F0 for k in range(1, int(23000 // F0) + 1) if k * F0 > TARGET / 2]
    visible = [f for f in above
               if naive_mag[max(0, int(round(fold(f, TARGET) / step)) - 2):
                            int(round(fold(f, TARGET) / step)) + 3].max() > 0.02 * naive_mag.max()]
    extra = naive_mag - clean_mag
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "above": len(above), "visible": len(visible),
        "loudest": sorted(round(float(freqs[i]), 1) for i in np.argsort(-extra)[:8]),
        "offsets": sorted({round(fold(f, TARGET)) % int(F0) for f in above}),
        "hf_share": float(np.sum(src_mag[src_f > TARGET / 2] ** 2) / np.sum(src_mag**2)),
        "energy_ratio": float(np.sum(naive_mag**2) / np.sum(clean_mag**2)),
        "rms": float(np.sqrt(np.mean((naive - clean) ** 2)) / np.sqrt(np.mean(clean**2))),
        "pass_db": gain_db(SR, 4000.0), "stop_db": gain_db(SR, 12000.0),
        "naive_is_reference": naive.size == len(source) // FACTOR,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the alias appears mirrored about the new Nyquist, at |f - 16000k|",
            result["visible"] == result["above"] == 125,
            f"all {result['visible']} of the {result['above']} components above "
            f"{TARGET // 2} Hz reappear at the predicted fold, and the loudest land at "
            f"{result['loudest'][0]}-{result['loudest'][-1]} Hz -- the *top* of the retained "
            "band, because the partials just above Nyquist fold just below it",
        ),
        practice.Check(
            "ANSWER: every alias sits 40 or 80 Hz off the harmonic grid, never on it",
            result["offsets"] == [40, 80],
            f"16000 mod {F0:.0f} = {TARGET % int(F0)}, so a partial at 120k folds to an offset of "
            f"{result['offsets']} Hz from the 120 Hz comb and never to 0. Aliasing cannot "
            "reinforce a real harmonic -- it triples the comb density, which is why naive "
            "decimation sounds metallic rather than bright",
        ),
        practice.Check(
            "MECHANISM: nothing is lost, it is moved -- decimation conserves energy",
            result["energy_ratio"] > 1.0 and result["rms"] > 0.1,
            f"the source carries {result['hf_share'] * 100:.2f}% of its energy above "
            f"{TARGET // 2} Hz; the naive band holds {(result['energy_ratio'] - 1) * 100:.1f}% "
            f"more energy than the filtered one and the two differ by {result['rms'] * 100:.1f}% "
            "relative RMS. The filter is what removes it; dropping samples only relocates it",
        ),
        practice.Check(
            "CONTROL: the naive arm is the lesson's own `downsample_naive`, unedited",
            result["naive_is_reference"],
            f"`downsample_naive(x, {FACTOR})` is `x[::{FACTOR}]` and ships with no filter beside "
            "it. `main()`'s Step 6 prints 'always low-pass filter before decimating' and the "
            "module contains nothing that does -- the lesson ships the broken half only",
        ),
        practice.Check(
            "CONTROL: the anti-aliased arm is built here, because torchaudio is absent",
            len(result["absent"]) == len(ABSENT) and result["stop_db"] < -60 < result["pass_db"],
            f"find_spec is None for {result['absent']}, and no .wav/.flac/.mp3 ships anywhere in "
            f"the reference tree, so neither 'record' nor 'Resample' is available. The {TAPS}-tap "
            f"windowed sinc standing in measures {result['pass_db']:.2f} dB at 4 kHz and "
            f"{result['stop_db']:.0f} dB at 12 kHz",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
