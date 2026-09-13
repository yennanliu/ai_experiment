"""Exercise 2 — nineteen of the eighty mel filters are narrower than an FFT bin.

    **Medium.** Build the full log-mel spectrogram using `numpy.fft`. Verify 80
    mel bins match `librosa.feature.melspectrogram(n_mels=80)` within numerical
    error.

Reading of the exercise: `librosa` returns None from `find_spec` -- so does
`torch`, `torchaudio` and `soundfile` -- so there is nothing to compare against.
`numpy` is present, so the spectrogram is built at Whisper's own settings
(`n_fft=400`, `hop=160`, 80 mels over 0-8000 Hz) and verified against properties
a correct filterbank must have rather than against a library that is not here.

**ANSWER: the spectrogram is (80, 98) for one second, and it is correct on every
property that can be checked without librosa.** The filterbank is `(80, 201)`;
the 80 triangles sum to **1.0000** at every FFT bin in their interior, a
partition of unity, so the transform conserves total power to **0.9997**; and a
440 Hz tone peaks at mel bin **15**, which is where 440 Hz falls in the mel
centres. The frame count matches the lesson's own `frame_signal` exactly.

**FINDING: 19 of the 80 filters are narrower than two FFT bins.** `n_fft=400` at
16 kHz gives 201 bins of **40 Hz**. The first mel triangle spans 44.9 Hz -- 1.12
bins -- so it touches one or two samples of the spectrum and is not a triangle in
any meaningful sense. A quarter of Whisper's mel resolution is finer than the
spectrogram it is computed from, and that is a property of Whisper's published
settings, not of this implementation.

**FINDING: 28 of the 80 bins cover the first eighth of the band.** Mel centres
below 1 kHz number 28, and `mel(1000) / mel(8000)` is **35.2%** -- so 35% of the
bins describe 12.5% of the spectrum. That is the mel scale doing its job, and it
is why the filters at the bottom are narrower than the FFT can resolve.

**FINDING: `frame_energy` tracks total power at r = 0.998 and keeps none of the
shape.** On an amplitude-modulated tone the lesson's one-number stand-in follows
the log total mel power almost exactly -- so it is an excellent measure of *how
loud* and carries nothing about *which frequencies*. The sequence the transformer
sees is `(3000, 1)` here and `(3000, 80)` in Whisper.

Structure: `filterbank` builds the 80 triangles; `log_mel` is the spectrogram;
`narrow` counts filters against the FFT's own resolution.
"""

from __future__ import annotations

import importlib.util
import math

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "10-audio-transformers-whisper"
RATE, N_FFT, HOP, MELS, FMAX = 16_000, 400, 160, 80, 8_000
ABSENT = ("librosa", "torch", "torchaudio", "soundfile")


def mel(hz):
    """The HTK mel scale, which is what n_mels is counted on."""
    return 2595 * math.log10(1 + hz / 700)


def hz(mels):
    """Its inverse."""
    return 700 * (10 ** (mels / 2595) - 1)


def filterbank(np, n_mels=MELS, n_fft=N_FFT, rate=RATE, fmax=FMAX):
    """(bank, fft frequencies, band edges) -- n_mels triangles over the rfft grid."""
    edges = hz(np.linspace(0, mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    rising = (freqs[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    falling = (edges[2:, None] - freqs[None, :]) / (edges[2:, None] - edges[1:-1, None])
    return np.clip(np.minimum(rising, falling), 0, None), freqs, edges


def power(np, signal, window=True):
    """Framed magnitude-squared spectrum, on the lesson's own frame grid."""
    taper = np.hanning(N_FFT + 1)[:-1] if window else np.ones(N_FFT)
    frames = np.stack([signal[s:s + N_FFT] * taper
                       for s in range(0, len(signal) - N_FFT + 1, HOP)])
    return np.abs(np.fft.rfft(frames, axis=-1)) ** 2


def log_mel(np, bank, signal):
    """log10 of the mel-weighted power, (n_mels, frames)."""
    return np.log10(np.maximum(bank @ power(np, signal).T, 1e-10))


def narrow(np, edges, bins=2, rate=RATE, n_fft=N_FFT):
    """How many triangles span fewer than `bins` FFT bins."""
    width = rate / n_fft
    return int(((edges[2:] - edges[:-2]) < bins * width).sum()), float(edges[2] - edges[0]), width


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    bank, freqs, edges = filterbank(np)
    tone = np.array(ref.sine_wave(440, 1.0))
    spectrogram = log_mel(np, bank, tone)
    interior = bank.sum(0)[(freqs > edges[1]) & (freqs < edges[-2])]
    time, centres = np.arange(RATE) / RATE, edges[1:-1]
    shaped = (0.2 + np.abs(np.sin(2 * np.pi * 3 * time))) * np.sin(2 * np.pi * 440 * time)
    spectra = power(np, shaped)
    energies = [ref.frame_energy(frame) for frame in ref.frame_signal(list(shaped))]
    return {
        "shape": spectrogram.shape, "bank": bank.shape,
        "frames": len(ref.frame_signal(list(tone))),
        "unity": (float(interior.min()), float(interior.max())),
        "conserved": float(((bank @ spectra.T).sum(0) / spectra.sum(1)).mean()),
        "peak": int(spectrogram[:, 10].argmax()),
        "expected": int(np.argmin(np.abs(centres - 440))),
        "narrow": narrow(np, edges), "low": int((centres < 1000).sum()),
        "share": mel(1000) / mel(FMAX),
        "correlation": float(np.corrcoef(np.array(energies),
                                         np.log((bank @ spectra.T).sum(0)))[0, 1]),
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
    }


def verify(result):
    low, high = result["unity"]
    count, first, width = result["narrow"]
    return [
        practice.Check(
            "ANSWER: correct on every property checkable without librosa",
            abs(low - 1) < 1e-9 and abs(high - 1) < 1e-9
            and result["peak"] == result["expected"],
            f"spectrogram {result['shape']} from a {result['bank']} bank, frame count matching "
            f"the lesson's own frame_signal at {result['frames']}. The triangles sum to "
            f"{low:.4f} at every interior FFT bin -- a partition of unity -- so power is "
            f"conserved to {result['conserved']:.4f}, and a 440 Hz tone peaks at mel bin "
            f"{result['peak']}, where 440 Hz falls among the centres",
        ),
        practice.Check(
            "FINDING: 19 of the 80 filters are narrower than two FFT bins",
            count == 19 and first < 2 * width,
            f"n_fft={N_FFT} at {RATE} Hz gives bins of {width:.0f} Hz. The first mel triangle "
            f"spans {first:.1f} Hz = {first / width:.2f} bins, and {count} of {MELS} span fewer "
            f"than two -- a quarter of Whisper's mel resolution is finer than the spectrogram it "
            "is computed from. That is a property of Whisper's published settings",
        ),
        practice.Check(
            "FINDING: 28 of the 80 bins cover the first eighth of the band",
            result["low"] == 28 and abs(result["share"] - 0.352) < 0.01,
            f"{result['low']} mel centres sit below 1 kHz, and mel(1000)/mel({FMAX}) is "
            f"{result['share']:.1%} -- so {result['low'] / MELS:.0%} of the bins describe "
            f"{1000 / FMAX:.1%} of the spectrum. That is the mel scale working as designed, and "
            "it is why the bottom filters are narrower than the FFT can resolve",
        ),
        practice.Check(
            "FINDING: frame_energy tracks total power and keeps none of the shape",
            result["correlation"] > 0.99,
            f"on an amplitude-modulated tone the lesson's one-number stand-in follows the log "
            f"total mel power at r = {result['correlation']:.4f}. An excellent measure of how "
            f"loud, carrying nothing about which frequencies: (3000, 1) against (3000, {MELS})",
        ),
        practice.Check(
            "CONTROL: there is nothing to compare against",
            result["absent"] == list(ABSENT),
            f"find_spec is None for {result['absent']}, so "
            "librosa.feature.melspectrogram(n_mels=80) cannot be called at all. numpy.fft is "
            "present, which is why the spectrogram is buildable and its invariants checkable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
