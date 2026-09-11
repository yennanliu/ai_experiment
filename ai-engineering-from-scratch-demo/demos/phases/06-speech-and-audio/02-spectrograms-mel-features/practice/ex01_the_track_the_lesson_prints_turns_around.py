"""Exercise 1 — the track the lesson prints turns around.

    **Easy.** Run `code/main.py`. It synthesizes a chirp (frequency swept
    200 -> 4000 Hz) and prints the argmax mel bin per frame. Plot (optional) and
    confirm it matches the sweep.

Reading of the exercise: "confirm it matches the sweep" is a claim to check, not
an instruction to agree with. The check is whether the printed argmax mel bin is
monotone -- a linear upward sweep can produce nothing else -- and it is not.

`main()`'s own Step 5 output, all 24 frames, is

    11 15 20 23 26 28 30 33 35 36 38 39 | 39 38 36 35 33 31 28 26 23 20 15 11

which rises for twelve frames and comes back down for twelve, a **mirror of
itself to within one bin at one frame**. The lesson prints a V and calls it a
sweep.

Two compounding causes, both arithmetic.

**`chirp` puts the instantaneous frequency into the phase.** `sin(2*pi*f(t)*t)`
with `f(t) = f0 + (f1-f0)*t/T` has derivative `f0 + 2*(f1-f0)*t/T`, so the sweep
runs at **twice** the requested rate and ends at **7800 Hz**, not 4000. A linear
sweep needs quadratic phase.

**At `sr=8000` that doubled sweep crosses Nyquist mid-clip.** 4000 Hz is reached
at `t = 0.2 s`, exactly half of the 0.4 s clip, so the second half folds back
down -- and frame 12 of 24 is exactly where the printed track turns. The fold is
the mirror.

Fixing only the phase fixes the plot: the same `stft_magnitude`, the same
filterbank, the same `log_transform`, driven by a quadratic-phase chirp, gives
`8 11 13 ... 38 39 39` -- **monotone non-decreasing across all 24 frames**, ending
at the top mel bin because the corrected sweep ends exactly at Nyquist.

Structure: `mel_track` runs the lesson's whole Step 5 pipeline and returns the
argmax mel bin per frame; `linear_chirp` is the same sweep with the phase a sweep
actually needs; `descents` and `mirror_error` are the two ways of saying the
track is not monotone.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "02-spectrograms-mel-features"
SR, FRAME, HOP, N_MELS = 8000, 256, 128, 40
F_LO, F_HI, SECONDS = 200.0, 4000.0, 0.4


def mel_track(ref, signal):
    """The lesson's Step 5 verbatim: STFT -> filterbank -> log -> argmax mel."""
    bank = ref.mel_filterbank(N_MELS, FRAME, SR)
    spectra = ref.stft_magnitude(signal, FRAME, HOP)
    frames = ref.log_transform(ref.apply_filterbank(spectra, bank))
    return [max(range(N_MELS), key=lambda m: frame[m]) for frame in frames]


def linear_chirp(f0, f1, sr, seconds, amp=0.5):
    """The sweep the lesson means: quadratic phase, so df/dt is constant."""
    return [amp * math.sin(2 * math.pi * (f0 * (i / sr) + (f1 - f0) * (i / sr) ** 2
                                          / (2 * seconds))) for i in range(int(sr * seconds))]


def instantaneous(t):
    """What `chirp`'s phase actually differentiates to: double the intended rate."""
    return F_LO + 2 * (F_HI - F_LO) * t / SECONDS


def descents(track):
    """Frame indices where the argmax mel bin falls. A sweep has none."""
    return [i for i in range(1, len(track)) if track[i] < track[i - 1]]


def mirror_error(track):
    """How far the track is from being its own reflection, in mel bins."""
    return max(abs(a - b) for a, b in zip(track, reversed(track)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lesson = mel_track(ref, ref.chirp(F_LO, F_HI, SR, SECONDS))
    fixed = mel_track(ref, linear_chirp(F_LO, F_HI, SR, SECONDS))
    nyquist_at = (SR / 2 - F_LO) * SECONDS / (2 * (F_HI - F_LO))
    return {
        "lesson": lesson, "fixed": fixed,
        "falls": descents(lesson), "fixed_falls": descents(fixed),
        "mirror": mirror_error(lesson),
        "end_hz": instantaneous(SECONDS), "nyquist_at": nyquist_at,
        "turn": lesson.index(max(lesson)), "top": max(fixed),
    }


def verify(result):
    lesson, falls = result["lesson"], result["falls"]
    return [
        practice.Check(
            "ANSWER: it does not match the sweep -- the printed track turns around",
            len(falls) > 0 and result["mirror"] <= 1,
            f"the argmax mel bin falls on {len(falls)} of {len(lesson) - 1} frame transitions "
            f"({falls[0]} onward), and the whole track is its own mirror to "
            f"{result['mirror']} bin: {lesson[:12]} | {lesson[12:]}. A linear sweep is monotone",
        ),
        practice.Check(
            "MECHANISM: `chirp` puts the instantaneous frequency into the phase",
            abs(result["end_hz"] - 7800.0) < 1e-9,
            f"`sin(2*pi*f(t)*t)` with a linear f(t) differentiates to f0 + 2*(f1-f0)*t/T, so the "
            f"sweep runs at twice the requested rate and ends at {result['end_hz']:.0f} Hz "
            f"instead of {F_HI:.0f}. A linear sweep needs quadratic phase, not f(t)*t",
        ),
        practice.Check(
            "MECHANISM: at sr=8000 the doubled sweep crosses Nyquist at the midpoint",
            abs(result["nyquist_at"] - SECONDS / 2) < 1e-9 and abs(result["turn"] - 11) <= 1,
            f"the doubled sweep reaches {SR // 2} Hz at t={result['nyquist_at']:.3f} s, exactly "
            f"half of the {SECONDS} s clip, so the second half folds back down -- and the printed "
            f"track turns at frame {result['turn']} of {len(lesson) - 1}. The fold is the mirror",
        ),
        practice.Check(
            "CONTROL: fixing only the phase makes the same pipeline monotone",
            not result["fixed_falls"] and result["top"] == N_MELS - 1,
            f"the same `stft_magnitude`, filterbank and `log_transform` driven by a "
            f"quadratic-phase chirp never falls ({len(result['fixed_falls'])} descents) and ends "
            f"at mel {result['top']} of {N_MELS - 1}: {result['fixed'][:6]} ... "
            f"{result['fixed'][-4:]}, because the corrected sweep ends exactly at Nyquist",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
