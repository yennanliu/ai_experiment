"""Exercise 3 — the Hard one is sixteen times cheaper than the Easy one.

    **Hard.** Build the STFT from scratch using only `math` and the DFT from
    Step 3. Frame size 400, hop 160, Hann window. Plot magnitudes with
    `matplotlib.pyplot.imshow`. This is the spectrogram of Lesson 02.

Reading of the exercise: "only `math` and the DFT from Step 3" is taken at its
word -- no numpy anywhere in this file, the transform is the lesson's own `dft`,
and the only thing built here is the framing, the window and the picture. The
signal is a 1 s linear chirp from 200 Hz to 4 kHz at 16 kHz, synthesised here
with the quadratic phase a sweep actually needs.

It works: **98 frames x 201 bins** at 40 Hz per bin, and the per-frame argmax
tracks the sweep to **19.5 Hz worst case**, inside half a bin. `matplotlib` is
absent, so `imshow` becomes a text raster of the same matrix, printed in the
lesson README.

Three things the exercise's own numbers say, none of them in the doc.

**It is the cheapest of the three exercises, and it is labelled Hard.** 98 frames
of 400 samples is `98 * 400^2 = 15.7M` inner DFT steps against Exercise 1's
`16000^2 = 256M` -- **16.3x less** work, **1.4 s against 24 s** measured. Framing
is what makes the O(N^2) DFT survivable, which is the actual lesson of the STFT
and is not what the exercise says it is about.

**400 / 160 is not a COLA pair.** Overlap-adding Hann(400) at hop 160 sums to
1.25 with **9.44% ripple**, so the spectrogram this builds cannot be inverted by
plain overlap-add. Hop 200 (`N/2`) and hop 100 (`N/4`) both sum flat to
**0.0000%**. The prescribed hop is the one in its own neighbourhood that is not
invertible, and 400/160 is exactly the doc's "25 ms with 10 ms hop".

**It is not the spectrogram of Lesson 02.** That lesson's `main` ships
`frame_len=256, hop=128` at `sr=8000`, a power-of-two frame at `N/2` hop, and its
`hann` uses the symmetric `N-1` denominator rather than the periodic `N` one.
Every parameter in the sentence is different from the lesson it points at.

Structure: `chirp` is the sweep; `hann` the periodic window; `stft` frames and
transforms with the lesson's `dft`; `cola` overlap-adds a window at one hop and
reports the ripple; `raster` turns the magnitude matrix into text; `lesson_two`
reads the next lesson's own frame settings out of its `main`.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import time

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "01-audio-fundamentals"
SR, FRAME, HOP = 16000, 400, 160
F_LO, F_HI, SECONDS = 200.0, 4000.0, 1.0
SHADES = " .:-=+*#%@"


def chirp(f0, f1, sr, seconds, amp=0.5):
    """A linear sweep: instantaneous frequency needs quadratic phase, not f(t)*t."""
    return [amp * math.sin(2 * math.pi * (f0 * (i / sr) + (f1 - f0) * (i / sr) ** 2
                                          / (2 * seconds))) for i in range(int(sr * seconds))]


def hann(n):
    """The periodic Hann window -- denominator `n`, the one that overlap-adds flat."""
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / n) for i in range(n)]


def stft(ref, signal, frame, hop):
    """Framed, windowed magnitudes from the lesson's own `dft`. One row per frame."""
    window = hann(frame)
    return [ref.magnitudes(ref.dft([signal[s + j] * window[j] for j in range(frame)]))
            [: frame // 2 + 1] for s in range(0, len(signal) - frame + 1, hop)]


def cola(frame, hop, periods=12):
    """Overlap-add ripple of Hann(`frame`) at `hop`, as (mean, fractional ripple)."""
    total = [0.0] * (frame * periods)
    window = hann(frame)
    for start in range(0, frame * periods - frame, hop):
        for j in range(frame):
            total[start + j] += window[j]
    steady = total[frame * 4: frame * 8]
    mean = sum(steady) / len(steady)
    return mean, (max(steady) - min(steady)) / mean


def shade(value, floor=-60.0):
    """One dB value as one character, clipped to the shading ramp."""
    return SHADES[min(len(SHADES) - 1, max(0, int((value - floor) / -floor * len(SHADES))))]


def raster(matrix, rows=20):
    """`imshow` as text: dB relative to the peak, low frequency at the bottom."""
    peak = max(max(row) for row in matrix)
    group = len(matrix[0]) // 2 // rows
    band = [[20 * math.log10(max(row[b], 1e-9) / peak) for row in matrix]
            for b in range(rows * group)]
    lines = [[max(column) for column in zip(*band[r * group:(r + 1) * group])]
             for r in range(rows)]
    return "\n".join("".join(shade(v) for v in line) for line in reversed(lines))


def lesson_two():
    """The next lesson's own frame settings, read out of its `main` rather than retyped."""
    module = parity.load_reference(PHASE, "02-spectrograms-mel-features", "main")
    tree = ast.parse(inspect.getsource(module.main))
    named = {n.targets[0].id: n.value.value for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and isinstance(n.value, ast.Constant)}
    return named, "N - 1" if "(N - 1)" in inspect.getsource(module.hann) else "N"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    signal = chirp(F_LO, F_HI, SR, SECONDS)
    start = time.perf_counter()
    matrix = stft(ref, signal, FRAME, HOP)
    seconds = time.perf_counter() - start
    centres = [(s + FRAME / 2) / SR for s in range(0, len(signal) - FRAME + 1, HOP)]
    errors = [abs(max(range(len(row)), key=lambda i: row[i]) * SR / FRAME
                  - (F_LO + (F_HI - F_LO) * c / SECONDS)) for row, c in zip(matrix, centres)]
    named, denominator = lesson_two()
    return {
        "frames": len(matrix), "bins": len(matrix[0]), "seconds": seconds,
        "worst_hz": max(errors), "steps": len(matrix) * FRAME * FRAME, "ex1_steps": SR * SR,
        "cola": {hop: cola(FRAME, hop) for hop in (100, HOP, 200)},
        "two": named, "denominator": denominator, "raster": raster(matrix),
        "plotting": [m for m in ("matplotlib", "PIL") if importlib.util.find_spec(m) is None],
    }


def verify(result):
    ripple = result["cola"]
    two = result["two"]
    return [
        practice.Check(
            "ANSWER: 98 frames x 201 bins, and the argmax tracks the sweep inside half a bin",
            result["frames"] == 98 and result["bins"] == 201 and result["worst_hz"] < SR / FRAME / 2,
            f"frame {FRAME} / hop {HOP} at {SR} Hz is 25 ms every 10 ms -> {result['frames']} "
            f"frames of {result['bins']} bins at {SR / FRAME:.0f} Hz each; the per-frame peak "
            f"follows the 200->4000 Hz sweep to {result['worst_hz']:.1f} Hz worst case",
        ),
        practice.Check(
            "FINDING: the Hard exercise is 16x cheaper than the Easy one",
            result["ex1_steps"] > 16 * result["steps"],
            f"{result['frames']} frames of {FRAME} is {result['steps'] / 1e6:.1f}M inner DFT "
            f"steps against Exercise 1's {result['ex1_steps'] / 1e6:.0f}M -- "
            f"{result['ex1_steps'] / result['steps']:.1f}x less, {result['seconds']:.1f} s "
            "against 24 s. Framing is what makes the O(N^2) DFT survivable",
        ),
        practice.Check(
            "FINDING: Hann(400) at hop 160 is not a COLA pair",
            ripple[HOP][1] > 0.09 and max(ripple[100][1], ripple[200][1]) < 1e-9,
            f"overlap-adding at hop {HOP} sums to {ripple[HOP][0]:.3f} with "
            f"{ripple[HOP][1] * 100:.2f}% ripple, so this spectrogram is not invertible by plain "
            f"overlap-add; hop 200 (N/2) and hop 100 (N/4) sum flat to "
            f"{max(ripple[100][1], ripple[200][1]):.1e}. The doc's own '25 ms / 10 ms' is the bad pair",
        ),
        practice.Check(
            "FINDING: this is not the spectrogram of Lesson 02",
            (two.get("frame_len"), two.get("hop")) == (256, 128) and result["denominator"] == "N - 1",
            f"Lesson 02's `main` ships frame_len={two.get('frame_len')}, hop={two.get('hop')} at "
            f"sr={two.get('sr')} -- a power-of-two frame at an N/2 hop -- and its `hann` divides by "
            f"`{result['denominator']}` where this one divides by `N`. Every parameter differs",
        ),
        practice.Check(
            "CONTROL: 400 is not a power of two, so the doc's own FFT rule excludes it",
            FRAME & (FRAME - 1) != 0,
            f"the doc defines the FFT as 'an O(N log N) algorithm requiring N = power of 2' and "
            f"then sets the frame to {FRAME} = 2^4 * 5^2, so the O(N^2) DFT is the only transform "
            "the lesson offers at this size. Real libraries factor 400 happily; this one cannot",
        ),
        practice.Check(
            "CONTROL: `imshow` is unbuildable here, so the plot is a text raster",
            "matplotlib" in result["plotting"] and len(result["raster"].splitlines()) == 20,
            f"find_spec is None for {result['plotting']}; the same matrix is rendered as "
            f"{len(result['raster'].splitlines())} rows of dB shading, low frequency at the "
            "bottom, and the diagonal it draws is the sweep. Printed in the lesson README",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
