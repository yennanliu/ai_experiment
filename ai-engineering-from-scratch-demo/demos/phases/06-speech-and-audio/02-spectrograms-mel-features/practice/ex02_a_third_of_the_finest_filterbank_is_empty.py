"""Exercise 2 — a third of the finest filterbank is empty.

    **Medium.** Re-run with `n_mels` in `{40, 80, 128}` and `frame_len` in
    `{200, 400, 800}`. Measure sharp-peak bandwidth across the time axis. Which
    combo resolves the chirp the best?

Reading of the exercise: "sharp-peak bandwidth across the time axis" is read as
the width of the energy ridge in each frame, at half the frame's peak, averaged
over frames -- and measured **twice**, once in mel bins and once in Hz, because
the exercise does not say which and the two disagree. `hop` stays at `main()`'s
128 and `n_fft` stays tied to `frame_len`, as `main()` ties it.

**40 mels at frame_len 200 resolves it best**: 2.17 mel bins, 116 Hz. Frame
length dominates n_mels by a wide margin -- 200 -> 800 costs **6.4x** in Hz
(116 -> 739), while 40 -> 128 mels costs at most 1.6x.

| mels \\ frame | 200 | 400 | 800 |
|---|---|---|---|
| 40 | **116 Hz** | 336 Hz | 739 Hz |
| 80 | 178 Hz | 385 Hz | 791 Hz |
| 128 | 181 Hz | 381 Hz | 804 Hz |

Three things the grid says that the doc does not.

**The doc's resolution trade runs backwards here.** "Larger FFT = better
frequency resolution" holds for a stationary tone. This chirp sweeps at
19,000 Hz/s, so an 800-sample frame spans **1900 Hz of sweep** against its own
**10 Hz** bin spacing: the smear is 190 bins wide and the FFT's resolution never
enters. Time-bandwidth sets the ridge whenever the signal moves.

**The two units rank the grid differently.** 40/400 beats 128/200 in mel bins
(3.82 against 5.00) and loses to it in Hz (336 against 181), because a mel bin is
not a fixed width -- it is `range / (n_mels + 1)`. "Bandwidth" measured in bins
is partly a measurement of `n_mels`.

**128 mels over a 200-sample frame leaves 40 of its 128 filters empty**, 31% of
the bank; 80/200 loses 12 and 128/400 loses 12. `mel_filterbank` rounds triangle
edges onto `n_fft//2+1 = 101` FFT bins, adjacent edges collide, and the row stays
all zero. `apply_filterbank` then returns 0.0, `log_transform` maps it to
`log(1e-10) = -23.026`, and Step 6's DCT runs over a bank a third of which is a
constant floor. Nothing warns.

Structure: `ridge` measures one representation's half-max width in bins and in
Hz; `mel_centres` gives the centre frequency of each mel filter; `grid` sweeps
the nine combos, caching one STFT per frame length; `dead_filters` counts the
all-zero rows.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "02-spectrograms-mel-features"
SR, HOP = 8000, 128
F_LO, F_HI, SECONDS = 200.0, 4000.0, 0.4
MELS, FRAMES = (40, 80, 128), (200, 400, 800)


def mel_centres(ref, n_mels):
    """Centre frequency of each triangular filter, in Hz."""
    lo, hi = ref.hz_to_mel(0.0), ref.hz_to_mel(SR / 2)
    return [ref.mel_to_hz(lo + (hi - lo) * i / (n_mels + 1)) for i in range(1, n_mels + 1)]


def dead_filters(bank):
    """Rows of the filterbank that are entirely zero, and so measure nothing."""
    return sum(1 for row in bank if not any(row))


def ridge(mel_spec, centres):
    """Half-max ridge width per frame, averaged: (mel bins, Hz)."""
    widths, spans = [], []
    for frame in mel_spec:
        peak = max(frame)
        if peak <= 0:
            continue
        lit = [i for i, v in enumerate(frame) if v >= peak / 2]
        widths.append(len(lit))
        spans.append(centres[lit[-1]] - centres[lit[0]])
    return sum(widths) / len(widths), sum(spans) / len(spans)


def grid(ref, signal):
    """The nine combos, one STFT per frame length rather than one per cell."""
    spectra = {length: ref.stft_magnitude(signal, length, HOP) for length in FRAMES}
    out = {}
    for n_mels in MELS:
        centres = mel_centres(ref, n_mels)
        for length in FRAMES:
            bank = ref.mel_filterbank(n_mels, length, SR)
            bins, hz = ridge(ref.apply_filterbank(spectra[length], bank), centres)
            out[(n_mels, length)] = {"dead": dead_filters(bank), "bins": bins, "hz": hz,
                                     "frames": len(spectra[length]), "fft": length // 2 + 1}
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cells = grid(ref, ref.chirp(F_LO, F_HI, SR, SECONDS))
    rate = 2 * (F_HI - F_LO) / SECONDS
    return {
        "cells": cells,
        "best_hz": min(cells, key=lambda k: cells[k]["hz"]),
        "best_bins": min(cells, key=lambda k: cells[k]["bins"]),
        "rate": rate, "smear": rate * max(FRAMES) / SR, "resolution": SR / max(FRAMES),
        "floor": ref.log_transform([[0.0]])[0][0],
        "frames": sorted({c["frames"] for c in cells.values()}),
    }


def verify(result):
    cells = result["cells"]
    short, long_ = cells[(40, 200)], cells[(40, 800)]
    fine, coarse = cells[(128, 200)], cells[(40, 400)]
    worst = max(cells.values(), key=lambda c: c["dead"])
    return [
        practice.Check(
            "ANSWER: 40 mels at frame_len 200 -- 2.17 mel bins, 116 Hz",
            result["best_hz"] == result["best_bins"] == (40, 200),
            f"it is narrowest on both readings ({short['bins']:.2f} bins, {short['hz']:.1f} Hz) "
            f"and frame length dominates n_mels: 200 -> 800 costs "
            f"{long_['hz'] / short['hz']:.1f}x in Hz where 40 -> 128 mels costs "
            f"{fine['hz'] / short['hz']:.1f}x",
        ),
        practice.Check(
            "FINDING: the doc's 'larger FFT = better resolution' runs backwards here",
            long_["hz"] > 6 * short["hz"],
            f"the chirp sweeps at {result['rate']:.0f} Hz/s, so an {max(FRAMES)}-sample frame "
            f"spans {result['smear']:.0f} Hz of sweep against its own "
            f"{result['resolution']:.0f} Hz bin spacing -- a {result['smear'] / result['resolution']:.0f}-bin "
            f"smear. Its ridge is {long_['hz']:.0f} Hz wide against {short['hz']:.0f}",
        ),
        practice.Check(
            "FINDING: mel bins and Hz rank the grid in opposite orders",
            coarse["bins"] < fine["bins"] and coarse["hz"] > fine["hz"],
            f"40/400 beats 128/200 in bins ({coarse['bins']:.2f} against {fine['bins']:.2f}) and "
            f"loses to it in Hz ({coarse['hz']:.0f} against {fine['hz']:.0f}), because a mel bin "
            "is range/(n_mels+1) and not a fixed width. The exercise does not say which unit",
        ),
        practice.Check(
            "FINDING: 128 mels over a 200-sample frame leaves 40 of 128 filters empty",
            fine["dead"] == 40 and worst["dead"] == 40,
            f"`mel_filterbank` rounds triangle edges onto {fine['fft']} FFT bins, adjacent edges "
            f"collide and the row stays all zero: {fine['dead']} of 128 dead at 128/200, "
            f"{cells[(80, 200)]['dead']} at 80/200, {cells[(128, 400)]['dead']} at 128/400. "
            f"`log_transform` turns each into a constant {result['floor']:.3f} and Step 6 DCTs it",
        ),
        practice.Check(
            "CONTROL: the three frame lengths do not share a time axis",
            len(result["frames"]) == len(FRAMES),
            f"at a fixed hop of {HOP} the frame counts are {result['frames']} for frame lengths "
            f"{list(FRAMES)}, so 'across the time axis' averages over a different number of "
            "frames in each column. The nine cells are not nine measurements of one thing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
