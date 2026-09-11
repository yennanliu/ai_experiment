"""Exercise 2 — the gate is a statement about microphone gain.

    **Medium.** Using `sounddevice`, build a passthrough loop that processes your
    mic in 20 ms frames and prints VAD state at each frame.

Reading of the exercise: `sounddevice`, `pyaudio` and `soundfile` are all absent
and there is no microphone, so the mic is synthesised -- four seconds of
speech-like bursts of 0.12-0.35 s separated by the 0.03-0.22 s gaps ordinary
speech has, over a noise floor at a chosen SNR. Everything else is the lesson's
own `rms_dbfs` and `vad`, called once per 20 ms frame exactly as a passthrough
callback would.

**"Prints VAD state at each frame" prints a flicker.** Over 200 frames the gate
changes state **24 times, 6.0 per second**, against **25** in the reference
labelling -- it tracks the truth to within one transition. The display is
accurate and unusable,
and the standard fix costs the budget: a **100 ms** hangover takes 24 flips to 4,
a **200 ms** hangover takes it to **0**, and 200 ms is **40%** of the
`target: < 500 ms` the same lesson prints one file over.

**The threshold is absolute, so it describes the microphone and not the speech.**
The gate fires on `rms_dbfs > -40`, which is an RMS of exactly **0.01**. With
speech at amplitude 0.15 the noise floor reaches that level at
`20*log10(0.15/0.01)` = **23.52 dB SNR**, and below it every silent frame reads
as speech:

| SNR | false accepts on silent frames |
|---:|---:|
| 40 dB | 0.2623 |
| 25 dB | 0.2623 |
| **20 dB** | **1.0000** |
| 10 dB | 1.0000 |

**And a frame is decided by one and a half of its samples.** Crossing the
threshold needs only `320 * (0.01/0.15)^2` = **1.42** loud samples out of 320:
measured, **1** loud sample trips the gate on 22% of draws and **4** on 83%. That
is why the false-accept rate is 0.26 even at 40 dB -- every frame straddling the
edge of a syllable is 99% silence and reads as speech.

Structure: `layout` places the voiced and unvoiced spans; `render` draws one
signal for that layout at an SNR; `framewise` runs the lesson's gate per 20 ms
frame; `flips` counts state changes; `hangover` holds the gate open;
`false_accept` scores the silent frames; `trip_rate` measures how few loud
samples decide a frame.
"""

from __future__ import annotations

import importlib.util
import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "11-real-time-audio-processing"
SR, SPEECH_AMP, SECONDS = 16000, 0.15, 4.0
SNRS, HOLDS, LOUD = (40, 25, 20, 10), (0, 100, 200), (1, 4)
TARGET_MS = 500.0
ABSENT = ("sounddevice", "pyaudio", "soundfile", "torch")


def layout(rng, seconds=SECONDS):
    """Per-sample truth: voiced bursts of 0.12-0.35 s with the gaps speech has."""
    truth, total = [], int(SR * seconds)
    while len(truth) < total:
        truth += [True] * int(SR * rng.uniform(0.12, 0.35))
        truth += [False] * int(SR * rng.uniform(0.03, 0.22))
    return truth[:total]


def render(rng, truth, snr_db):
    """One signal for that layout, over a noise floor at `snr_db`."""
    floor = SPEECH_AMP / 10 ** (snr_db / 20)
    return [(SPEECH_AMP * rng.gauss(0, 1) if speech else 0.0) + floor * rng.gauss(0, 1)
            for speech in truth]


def framewise(ref, signal, truth):
    """(gate decisions, majority-vote truth) for every 20 ms frame."""
    width = int(SR * ref.CHUNK_MS / 1000)
    starts = range(0, len(signal) - width + 1, width)
    return ([ref.vad(signal[i:i + width]) for i in starts],
            [sum(truth[i:i + width]) / width > 0.5 for i in starts])


def flips(states):
    return sum(1 for first, second in zip(states, states[1:]) if first != second)


def hangover(states, hold_frames):
    """Hold the gate open for `hold_frames` after the last active frame."""
    out, hold = [], 0
    for state in states:
        out.append(state or hold > 0)
        hold = hold_frames if state else max(0, hold - 1)
    return out


def false_accept(decisions, gold):
    silent = [decision for decision, speech in zip(decisions, gold) if not speech]
    return sum(silent) / len(silent)


def trip_rate(ref, rng, count, trials=400):
    """Share of frames with `count` loud samples of 320 that trip the gate."""
    width = int(SR * ref.CHUNK_MS / 1000)
    padded = [0.0] * (width - count)
    return sum(ref.vad([SPEECH_AMP * rng.gauss(0, 1) for _ in range(count)] + padded)
               for _ in range(trials)) / trials


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    truth = layout(random.Random(7))
    runs = {}
    for snr in SNRS:
        decisions, gold = framewise(ref, render(random.Random(7), truth, snr), truth)
        runs[snr] = (decisions, gold)
    base, gold = runs[SNRS[0]]
    threshold_rms = 10 ** (ref.VAD_THRESHOLD_DBFS / 20)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "frames": len(base), "flips": flips(base), "gold_flips": flips(gold),
        "held": {ms: flips(hangover(base, ms // ref.CHUNK_MS)) for ms in HOLDS},
        "false_accept": {snr: false_accept(*runs[snr]) for snr in SNRS},
        "threshold": ref.VAD_THRESHOLD_DBFS, "rms": threshold_rms,
        "cliff": 20 * math.log10(SPEECH_AMP / threshold_rms),
        "needed": 320 * (threshold_rms / SPEECH_AMP) ** 2,
        "trip": {n: trip_rate(ref, random.Random(3), n) for n in LOUD},
    }


def verify(result):
    held, accepts = result["held"], result["false_accept"]
    return [
        practice.Check(
            "ANSWER: printing the state at every frame prints a flicker",
            abs(result["flips"] - result["gold_flips"]) <= 1 < result["flips"] - 20
            and len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}, so the mic is synthesised. Over "
            f"{result['frames']} frames the gate changes state {result['flips']} times, "
            f"{result['flips'] / SECONDS:.1f} per second, against {result['gold_flips']} in the "
            "reference labelling -- it tracks the truth to within one transition. The display "
            "is accurate and unusable",
        ),
        practice.Check(
            "MECHANISM: the standard fix costs 40% of the lesson's own budget",
            held[HOLDS[-1]] == 0 < held[HOLDS[0]],
            f"a {HOLDS[1]} ms hangover takes {held[HOLDS[0]]} flips to {held[HOLDS[1]]} and a "
            f"{HOLDS[-1]} ms hangover takes it to {held[HOLDS[-1]]}, but {HOLDS[-1]} ms is "
            f"{HOLDS[-1] / TARGET_MS:.0%} of the 'target: < {TARGET_MS:.0f} ms' the same lesson "
            "prints, and it is added to every end-of-turn decision",
        ),
        practice.Check(
            "FINDING: the threshold describes the microphone, not the speech",
            accepts[20] == 1.0 > accepts[25] and 23 < result["cliff"] < 24,
            f"the gate fires on rms_dbfs > {result['threshold']:.0f}, an RMS of "
            f"{result['rms']:.2f}, so a floor at amplitude {SPEECH_AMP}/10**(SNR/20) reaches it "
            f"at {result['cliff']:.2f} dB SNR. False accepts on silent frames: "
            f"{accepts[40]:.4f} at 40 dB, {accepts[25]:.4f} at 25 dB, {accepts[20]:.4f} at 20 dB",
        ),
        practice.Check(
            "MECHANISM: one and a half samples of 320 decide the frame",
            result["needed"] < 2 and 0.1 < result["trip"][1] < result["trip"][4],
            f"crossing {result['rms']:.2f} RMS needs 320*({result['rms']:.2f}/{SPEECH_AMP})^2 = "
            f"{result['needed']:.2f} loud samples: measured, {LOUD[0]} trips the gate on "
            f"{result['trip'][1]:.0%} of draws and {LOUD[1]} on {result['trip'][4]:.0%}. That is "
            f"why silent frames false-accept at {accepts[40]:.2f} even at 40 dB -- a frame "
            "straddling a syllable edge is 99% silence and reads as speech",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
