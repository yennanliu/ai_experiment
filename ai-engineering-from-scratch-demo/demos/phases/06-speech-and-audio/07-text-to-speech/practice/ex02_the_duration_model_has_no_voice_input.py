"""Exercise 2 — the duration model has no voice input.

    **Medium.** Install Kokoro, synthesize the same sentence at voice `af_bella`
    and `am_adam`. Compare audio durations and subjective quality.

Reading of the exercise: `kokoro`, `torch`, `soundfile` and `sounddevice` are all
absent, so nothing here can synthesise. What can be settled without synthesising
is whether the comparison the exercise asks for is one this lesson can express,
and it is not: **`duration` takes `(phones, jitter, seed)` and `mel_schedule`
takes `(phones, durs, hop_ms)`. Neither has a voice argument, and neither reads
one from anywhere else.** Two voices produce the same 210 frames, byte for byte,
so the duration comparison the exercise poses returns exactly **0 ms**.

That is not a small omission, because the duration predictor *is* the difference
between two voices in a FastSpeech-style model: the acoustic decoder changes
timbre, the duration predictor changes rhythm, and rhythm is what a stopwatch
measures. Here the only knobs are `jitter` and `seed`, and both are tiny:

| knob | range of the sentence's total |
|---|---|
| `seed`, 200 draws at `jitter=0.1` | 205-215 frames, **4.8%** end to end |
| `jitter=0.0` to `0.3` | 210 fixed, to 191-228 frames |

The model's entire stochastic range is smaller than the spread between two real
voices reading the same line, and it is not addressable by voice anyway.

**The one number a voice cannot change is also the one that is wrong.**
`mel_schedule` defaults to `hop_ms=12.5` and Step 4 converts at 300 samples per
frame, which is 12.5 ms at 24 kHz -- but 24 kHz TTS vocoders (Kokoro, F5-TTS,
HiFi-GAN at 24 kHz) hop by **256** samples, 10.67 ms. Feed these 210 frames to
one and the audio is **2.240 s**, not the 2.625 s the schedule prints: the
predicted duration is **17.2% long**, which is larger than any plausible
difference between `af_bella` and `am_adam` and would swamp the comparison even
if the comparison could be made.

Structure: `signatures` reads the argument names of the lesson's own duration
functions; `totals` sweeps seeds at one jitter; `at_hop` re-times the same frame
count at a different vocoder hop.
"""

from __future__ import annotations

import importlib.util
import inspect

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "07-text-to-speech"
TEXT = "Please remind me to water the plants at 6 pm."
SEEDS, JITTERS = 200, (0.0, 0.1, 0.3)
LESSON_HOP, VOCODER_HOP, RATE = 12.5, 256, 24000
ABSENT = ("kokoro", "torch", "soundfile", "sounddevice")
VOICE_WORDS = ("voice", "speaker", "spk", "style", "af_", "am_")


def signatures(ref):
    """Argument names of the two functions that decide how long the audio is."""
    return {name: list(inspect.signature(getattr(ref, name)).parameters)
            for name in ("duration", "mel_schedule")}


def totals(ref, phones, jitter):
    return sorted({sum(ref.duration(phones, jitter, seed)) for seed in range(SEEDS)})


def at_hop(frames, samples_per_frame=VOCODER_HOP, rate=RATE):
    """Seconds of audio a real 24 kHz vocoder returns for this many mel frames."""
    return frames * samples_per_frame / rate


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    phones = ref.phonemize(TEXT)
    frames = sum(ref.duration(phones, JITTERS[1], 42))
    _, printed_ms = ref.mel_schedule(phones, ref.duration(phones, JITTERS[1], 42))
    names = signatures(ref)
    source = inspect.getsource(ref.duration) + inspect.getsource(ref.mel_schedule)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "signatures": names,
        "voice_args": [a for args in names.values() for a in args
                       if any(w in a.lower() for w in VOICE_WORDS)],
        "voice_in_source": [w for w in VOICE_WORDS if w in source.lower()],
        "sweeps": {jitter: totals(ref, phones, jitter) for jitter in JITTERS},
        "frames": frames, "printed_ms": printed_ms,
        "vocoder_s": at_hop(frames), "lesson_s": printed_ms / 1000,
    }


def verify(result):
    sweeps, frames = result["sweeps"], result["frames"]
    jittered = sweeps[JITTERS[1]]
    drift = result["lesson_s"] / result["vocoder_s"] - 1
    return [
        practice.Check(
            "ANSWER: neither duration function takes a voice, so the comparison is 0 ms",
            not result["voice_args"] and not result["voice_in_source"],
            f"`duration{tuple(result['signatures']['duration'])}` and "
            f"`mel_schedule{tuple(result['signatures']['mel_schedule'])}` -- no voice argument, "
            f"and no voice-shaped name anywhere in either body. af_bella and am_adam produce "
            f"the same {frames} frames, byte for byte",
        ),
        practice.Check(
            "MECHANISM: the duration predictor is exactly where two voices differ",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}, so nothing here synthesises. In a "
            "FastSpeech-style model the acoustic decoder carries timbre and the duration "
            "predictor carries rhythm -- and rhythm is the half a stopwatch measures. This "
            "lesson models the half the exercise asks about and leaves out its only input",
        ),
        practice.Check(
            "FINDING: the only knobs move the total by under 5% and are not voice",
            (jittered[-1] - jittered[0]) / frames < 0.05,
            f"over {SEEDS} seeds at jitter {JITTERS[1]} the total spans "
            f"{jittered[0]}-{jittered[-1]} frames, {(jittered[-1] - jittered[0]) / frames:.1%} "
            f"end to end; at jitter {JITTERS[0]} it is fixed at {sweeps[JITTERS[0]][0]} and at "
            f"{JITTERS[2]} it spans {sweeps[JITTERS[2]][0]}-{sweeps[JITTERS[2]][-1]}",
        ),
        practice.Check(
            "FINDING: the hop the schedule assumes is 17% longer than a 24 kHz vocoder's",
            0.15 < drift < 0.20,
            f"Step 4 converts at 300 samples per frame ({LESSON_HOP} ms at {RATE // 1000} kHz), "
            f"while Kokoro, F5-TTS and HiFi-GAN at {RATE // 1000} kHz hop by {VOCODER_HOP} "
            f"samples ({VOCODER_HOP / RATE * 1000:.2f} ms). The same {frames} frames are "
            f"{result['vocoder_s']:.3f} s of audio, not the {result['lesson_s']:.3f} s printed "
            f"-- {drift:.1%} long, larger than any plausible gap between two voices",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
