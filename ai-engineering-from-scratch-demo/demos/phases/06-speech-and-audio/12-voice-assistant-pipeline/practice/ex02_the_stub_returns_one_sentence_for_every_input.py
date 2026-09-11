"""Exercise 2 — the stub returns one sentence for every input, including silence.

    **Medium.** Replace the STT stub with a real Whisper model on a pre-recorded
    `.wav`. Measure WER and end-to-end latency.

Reading of the exercise: `whisper`, `torch`, `transformers`, `soundfile`,
`sounddevice`, `silero_vad` and `openai` are all absent, and a scan of the whole
reference checkout finds **0** files ending `.wav`, `.flac`, `.mp3`, `.ogg` or
`.m4a`, so there is neither a model to install nor a recording to run it on. What
survives is the pair of measurements, and both are decided before any model is
chosen.

**WER is 0.000 for every input, because `streaming_stt` never reads its
argument.** It sleeps for a duration derived from `len(utterance)` and then
returns the literal `'set a timer for five minutes'`. Four probes -- the captured
turn, pure silence, white noise, and an empty buffer -- return **1** distinct
transcript, and scored against that sentence with Lesson 04's `wer` each is
**0.000**. Swapping in a real Whisper can only make this number worse; the
exercise asks for a measurement whose shipped value is the floor.

**And silence produces a full sentence.** The doc's own third failure mode is
"Silence hallucination. Whisper outputs 'Thanks for watching' on the silent
warm-up frames. Always VAD-gate." The stub does exactly that, on an all-zero
buffer, unconditionally -- the one behaviour the lesson warns about is the only
behaviour it has.

**`streaming_stt` does not stream.** `main()` calls it once, after the capture
loop has returned, on the whole buffer, and gets back one final string -- no
partials, which is what the doc's Step 3 says streaming STT is for. Its cost is
`80 ms + duration * 0.05`, and all of it is serial with capture:

| arrangement | VAD | STT | LLM | TTS | total |
|---|---:|---:|---:|---:|---:|
| as shipped, one call after capture | 400.0 | **166.0** | 120.0 | 100.0 | **786.0** |
| streaming, only the tail left to pay | 400.0 | **81.0** | 120.0 | 100.0 | **701.0** |

**85.0 ms, 10.8% of the budget**, is the cost of the architecture the function is
named after and does not have.

**The fixed 80 ms dominates only at this length.** It is **48.2%** of the STT row
for the demo's 1.72 s turn, 13.8% at 10 s and 5.1% at 30 s -- so the demo's STT
figure is mostly its own constant, and it stops being so as soon as the utterance
is realistic.

Structure: `audio_files` scans the reference checkout; `probes` builds four very
different buffers; `stt_ms` is the stub's own latency model; `budget` sums the
four Step 5 rows for a given STT cost.
"""

from __future__ import annotations

import importlib.util
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "12-voice-assistant-pipeline"
METRIC_LESSON = "04-speech-recognition-asr"
SR, CHUNK_MS, CAPTURED_MS = 16000, 20, 1720.0
STT_FIXED_MS, STT_RTF, HANG_MS, LLM_MS, TTS_MS = 80.0, 0.05, 400.0, 120.0, 100.0
LENGTHS = (1.72, 10.0, 30.0)
SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
ABSENT = ("whisper", "torch", "transformers", "soundfile", "sounddevice", "silero_vad", "openai")


def audio_files():
    """Every recording anywhere in the reference checkout -- the exercise needs one."""
    root = parity.find_reference_root()
    return [p.name for p in root.rglob("*") if p.suffix.lower() in SUFFIXES]


def probes():
    """Four buffers with nothing in common but their type."""
    rnd = random.Random(1)
    return {"the captured turn": [0.2] * int(CAPTURED_MS / 1000 * SR),
            "pure silence": [0.0] * SR,
            "white noise": [rnd.gauss(0, 1) for _ in range(SR)],
            "an empty buffer": []}


def stt_ms(duration_ms):
    """The stub's own latency model: a fixed cost plus a real-time factor."""
    return STT_FIXED_MS + duration_ms * STT_RTF


def budget(stt):
    return HANG_MS + stt + LLM_MS + TTS_MS


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scorer = parity.load_reference(PHASE, METRIC_LESSON, "main").wer
    heard = {name: ref.streaming_stt(buffer) for name, buffer in probes().items()}
    spoken = next(iter(heard.values()))
    serial, streamed = stt_ms(CAPTURED_MS), stt_ms(CHUNK_MS)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "recordings": audio_files(),
        "heard": heard, "distinct": len(set(heard.values())),
        "wer": {name: scorer(spoken, text) for name, text in heard.items()},
        "silence_words": len(heard["pure silence"].split()),
        "serial": serial, "streamed": streamed,
        "total": budget(serial), "overlapped": budget(streamed),
        "fixed_share": {secs: STT_FIXED_MS / stt_ms(secs * 1000) for secs in LENGTHS},
    }


def verify(result):
    saving = result["total"] - result["overlapped"]
    share = result["fixed_share"]
    spoken = next(iter(result["heard"].values()))
    return [
        practice.Check(
            "CONTROL: there is no model to install and no recording to run it on",
            len(result["absent"]) == len(ABSENT) and not result["recordings"],
            f"find_spec is None for {result['absent']}, and scanning the whole reference "
            f"checkout for {list(SUFFIXES)} finds {len(result['recordings'])} files. Both "
            "measurements the exercise asks for are decided before a model is chosen",
        ),
        practice.Check(
            "ANSWER: WER is 0.000 for every input, because the stub never reads its argument",
            result["distinct"] == 1 and max(result["wer"].values()) == 0.0,
            f"the captured turn, pure silence, white noise and an empty buffer return "
            f"{result['distinct']} distinct transcript, {spoken!r}, and Lesson 04's `wer` scores "
            f"each at {max(result['wer'].values()):.3f}. `streaming_stt` uses `len(utterance)` "
            "only to pick a sleep",
        ),
        practice.Check(
            "FINDING: silence produces a full sentence -- the doc's own third failure mode",
            result["silence_words"] > 0,
            f"an all-zero buffer transcribes as {result['heard']['pure silence']!r}, "
            f"{result['silence_words']} words. The doc lists 'Silence hallucination: Whisper "
            "outputs \"Thanks for watching\" on the silent warm-up frames. Always VAD-gate' as a "
            "failure mode, and the stub does it unconditionally",
        ),
        practice.Check(
            "FINDING: the STT cost is fully serial with capture, and need not be",
            saving > 0.1 * result["total"],
            f"`streaming_stt` is called once, after the capture loop returns, on the whole "
            f"buffer, and returns one final string -- no partials. That costs "
            f"{result['serial']:.1f} ms where a streaming reader would owe only the tail, "
            f"{result['streamed']:.1f} ms: the budget falls from {result['total']:.1f} to "
            f"{result['overlapped']:.1f} ms, a saving of {saving:.1f} ms "
            f"({saving / result['total']:.1%})",
        ),
        practice.Check(
            "MECHANISM: the fixed 80 ms dominates only at this utterance length",
            share[LENGTHS[0]] > 0.4 > share[LENGTHS[-1]],
            f"the constant is {share[LENGTHS[0]]:.1%} of the STT row for the demo's "
            f"{LENGTHS[0]} s turn, {share[LENGTHS[1]]:.1%} at {LENGTHS[1]:.0f} s and "
            f"{share[LENGTHS[-1]]:.1%} at {LENGTHS[-1]:.0f} s. The figure the demo prints is "
            "mostly its own constant, and stops being so as soon as the turn is realistic",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
