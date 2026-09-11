"""Exercise 2 — the schedule costs nineteen points of WER before the model runs.

    **Medium.** Install `faster-whisper`, transcribe a 10-minute podcast, compare
    WER against a human transcript. Try `language="auto"` vs forced
    `language="en"`.

Reading of the exercise: `faster_whisper`, `ctranslate2`, `torch`, `transformers`
and `soundfile` are all absent, there is no podcast anywhere in the reference
tree and no human transcript to compare against, so the transcription cannot run.
What can run is everything the transcription would be wrapped in, and that is
where the answer is: **the chunk schedule `code/main.py` ships puts a WER floor
of 0.191 on a 10-minute clip, with a perfect transcriber.**

The demonstration is a 600 s reference transcript at 150 words per minute,
1,500 words with timings, chunked on `chunk_schedule(600, 30, 5)`, each chunk
transcribed exactly right, and the chunks concatenated. WER is the lesson's own
`wer`, imported from Lesson 04 -- this lesson ships no metric of its own.

| merge | hypothesis words | WER |
|---|---:|---:|
| concatenate the 24 chunks | **1787** | **0.1913** |
| drop the first 5 s of every chunk after the first | 1500 | **0.0000** |

The cause is arithmetic, not ASR. 24 chunks spanning 30 s each, stepping by
`chunk_s - stride_s = 25`, decode **715 s of audio for a 600 s clip** -- 19% more
than exists -- and the extra **115 s** is speech transcribed twice. The overlap
is deliberate and right; what is missing is any way to undo it. `chunk_schedule`
returns `(start, end)` pairs and nothing else, so no caller can tell which part
of a chunk is the overlap without re-deriving `stride_s`, which is not in the
return value.

The other half of the exercise cannot be posed against this code at all.
`build_prompt` takes a language key and looks it up in a three-entry `LANG` dict,
so `build_prompt("auto")` raises `KeyError: 'auto'` -- the forced arm is the only
arm the lesson can express. The doc's own Pitfalls section already gives the
verdict the comparison is meant to reach: "Whisper's auto LID mis-routes noisy
clips to Japanese or Welsh; force `language='en'` when you know."

Structure: `transcript` lays 1,500 words on a 600 s timeline; `spoken` returns
the words inside a span; `naive` and `trimmed` are the two merges; `budget`
totals the audio each schedule decodes.
"""

from __future__ import annotations

import importlib.util
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "05-whisper-architecture-finetuning"
METRIC_LESSON = "04-speech-recognition-asr"
TOTAL, CHUNK, STRIDE, WPM = 600.0, 30.0, 5.0, 150
ABSENT = ("faster_whisper", "ctranslate2", "torch", "transformers", "soundfile")
LEXICON = [f"w{i:03d}" for i in range(400)]


def transcript(seconds=TOTAL, rate=WPM):
    """A reference transcript with one timestamp per word, evenly spaced."""
    rnd = random.Random(0)
    step = 60.0 / rate
    return [(i * step, rnd.choice(LEXICON)) for i in range(int(seconds * rate / 60))]


def spoken(words, start, end):
    return [word for when, word in words if start <= when < end]


def naive(words, schedule):
    """What concatenating `chunk_schedule`'s spans produces."""
    return " ".join(word for start, end in schedule for word in spoken(words, start, end))


def trimmed(words, schedule, stride=STRIDE):
    """The same, dropping each chunk's leading overlap -- the number the schedule omits."""
    out = []
    for index, (start, end) in enumerate(schedule):
        out += spoken(words, start if index == 0 else start + stride, end)
    return " ".join(out)


def budget(schedule):
    """Seconds of audio the schedule asks a model to decode."""
    return sum(end - start for start, end in schedule)


def auto_language(ref):
    """Whether the lesson's prompt builder can express `language='auto'` at all."""
    try:
        ref.build_prompt("auto")
    except KeyError as exc:
        return f"KeyError: {exc}"
    return "accepted"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    wer = parity.load_reference(PHASE, METRIC_LESSON, "main").wer
    words = transcript()
    schedule = ref.chunk_schedule(TOTAL, CHUNK, STRIDE)
    reference = " ".join(word for _, word in words)
    concatenated, deduplicated = naive(words, schedule), trimmed(words, schedule)
    decoded = budget(schedule)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "chunks": len(schedule), "decoded": decoded, "overlap": decoded - TOTAL,
        "reference_words": len(words), "naive_words": len(concatenated.split()),
        "naive_wer": wer(reference, concatenated), "trimmed_wer": wer(reference, deduplicated),
        "fields": len(schedule[0]), "auto": auto_language(ref),
        "languages": list(ref.LANG),
    }


def verify(result):
    return [
        practice.Check(
            "CONTROL: nothing here can transcribe, and nothing here is a podcast",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}, and the reference tree ships no audio "
            "file and no human transcript. What the transcription would be wrapped in is "
            "entirely present, so that is what gets measured",
        ),
        practice.Check(
            "ANSWER: a perfect transcriber scores WER 0.191 on this schedule",
            0.18 < result["naive_wer"] < 0.21 and result["trimmed_wer"] == 0.0,
            f"{result['reference_words']} reference words, chunked on chunk_schedule("
            f"{TOTAL:.0f}, {CHUNK:.0f}, {STRIDE:.0f}) and transcribed exactly right, concatenate "
            f"to {result['naive_words']} words and score {result['naive_wer']:.4f}. Dropping "
            f"each chunk's leading {STRIDE:.0f} s scores {result['trimmed_wer']:.4f}",
        ),
        practice.Check(
            "MECHANISM: the schedule decodes 715 s of audio for a 600 s clip",
            result["overlap"] > 0.15 * TOTAL,
            f"{result['chunks']} chunks of {CHUNK:.0f} s stepping by {CHUNK - STRIDE:.0f} s is "
            f"{result['decoded']:.0f} s, {result['decoded'] / TOTAL - 1:.0%} more than exists, "
            f"and the extra {result['overlap']:.0f} s is speech transcribed twice. The overlap "
            "is deliberate; the duplication is what nothing downstream removes",
        ),
        practice.Check(
            "MECHANISM: the schedule does not carry the number needed to undo itself",
            result["fields"] == 2,
            f"`chunk_schedule` returns {result['fields']}-tuples of (start, end) and nothing "
            f"else, so a caller cannot tell which part of a chunk is overlap without "
            f"re-deriving `stride_s`. The fix is one number the return value omits",
        ),
        practice.Check(
            "FINDING: the lesson cannot express the `language='auto'` arm at all",
            result["auto"].startswith("KeyError"),
            f"`build_prompt('auto')` raises {result['auto']} -- `LANG` holds "
            f"{result['languages']} and the builder indexes it directly, so the forced arm is "
            "the only one available. The doc's Pitfalls already state the verdict: auto LID "
            "'mis-routes noisy clips to Japanese or Welsh; force language=\"en\" when you know'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
