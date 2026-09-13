"""Exercise 3 — the prescribed overlap costs 25 points of WER before ASR runs.

    **Hard.** Implement streaming inference: chunk audio into 10 s windows with
    2 s overlap, run Whisper on each chunk, merge transcripts. Measure word-error
    rate vs single-pass on a 5-minute podcast sample.

Reading of the exercise: there is no Whisper, no audio and no network, so the
recogniser is replaced by a *perfect* one -- each chunk returns exactly the words
spoken inside it -- and the two things the exercise actually specifies are
measured: the chunk arithmetic, and the merge. A perfect recogniser is the right
control, because every error left is one the protocol created.

**ANSWER: naive concatenation gives 24.7% WER with a perfect recogniser.** A
5-minute file at 180 wpm is 900 words; 10-second windows at 2-second overlap make
**38 chunks** whose spans total **374 seconds** of a 300-second file, and
concatenating them emits **1,122 words**. The 222 extra are the overlap, and they
are insertions. A longest-suffix-prefix merge takes the same chunks to **0.0%**.

**FINDING: the chunk size is smaller than Whisper's window, so streaming costs
3.8x.** Whisper's encoder takes a fixed 30-second input and pads anything shorter
to it, so a 10-second chunk costs a full 30-second forward pass. Single-pass is
`ceil(300/30) = 10` windows = 300 s of encoder input; streaming is `38 * 30 =
1,140` s. The exercise's own chunk length turns a latency optimisation into
**3.8x the compute**.

**FINDING: the merge is where the difficulty is, and it is a string problem.**
With clean chunk boundaries the suffix-prefix merge is exact. Clip one word at
each chunk edge -- what a real recogniser does to a word split across a boundary
-- and naive concatenation still reads **16.7%** while the merge reads **0.2%**.
The 2 seconds of overlap the exercise prescribes exists to make that merge
possible; nothing in the exercise says to do it, and skipping it is the single
largest error source in the pipeline.

**CONTROL: nothing in the setup exists.** `torch`, `whisper`, `transformers` and
`soundfile` all return None from `find_spec`, and a 5-minute podcast sample would
have to be downloaded.

Structure: `chunks` is the window schedule; `wer` is Levenshtein over words;
`stitch` is the longest-suffix-prefix merge.
"""

from __future__ import annotations

import importlib.util
import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "10-audio-transformers-whisper"
MINUTES, WPM, WINDOW, OVERLAP, WHISPER = 5, 180, 10, 2, 30
ABSENT = ("torch", "whisper", "transformers", "soundfile")


def chunks(total, window=WINDOW, overlap=OVERLAP):
    """The exercise's schedule: `window`-second spans stepping by window - overlap."""
    spans, start = [], 0
    while start < total:
        spans.append((start, min(start + window, total)))
        if spans[-1][1] >= total:
            break
        start += window - overlap
    return spans


def wer(reference, hypothesis):
    """Word error rate: Levenshtein over words, divided by the reference length."""
    previous = list(range(len(hypothesis) + 1))
    for i, want in enumerate(reference, 1):
        row = [i]
        for j, got in enumerate(hypothesis, 1):
            row.append(min(previous[j] + 1, row[-1] + 1, previous[j - 1] + (want != got)))
        previous = row
    return previous[-1] / len(reference)


def stitch(parts):
    """Merge by the longest suffix of the output that prefixes the next chunk."""
    out = list(parts[0])
    for part in parts[1:]:
        best = next((k for k in range(min(len(out), len(part)), 0, -1)
                     if out[-k:] == part[:k]), 0)
        out += part[best:]
    return out


def transcribe(words, spans, seconds, clip=False):
    """A perfect recogniser: each chunk returns exactly the words spoken inside it."""
    rate = len(words) / seconds
    parts = [words[int(a * rate):int(b * rate)] for a, b in spans]
    return [p[1:-1] if clip and len(p) > 2 else p for p in parts]


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(0)
    words = [f"w{rng.randrange(400)}" for _ in range(MINUTES * WPM)]
    seconds = MINUTES * 60
    spans = chunks(seconds)
    clean, clipped = transcribe(words, spans, seconds), transcribe(words, spans, seconds, True)
    flat = lambda parts: [w for part in parts for w in part]
    return {
        "words": len(words), "seconds": seconds, "spans": len(spans),
        "covered": sum(b - a for a, b in spans),
        "single": math.ceil(seconds / WHISPER) * WHISPER, "streamed": len(spans) * WHISPER,
        "naive": wer(words, flat(clean)), "merged": wer(words, stitch(clean)),
        "emitted": len(flat(clean)),
        "naive_clipped": wer(words, flat(clipped)), "merged_clipped": wer(words, stitch(clipped)),
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: naive concatenation is 24.7% WER with a perfect recogniser",
            result["naive"] > 0.2 and result["merged"] == 0.0,
            f"{MINUTES} minutes at {WPM} wpm is {result['words']} words; {WINDOW}s windows at "
            f"{OVERLAP}s overlap make {result['spans']} chunks covering {result['covered']}s of a "
            f"{result['seconds']}s file, and concatenating them emits {result['emitted']} words. "
            f"The {result['emitted'] - result['words']} extra are the overlap, and they are "
            f"insertions: {result['naive']:.1%} WER. The suffix-prefix merge gives "
            f"{result['merged']:.1%}",
        ),
        practice.Check(
            "FINDING: the chunk is smaller than Whisper's window, so streaming costs 3.8x",
            result["streamed"] / result["single"] > 3.5,
            f"Whisper's encoder takes a fixed {WHISPER}s input and pads anything shorter, so a "
            f"{WINDOW}s chunk costs a full {WHISPER}s forward pass. Single-pass is "
            f"{result['single'] // WHISPER} windows = {result['single']}s of encoder input; "
            f"streaming is {result['spans']} * {WHISPER} = {result['streamed']}s, "
            f"{result['streamed'] / result['single']:.1f}x. The chunk length turns a latency "
            "optimisation into more compute",
        ),
        practice.Check(
            "FINDING: the merge is the difficulty, and it survives ragged edges",
            result["naive_clipped"] > 0.15 and result["merged_clipped"] < 0.01,
            f"clip one word at each chunk edge -- what a recogniser does to a word split across a "
            f"boundary -- and naive concatenation still reads {result['naive_clipped']:.1%} while "
            f"the merge reads {result['merged_clipped']:.1%}. The {OVERLAP}s of overlap exists to "
            "make that merge possible, and nothing in the exercise says to perform it",
        ),
        practice.Check(
            "CONTROL: a perfect recogniser is the right control",
            result["merged"] == 0.0,
            "each chunk returns exactly the words spoken inside it, so every error left is one "
            f"the protocol created rather than one the model made. That the merge reaches "
            f"{result['merged']:.1%} is what proves the {result['naive']:.1%} belongs to the "
            "concatenation and not to the chunking",
        ),
        practice.Check(
            "CONTROL: nothing in the setup exists",
            result["absent"] == list(ABSENT),
            f"find_spec is None for {result['absent']}, and a 5-minute podcast sample would have "
            "to be downloaded. What the exercise specifies beyond the model -- the window "
            "schedule, the overlap, the merge and the metric -- is all buildable, and all of it "
            "is where the answer turns out to live",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
