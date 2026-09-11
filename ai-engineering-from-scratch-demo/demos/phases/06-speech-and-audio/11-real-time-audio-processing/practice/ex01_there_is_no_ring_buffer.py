"""Exercise 1 — there is no ring buffer, and the stream is 1.9 seconds.

    **Easy.** Run `code/main.py`. Simulates a ring buffer + energy VAD; prints
    stage latencies for a fake 10-second stream.

Reading of the exercise: one sentence makes three claims about the file -- a ring
buffer, an energy VAD, and a 10-second stream -- so the exercise is read as "run
it and check the three". The VAD is there. The other two are not.

**The stream is 95 chunks of 20 ms: 1900 ms.** The exercise names 10 s, which
would be 500 chunks, so the demo is **5.3x short**; Step 1's own caption says
"1.5 s of user speech" while generating 1.9 s of stream, and the VAD gate keeps
**24000 samples, exactly 1.500 s** of it.

**Nothing in the module is a ring buffer.** The substring `ring` does not occur
anywhere in `code/main.py`; `buffered` is a plain list that `extend`s without a
bound. The doc's own Step 1 defines a `RingBuffer` and sizes it at "32,000
samples at 16 kHz = 2 s", and no line imports or reimplements it. At the 1.5 s
this demo keeps, an unbounded list and a 2 s ring behave identically -- which is
why the omission is invisible here. At the **10 s stream the exercise names**, a
speech-only buffer is 160,000 samples, **5.0x that capacity**, and a ring would
have been dropping samples for eight of those ten seconds.

**The end-to-end number omits three of the six stages in the table four lines
below it.** `main()` times STT, LLM and TTS TTFA. Step 4 also lists `network in`
(50-100 ms), `VAD` (20-80 ms) and `network out` (50-100 ms). Adding the table's
own **minima** to the measured 405 ms nominal gives **525 ms** -- already past
the `target: < 500 ms` printed on the same line -- and its maxima give 685 ms.

**And the table's TOTAL row is not the sum of its own column.** The six stages
sum to **420-1380 ms**; the row printed underneath them says **400-1400**.

Structure: `stream` rebuilds Step 1's chunk list; `gate` replays Step 2's VAD
loop; `budget` times the three simulated stages against their closed-form sleeps;
`TABLE` is Step 4's own figures, transcribed.
"""

from __future__ import annotations

import inspect
import random
import time

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "11-real-time-audio-processing"
SR, SPEECH_CHUNKS, SILENCE_CHUNKS = 16000, 75, 20
STATED_SECONDS, RING_SAMPLES, TARGET_MS = 10.0, 32000, 500.0
MEASURED = ("STT stream", "LLM stream", "TTS TTFA")
TABLE = {"network in": (50, 100), "VAD": (20, 80), "STT stream": (100, 300),
         "LLM stream": (100, 500), "TTS TTFA": (100, 300), "network out": (50, 100)}
PRINTED_TOTAL = (400, 1400)


def stream(ref, seed=0):
    """Step 1's chunk list: 75 speech chunks then 20 of near-silence."""
    rng = random.Random(seed)
    speech = [ref.simulate_chunk(True, rng) for _ in range(SPEECH_CHUNKS)]
    return speech + [ref.simulate_chunk(False, rng) for _ in range(SILENCE_CHUNKS)]


def gate(ref, chunks):
    """Step 2 verbatim: extend on speech, break on the first silence past 0.3 s."""
    buffered, in_speech = [], False
    for chunk in chunks:
        if ref.vad(chunk):
            buffered.extend(chunk)
            in_speech = True
        elif in_speech and len(buffered) >= SR * 0.3:
            break
    return buffered


def budget(ref, seconds):
    """(measured ms, closed-form ms) for the three stages `main()` times."""
    nominal = {"STT stream": 80 + seconds * 50, "LLM stream": 150.0, "TTS TTFA": 100.0}
    calls = ((ref.fake_stt, seconds), (ref.fake_llm, "x"), (ref.fake_tts_first_audio, "y"))
    measured = {}
    for name, (fn, arg) in zip(MEASURED, calls):
        start = time.perf_counter()
        fn(arg)
        measured[name] = (time.perf_counter() - start) * 1000
    return measured, nominal


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chunks = stream(ref)
    buffered = gate(ref, chunks)
    measured, nominal = budget(ref, len(buffered) / SR)
    omitted = [name for name in TABLE if name not in MEASURED]
    source = inspect.getsource(ref)
    return {
        "chunks": len(chunks), "stream_ms": len(chunks) * ref.CHUNK_MS,
        "stated_chunks": STATED_SECONDS * 1000 / ref.CHUNK_MS,
        "buffered": len(buffered), "ring_ratio": STATED_SECONDS * SR / RING_SAMPLES,
        "has_ring": "ring" in source.lower(),
        "dead_seed": "random.seed(0)" in source and "random.gauss" not in source,
        "nominal_total": sum(nominal.values()),
        "measured_total": sum(measured.values()),
        "omitted": sorted(omitted),
        "omitted_low": sum(TABLE[name][0] for name in omitted),
        "omitted_high": sum(TABLE[name][1] for name in omitted),
        "column": (sum(low for low, _ in TABLE.values()),
                   sum(high for _, high in TABLE.values())),
    }


def verify(result):
    ratio = result["measured_total"] / result["nominal_total"]
    floor = result["nominal_total"] + result["omitted_low"]
    return [
        practice.Check(
            "ANSWER: the stream is 1900 ms, not the 10 s the exercise names",
            result["chunks"] == 95 and result["stream_ms"] == 1900,
            f"{result['chunks']} chunks of 20 ms is {result['stream_ms']} ms against the "
            f"{result['stated_chunks']:.0f} chunks 10 s would need -- "
            f"{STATED_SECONDS * 1000 / result['stream_ms']:.1f}x short. Step 1's caption says "
            f"'1.5 s of user speech' over a 1.9 s stream, and the gate keeps "
            f"{result['buffered']} samples, exactly {result['buffered'] / SR:.3f} s",
        ),
        practice.Check(
            "FINDING: nothing in the module is a ring buffer",
            not result["has_ring"],
            f"the substring 'ring' does not occur anywhere in `code/main.py`; `buffered` is a "
            f"list that `extend`s without a bound. The doc's Step 1 defines a `RingBuffer` at "
            f"{RING_SAMPLES} samples = 2 s and nothing imports it. At the 10 s stream the "
            f"exercise names a speech-only buffer is {STATED_SECONDS * SR:.0f} samples -- "
            f"{result['ring_ratio']:.1f}x that capacity",
        ),
        practice.Check(
            "FINDING: three of the six stages are never timed, and they cost the target",
            floor > TARGET_MS,
            f"`main()` times {list(MEASURED)} and Step 4 also lists {result['omitted']}. Adding "
            f"the table's own minima to the {result['nominal_total']:.0f} ms measured gives "
            f"{floor:.0f} ms, past the 'target: < {TARGET_MS:.0f} ms' printed on the same line; "
            f"its maxima give {result['nominal_total'] + result['omitted_high']:.0f} ms",
        ),
        practice.Check(
            "FINDING: the TOTAL row is not the sum of its own column",
            result["column"] != PRINTED_TOTAL,
            f"the six stages sum to {result['column'][0]}-{result['column'][1]} ms and the row "
            f"printed underneath them says {PRINTED_TOTAL[0]}-{PRINTED_TOTAL[1]}. Both ends are "
            "off by 20 ms, in opposite directions",
        ),
        practice.Check(
            "CONTROL: the latency measured is the sleep, and the demo's seed is dead",
            1.0 <= ratio < 1.3 and result["dead_seed"],
            f"measured total is {ratio:.3f}x the closed-form "
            f"{result['nominal_total']:.0f} ms, so the pipeline is timing `time.sleep` and "
            "nothing else. `main()` also calls `random.seed(0)` and then draws only from its "
            "own `random.Random(0)`, so the module-level seed changes no output",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
