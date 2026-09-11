"""Exercise 1 — both tiers agree on every chunk.

    **Easy.** Run `code/main.py`. It simulates a speech + silence + speech +
    coughs sequence and tests three VAD tiers.

Reading of the exercise: the run prints two event lists under two headings that
claim a difference -- "energy-only VAD turn events (**many false positives on
cough**)" and "Silero-style VAD turn events (**rejects cough**)" -- so the check
is whether the two arms differ. They do not. The lists are identical, element for
element, and the two detectors agree on **151 of 151 chunks**:

    t= 440 ms START   t=1480 ms END   t=2040 ms START   t=2800 ms END

There are no false positives in the first list to be many of, and nothing in the
second list was rejected that the first accepted.

**The cough is not rejected by the Silero-style VAD.** `fake_silero_vad` scores
it **0.55**, above the 0.5 the caller compares against, so it is classified as
speech -- same as the energy gate, which reads it at **-9.1 dBFS**, well above
the -40 dBFS threshold. What stops a turn from starting is `TurnDetector`'s
`min_speech_ms = 250` against one 20 ms chunk, and **both arms run the same
state machine**. The guard the demo credits to the model belongs to the layer
underneath it.

**The confident branch is unreachable.** `fake_silero_vad` returns 0.92 only when
`rms > 0.08 and not transient`, and `transient` is
`max(chunk) - min(chunk) > 0.6 and duration < 0.03`. Every chunk is 320 samples
at 16 kHz, so `duration` is **0.0200 s** and the second half of that conjunction
is always true; and any chunk loud enough to pass `rms > 0.08` has a peak-to-peak
range above 0.6 -- the narrowest in the stream is speech at **0.954**, the
cough at **3.545**. So `not transient` is false
wherever the first test succeeds, and the whole stream takes exactly **two**
distinct probabilities, `{0.02, 0.55}`.

Two smaller things. The demo runs **two** detectors, not the three tiers the
exercise names -- the doc's third tier is a semantic turn detector and no such
name exists in the module. And the "coughs" are one chunk: `("cough", 1)`, 20 ms
of a 3020 ms stream.

Structure: `stream` rebuilds `main()`'s chunk sequence; `events` runs one
detector through a fresh `TurnDetector`; `probe` reports what each chunk kind
looks like to both tiers; `reads` asks whether a parameter is used in a body.
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "14-voice-activity-detection-turn-taking"
SEQUENCE = (("silence", 10), ("speech", 40), ("silence", 30), ("cough", 1),
            ("silence", 10), ("speech", 25), ("silence", 35))
CHUNK_MS, SR, SEED, GATE = 20, 16000, 42, 0.5
SEMANTIC = ("semantic", "llm", "word", "text", "transformer")


def stream(ref, seed=SEED):
    """`main()`'s own chunk sequence, rebuilt with its seed."""
    rng = random.Random(seed)
    random.seed(seed)
    return [(kind, ref.synth_chunk(kind, rng)) for kind, count in SEQUENCE for _ in range(count)]


def events(ref, flags):
    """Turn events from one detector's per-chunk decisions, as (ms, event)."""
    detector = ref.TurnDetector()
    fired = [(index * CHUNK_MS, detector.update(active)) for index, active in enumerate(flags)]
    return [(when, event) for when, event in fired if event]


def dbfs(chunk):
    return 20.0 * math.log10(max((sum(x * x for x in chunk) / len(chunk)) ** 0.5, 1e-10))


def probe(ref, chunks):
    """Per chunk kind: probabilities, dBFS levels, and the *smallest* peak-to-peak span."""
    seen = {}
    for kind, chunk in chunks:
        cell = seen.setdefault(kind, {"probs": set(), "level": [], "span": 9.9, "n": 0})
        cell["probs"].add(ref.fake_silero_vad(chunk, None))
        cell["level"].append(dbfs(chunk))
        cell["span"] = min(cell["span"], max(chunk) - min(chunk))
        cell["n"] += 1
    return seen


def semantic_names(ref):
    """Module-level names that would belong to the doc's third tier. There are none."""
    return [n for n in dir(ref) if any(word in n.lower() for word in SEMANTIC)]


def reads(function, name):
    """Is `name` mentioned anywhere in the body, or only in the signature?"""
    return name in inspect.getsource(function).split("\n", 1)[1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chunks = stream(ref)
    energy = [ref.energy_vad(chunk) for _, chunk in chunks]
    silero = [ref.fake_silero_vad(chunk, None) >= GATE for _, chunk in chunks]
    seen = probe(ref, chunks)
    return {
        "chunks": len(chunks), "agree": sum(a == b for a, b in zip(energy, silero)),
        "energy_events": events(ref, energy), "silero_events": events(ref, silero),
        "probs": sorted({p for cell in seen.values() for p in cell["probs"]}),
        "cough": sorted(seen["cough"]["probs"]), "cough_dbfs": seen["cough"]["level"][0],
        "cough_chunks": seen["cough"]["n"], "spans": {k: v["span"] for k, v in seen.items()},
        "duration": len(chunks[0][1]) / SR, "min_speech_ms": ref.TurnDetector().min_speech_ms,
        "uses_threshold": reads(ref.fake_silero_vad, "threshold"),
        "uses_prev": reads(ref.fake_silero_vad, "prev_state"),
        "semantic": semantic_names(ref),
    }


def verify(result):
    energy, silero = result["energy_events"], result["silero_events"]
    return [
        practice.Check(
            "ANSWER: the two event lists are identical, and so is every chunk decision",
            energy == silero and result["agree"] == result["chunks"],
            f"both arms emit {energy} -- the same four events -- and the two detectors agree on "
            f"{result['agree']}/{result['chunks']} chunks. There are no false positives in the "
            'list headed "many false positives on cough", and nothing was rejected in the list '
            'headed "rejects cough"',
        ),
        practice.Check(
            "FINDING: the Silero-style tier calls the cough speech; the state machine stops it",
            result["cough"] == [0.55] and result["cough_dbfs"] > -40.0,
            f"the cough scores {result['cough'][0]} -- above the {GATE} the caller compares "
            f"against -- and reads {result['cough_dbfs']:.1f} dBFS, above the -40 dBFS gate, so "
            f"both tiers class it as speech. What blocks a START is `min_speech_ms = "
            f"{result['min_speech_ms']}` against {result['cough_chunks']} chunk of "
            f"{CHUNK_MS} ms, and both arms run the same `TurnDetector`",
        ),
        practice.Check(
            "MECHANISM: the 0.92 branch cannot be reached, so the stream has two probabilities",
            result["probs"] == [0.02, 0.55] and result["duration"] < 0.03,
            f"0.92 needs `rms > 0.08 and not transient`, and `transient` requires "
            f"`duration < 0.03` -- every chunk is {result['duration']:.4f} s -- and a "
            f"peak-to-peak span above 0.6, which every loud chunk has -- narrowest speech "
            f"{result['spans']['speech']:.3f}, cough {result['spans']['cough']:.3f}. So the "
            f"whole stream takes {result['probs']}",
        ),
        practice.Check(
            "FINDING: two tiers are instantiated, not three, and the coughs are one chunk",
            not result["semantic"] and result["cough_chunks"] == 1,
            f"`main()` compares `energy_vad` and `fake_silero_vad`; the doc's third tier is a "
            f"semantic turn detector and no module-level name matches {list(SEMANTIC)} "
            f"({result['semantic']}). The cough is `('cough', {result['cough_chunks']})` -- "
            f"{result['cough_chunks'] * CHUNK_MS} ms of a {result['chunks'] * CHUNK_MS} ms stream",
        ),
        practice.Check(
            "CONTROL: `fake_silero_vad` never reads two of its three parameters",
            not result["uses_threshold"] and not result["uses_prev"],
            "`prev_state` and `threshold=0.5` appear in the signature and nowhere in the body, "
            "which hardcodes 0.92, 0.55 and 0.02. The caller re-implements the comparison as "
            "`silero_prob >= 0.5`, so the argument is inert",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
