"""Exercise 2 — the threshold has three settings, and the doc recommends two of them.

    **Medium.** Install `silero-vad`, process a 5-min recording, tune threshold
    to minimize both first-word clips and false triggers. Report
    precision/recall.

Reading of the exercise: `silero_vad`, `torch` and `torchaudio` are all absent
and the reference tree ships no recording of any length, so the tuning happens
against `fake_silero_vad`, which is the thing the lesson actually provides. Its
output takes two values on this stream (Exercise 1), so sweeping the threshold in
hundredths over `[0, 1]` -- **101 settings** -- produces **3** distinct event
lists:

| threshold band | turn events | precision | recall | F1 |
|---|---|---:|---:|---:|
| 0.00 – 0.02 | one START, never an END | 0.000 | 0.000 | **0.000** |
| **0.03 – 0.55** | START, END, START, END | 1.000 | 1.000 | **1.000** |
| 0.56 – 1.00 | none at all | 0.000 | 0.000 | **0.000** |

The doc names two thresholds -- `0.5` as the default and `0.3` as "sensitive" --
and both fall in the **same** band, so they are the same setting. The dial has one
usable position and two ways to break, and the "minimize both" the exercise asks
for has no interior to search.

**The parameter is inert anyway.** `fake_silero_vad(chunk, prev_state,
threshold=0.5)` never mentions `threshold` after the signature line; the caller
re-implements the comparison as `silero_prob >= 0.5`. Passing a different
`threshold=` changes nothing.

**And first-word clipping is not on this dial.** Speech begins at **200 ms** and
START fires at **440 ms**: **240 ms clipped**, identical at 0.3 and at 0.5,
because the delay is `min_speech_ms = 250` and not the threshold. The parameter
that exists to fix it, `pre_roll_ms = 300`, is assigned in `__init__` and **never
read in `update()`** -- so the demo ships the pitfall its own doc names, "No
pre-roll buffer. First 200-300 ms of user audio lost." The only threshold that
shortens the clip is 0.02, which shortens it to 40 ms by calling silence speech,
and then no END ever fires.

The end-pointing delay is off by one chunk in the other direction:
`silence_hangover_ms` is **500** and the measured gap from speech offset to END
is **480 ms**, because `silence_ms` is compared after being incremented.

Structure: `stream` rebuilds `main()`'s chunks; `gold` reads the turn ends out of
the chunk labels; `sweep` runs every threshold; `score` is precision, recall and
F1 against the gold ends with a collar.
"""

from __future__ import annotations

import importlib.util
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "14-voice-activity-detection-turn-taking"
SEQUENCE = (("silence", 10), ("speech", 40), ("silence", 30), ("cough", 1),
            ("silence", 10), ("speech", 25), ("silence", 35))
CHUNK_MS, SEED, COLLAR, STEPS = 20, 42, 700, 100
DOC_THRESHOLDS = (0.3, 0.5)
ABSENT = ("silero_vad", "torch", "torchaudio", "onnxruntime")


def stream(ref, seed=SEED):
    rng = random.Random(seed)
    random.seed(seed)
    return [(kind, ref.synth_chunk(kind, rng)) for kind, count in SEQUENCE for _ in range(count)]


def gold(chunks):
    """Turn boundaries read off the chunk labels: the end of each run of speech."""
    speech = [kind == "speech" for kind, _ in chunks] + [False]
    return [i * CHUNK_MS for i in range(1, len(speech))
            if speech[i - 1] and not speech[i]]


def events(ref, probs, threshold):
    detector = ref.TurnDetector()
    fired = [(i * CHUNK_MS, detector.update(p >= threshold)) for i, p in enumerate(probs)]
    return [(when, event) for when, event in fired if event]


def score(ends, truth, collar=COLLAR):
    """Precision, recall and F1 of predicted turn ends against gold, within a collar."""
    used, hits = set(), 0
    for when in ends:
        near = [g for g in truth if abs(when - g) <= collar and g not in used]
        if near:
            used.add(near[0])
            hits += 1
    precision = hits / max(1, len(ends))
    recall = hits / max(1, len(truth))
    return precision, recall, 2 * precision * recall / max(1e-9, precision + recall)


def clipping(ref, probs, onset, thresholds=DOC_THRESHOLDS):
    """Milliseconds of speech already past when START fires, per threshold."""
    out = {}
    for threshold in thresholds:
        starts = [w for w, e in events(ref, probs, threshold) if e == "START"]
        out[threshold] = starts[0] - onset if starts else None
    return out


def sweep(ref, probs, truth, steps=STEPS):
    """Every threshold in hundredths, grouped by the event list it produces."""
    bands = {}
    for step in range(steps + 1):
        threshold = step / steps
        fired = tuple(events(ref, probs, threshold))
        bands.setdefault(fired, []).append(threshold)
    return {key: (min(v), max(v), score([w for w, e in key if e == "END"], truth))
            for key, v in bands.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chunks = stream(ref)
    probs = [ref.fake_silero_vad(chunk, None) for _, chunk in chunks]
    truth = gold(chunks)
    onset = min(i for i, (kind, _) in enumerate(chunks) if kind == "speech") * CHUNK_MS
    detector = ref.TurnDetector()
    body = inspect.getsource(ref.TurnDetector.update)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "bands": sweep(ref, probs, truth), "gold": truth, "onset": onset,
        "clip": clipping(ref, probs, onset),
        "hangover": detector.silence_hangover_ms, "min_speech": detector.min_speech_ms,
        "measured_hangover": [w for w, e in events(ref, probs, 0.5) if e == "END"][0] - truth[0],
        "pre_roll": detector.pre_roll_ms, "pre_roll_read": "pre_roll" in body,
        "uses_threshold": "threshold" in inspect.getsource(ref.fake_silero_vad).split("\n", 1)[1],
    }


def verify(result):
    bands = result["bands"]
    good = [v for v in bands.values() if v[2][2] == 1.0]
    clips = set(result["clip"].values())
    return [
        practice.Check(
            "CONTROL: no Silero, no recording, so the dial under test is the lesson's own",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']} and the reference tree ships no audio "
            f"file of any length, so the threshold is swept against `fake_silero_vad` over "
            f"{result['gold']} ms of gold turn ends",
        ),
        practice.Check(
            "ANSWER: 101 thresholds produce 3 distinct behaviours, and only one works",
            len(bands) == 3 and len(good) == 1,
            "the bands are "
            + "; ".join(f"{lo:.2f}-{hi:.2f} -> P {s[0]:.3f} R {s[1]:.3f} F1 {s[2]:.3f}"
                        for lo, hi, s in sorted(bands.values()))
            + ". Below the middle band silence counts as speech, so one START fires and no END "
              "ever can; above it nothing is speech and no event fires at all",
        ),
        practice.Check(
            "FINDING: the doc's two recommended thresholds are the same setting",
            len({(lo, hi) for lo, hi, _ in bands.values()
                 if lo <= min(DOC_THRESHOLDS) and max(DOC_THRESHOLDS) <= hi}) == 1,
            f"{DOC_THRESHOLDS[1]} is called the default and {DOC_THRESHOLDS[0]} 'sensitive', and "
            f"both land in the same band, so they produce identical events and identical "
            f"clipping. `threshold` is inert in `fake_silero_vad` besides -- it is never "
            f"mentioned after the signature ({result['uses_threshold']}), and the caller "
            "re-implements the comparison",
        ),
        practice.Check(
            "FINDING: first-word clipping is 240 ms and is not on the threshold dial",
            clips == {240} and not result["pre_roll_read"],
            f"speech begins at {result['onset']} ms and START fires 240 ms later at both "
            f"{DOC_THRESHOLDS[0]} and {DOC_THRESHOLDS[1]} -- the delay is "
            f"`min_speech_ms = {result['min_speech']}`, not the threshold. `pre_roll_ms = "
            f"{result['pre_roll']}` is assigned in `__init__` and never read in `update()`, so "
            "the demo ships the pitfall its own doc names",
        ),
        practice.Check(
            "CONTROL: the end-pointing delay is one chunk short of its setting",
            result["measured_hangover"] == result["hangover"] - CHUNK_MS,
            f"`silence_hangover_ms` is {result['hangover']} and the measured gap from speech "
            f"offset to END is {result['measured_hangover']} ms, because `silence_ms` is "
            f"compared after being incremented by {CHUNK_MS} ms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
