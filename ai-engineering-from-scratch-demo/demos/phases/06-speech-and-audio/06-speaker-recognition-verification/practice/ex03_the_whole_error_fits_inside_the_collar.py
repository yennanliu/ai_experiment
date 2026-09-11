"""Exercise 3 — the whole error fits inside the collar.

    **Hard.** Build the full enroll -> diarize -> verify pipeline with
    `pyannote.audio`. Evaluate DER on AMI dev set.

Reading of the exercise: `pyannote`, `torch`, `torchaudio` and `speechbrain` are
absent and AMI is not here, so this is the `DESIGN D11` scaled-down run, built to
the exercise's own three stages. **Enroll**: one clean utterance per voice through
the lesson's `embed_mfcc_stats`. **Diarize**: an energy VAD, then 0.5 s windows
slid inside the speech regions only. **Verify**: every window scored against every
enrolled speaker with the lesson's `cosine`, best wins. DER is then scored at 10 ms
against the NIST definition -- missed speech plus false alarm plus confusion, over
reference speech time. The corpus is scaled down; the metric is the real one.

It works: **DER 0.0134** -- 0.13% missed, 1.21% false alarm, **0.00% confusion**,
every window of every speaker verified against the right model. And that number is
not the answer, because two conventions the exercise does not mention move it
further than the pipeline does.

**The collar.** Scored with the standard 0.25 s forgiveness collar around every
reference boundary, the same hypothesis scores **0.0000**. All of the error lies
within a quarter-second of a turn boundary, which is exactly where a sliding
window cannot help but straddle two speakers. "Report DER" has two answers, 1.34%
and 0.00%, for one segmentation -- and AMI numbers in the literature are quoted
both ways.

**The mapping.** Enrolment is what makes the labels *named*. A clustering
diarizer -- which is what `pyannote` is -- produces anonymous clusters, so DER has
to search the assignment, and over the six permutations of three labels this same
hypothesis scores from **0.0134 to 1.0121**: a range wider than the metric itself,
for a step nothing in the name "diarization error rate" announces.

And of "enroll -> diarize -> verify", `code/main.py` ships the last stage. It has
embeddings and a trial list; no VAD, no change detection, no clustering, no
overlap handling, and no DER. Everything above the embeddings was written here,
the metric included.

Structure: `conversation` builds the audio and its reference turns;
`regions` is the VAD; `windows` slides inside it; `enroll` and `label` are the other two
stages; `cell` scores one 10 ms frame and `der` accumulates them.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "06-speaker-recognition-verification"
SR, WINDOW, HOP, COLLAR, STEP = 8000, 0.5, 0.25, 0.25, 0.01
VOICES = {"alice": [200, 400, 600], "bob": [300, 600, 1200], "carol": [260, 520, 780]}
SECONDS, NOISE, VAD_RMS, FRAME = 20.0, 0.05, 0.05, 0.02
ABSENT = ("pyannote", "torch", "torchaudio", "speechbrain")


def conversation(ref, seconds=SECONDS, seed=11):
    """Turns of 0.8-2.0 s separated by 0.3-0.8 s of near-silence, plus the reference."""
    rnd, audio, turns, clock = random.Random(seed), [], [], 0.0
    random.seed(seed)
    while clock < seconds:
        who, span, gap = rnd.choice(list(VOICES)), rnd.uniform(0.8, 2.0), rnd.uniform(0.3, 0.8)
        audio += ref.tone_mix(VOICES[who], SR, span, noise=NOISE)
        audio += [random.gauss(0, 0.005) for _ in range(int(SR * gap))]
        turns.append((clock, clock + span, who))
        clock += span + gap
    return audio, turns, clock


def regions(audio, threshold=VAD_RMS):
    """Speech spans by frame energy -- the stage the lesson does not have."""
    size = int(FRAME * SR)
    loud = [math.sqrt(sum(x * x for x in audio[i:i + size]) / size) > threshold
            for i in range(0, len(audio) - size + 1, size)]
    edges, previous = [], False
    for index, active in enumerate([*loud, False]):
        if active != previous:
            edges.append(index * FRAME)
            previous = active
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= WINDOW]


def windows(spans):
    """Sliding windows inside each span, plus a final one flush to its end."""
    out = []
    for start, end in spans:
        starts = [start + i * HOP for i in range(max(0, int((end - start - WINDOW) / HOP) + 1))]
        if starts and starts[-1] + WINDOW < end - 1e-9:
            starts.append(end - WINDOW)
        out += [(s, s + WINDOW) for s in starts]
    return out


def enroll(ref, seconds=1.0, seed=5):
    """One clean utterance per voice -- the only stage `main()` already has."""
    random.seed(seed)
    return {name: ref.embed_mfcc_stats(ref.tone_mix(freqs, SR, seconds, noise=NOISE), SR)
            for name, freqs in VOICES.items()}


def label(ref, audio, spans, models):
    """Verify each window against every enrolled speaker; the best cosine wins."""
    out = []
    for a, b in spans:
        window = ref.embed_mfcc_stats(audio[int(a * SR):int(b * SR)], SR)
        out.append(max(models, key=lambda n: ref.cosine(window, models[n])))
    return out


def at(spans, when, values=None):
    """The label covering `when`, or None -- reference and hypothesis alike."""
    hit = next((i for i, s in enumerate(spans) if s[0] <= when < s[1]), None)
    return None if hit is None else (spans[hit][2] if values is None else values[hit])


def cell(truth, guess, mapping):
    """One 10 ms frame as (reference speech, missed, false alarm, confusion)."""
    speech, heard = truth is not None, guess is not None
    return (STEP * speech, STEP * (speech and not heard), STEP * (heard and not speech),
            STEP * (speech and heard and mapping[guess] != truth))


def der(turns, spans, labels, mapping, collar=0.0):
    """NIST DER at 10 ms: (missed + false alarm + confusion) / reference speech."""
    totals = [0.0] * 4
    for step in range(int(turns[-1][1] / STEP) + 1):
        when = step * STEP
        if collar and any(abs(when - s) < collar or abs(when - e) < collar for s, e, _ in turns):
            continue
        cells = cell(at(turns, when), at(spans, when, labels), mapping)
        totals = [running + one for running, one in zip(totals, cells)]
    speech, miss, alarm, confusion = totals
    return {"der": (miss + alarm + confusion) / speech, "miss": miss / speech,
            "alarm": alarm / speech, "confusion": confusion / speech}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    audio, turns, _ = conversation(ref)
    gated, named = windows(regions(audio)), dict(zip(VOICES, VOICES))
    labels = label(ref, audio, gated, enroll(ref))
    scored = {order: der(turns, gated, labels, dict(zip(VOICES, order)))["der"]
              for order in itertools.permutations(VOICES)}
    return {"absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
            "scored": der(turns, gated, labels, named), "orders": len(scored),
            "collared": der(turns, gated, labels, named, COLLAR),
            "identity": scored[tuple(VOICES)], "worst": max(scored.values()),
            "windows": len(gated), "found": len(set(labels))}


def verify(result):
    scored, collared = result["scored"], result["collared"]
    return [
        practice.Check(
            "ANSWER: the scaled-down pipeline scores DER 0.0134 with zero confusion",
            scored["confusion"] == 0.0 and scored["der"] < 0.05 and result["found"] == 3,
            f"{result['absent']} are all absent and AMI is not here, so: enrol one voice each, "
            f"VAD, {result['windows']} windows of {WINDOW} s, verify each against all three. "
            f"DER {scored['der']:.4f} = {scored['miss']:.4f} miss + {scored['alarm']:.4f} "
            f"alarm + {scored['confusion']:.4f} confusion",
        ),
        practice.Check(
            "FINDING: the standard 0.25 s collar takes the same hypothesis to 0.0000",
            collared["der"] == 0.0,
            f"every one of those {scored['der'] * 100:.2f} points lies within {COLLAR} s of a "
            "reference boundary, where a sliding window must straddle two speakers. The "
            "literature quotes AMI both ways, so 'report DER' has two answers",
        ),
        practice.Check(
            "FINDING: the label mapping moves DER further than the pipeline does",
            result["worst"] > 1.0 > result["identity"],
            f"enrolment is what makes these labels named; a clustering diarizer produces "
            f"anonymous clusters, so DER must search the assignment: over the "
            f"{result['orders']} permutations this same hypothesis scores "
            f"{result['identity']:.4f} to {result['worst']:.4f}",
        ),
        practice.Check(
            "CONTROL: of enroll -> diarize -> verify, `main()` ships the last stage",
            len(result["absent"]) == len(ABSENT),
            "it has embeddings and a trial list: no VAD, no clustering, no DER",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
