"""Exercise 5 -- per-speaker turns answer 78% of "what did Alice say" queries and the shipped index 0%.

    Add speaker-diarized transcript: run pyannote speaker diarization on the audio and embed per-speaker transcripts. Demonstrate "what did Alice say about X?" queries.

Reading of the exercise: pyannote needs audio, a GPU-sized model and a
Hugging Face token, and the lesson has no audio. So the diarizer is replaced
by what it outputs: speaker turns with anonymous labels (`SPEAKER_00`, ...),
with errors injected at pyannote community-1's published DERs (11.2%
VoxConverse, 17.0% AMI, 20.2% DIHARD 3; github.com/pyannote/pyannote-audio,
read 2026-09-29). An error relabels whole turns, chosen at random until that
share of speech time is wrong. The recording is a 3-scene meeting: Alice, Bob
and Carol each introduce themselves ("i am alice"), then take turns on six
topics, two speakers per topic. A label gets a name from the turn that
contains its introduction. Each turn is embedded with the lesson's
`fake_embed`. A query "what did <name> say about <topic>?" ranks only that
name's turns, by cosine to the topic words. The baseline is the lesson as
shipped: one transcript per scene, `multi_vector_search`, then
`ground_window`. An answer is right if its window overlaps the gold turn at
IoU >= 0.5. Each setting runs 20 seeds.

**ANSWER: per-speaker turns answer the query; the shipped index cannot.**
With a clean diarization, 77.9% of the 12 queries x 20 seeds land on the
right turn. The misses are hash collisions in the lesson's 24-dim
`fake_embed`: at `EMB_DIM = 256` the same run scores 99.6%. The shipped
scene index answers 0%. Accuracy falls with DER: 0.112 -> 52.1%, 0.17 ->
46.3%, 0.202 -> 44.6%.

**FINDING: a relabelled introduction takes out a speaker's whole set of
answers.** A wrong label on an intro turn leaves a speaker unnamed or names
two labels alike. At DER 0.202 that happens in 45% of runs, and those runs
hold 49% of the misses.

**FINDING: the shipped grounding keys on "the".** Every one of the 12 topic
turns contains "the", and so does the query, so `ground_window` stretches
from the first "the" to the last. Its windows average 22.5 s against 6 s
gold turns. The transcript also carries no speaker, so nothing in it can
tell Alice's budget view from Bob's.

Structure: `TURNS` is the meeting; `diarize` injects the label errors;
`ask_diarized` answers from one speaker's turns; `shipped_answers` runs the
lesson's index as shipped.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "12-video-understanding-pipeline"
DERS, SEEDS = (0.0, 0.112, 0.17, 0.202), 20
NAMES = ("alice", "bob", "carol")
SAID = {"budget": {"alice": "cut travel spend by ten percent", "bob": "double the cloud reserve"},
    "launch": {"bob": "slip the date to march", "carol": "ship the beta to partners first"},
    "hiring": {"alice": "freeze new roles until spring", "carol": "open two designer seats"},
    "security": {"carol": "rotate every api key monthly", "bob": "audit vendor access logs"},
    "pricing": {"alice": "raise the pro tier slightly", "bob": "keep student plans free"},
    "roadmap": {"carol": "drop the legacy importer", "alice": "focus on mobile sync"}}  # topic -> speaker -> words


LINES = [(n, f"hi everyone i am {n}", 3) for n in NAMES]  # (speaker, words, seconds): intros, then the topics
LINES += [(n, f"on the {topic} we should {words}", 6) for topic, said in SAID.items() for n, words in said.items()]
TURNS = [{"speaker": n, "text": t, "start": e - d, "end": e}
         for (n, t, d), e in zip(LINES, itertools.accumulate(d for *_, d in LINES))]
QUERIES = [(f"what did {tr['speaker']} say about the {tr['text'].split()[2]}?", tr["speaker"], tr["text"].split()[2],
            (tr["start"], tr["end"])) for tr in TURNS[len(NAMES):]]  # (query, name, topic, gold window)


def diarize(rng, der):  # pyannote-shaped: anonymous labels, whole turns relabelled until `der` of speech is wrong
    labels = {n: f"SPEAKER_{i:02d}" for i, n in enumerate(NAMES)}
    out, wrong = [dict(tr, label=labels[tr["speaker"]]) for tr in TURNS], 0.0
    for tr in rng.sample(out, len(out)):
        if wrong >= der * TURNS[-1]["end"]:
            break
        tr["label"] = rng.choice([v for v in labels.values() if v != tr["label"]])
        wrong += tr["end"] - tr["start"]
    return out


def iou(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))


def ask_diarized(ref, turns, label, topic):  # only that label's turns, ranked by the lesson's embedding
    best = max((tr for tr in turns if tr["label"] == label), default={"start": 0.0, "end": 0.0},
               key=lambda tr: ref.cosine(ref.fake_embed(topic), ref.fake_embed(tr["text"])))
    return best["start"], best["end"]


def shipped_answers(ref):  # the lesson as shipped: 3 mixed-transcript scenes, RRF search, ground_window
    scenes = [ref.Scene("meet", i, int(p[0]["start"] * 1000), int(p[-1]["end"] * 1000), "meeting room",
                        " ".join(tr["text"] for tr in p), "table people laptop")
              for i, p in enumerate((TURNS[:6], TURNS[6:12], TURNS[12:]))]
    list(map(ref.Scene.embed, scenes))
    return [tuple(t / 1000 for t in ref.ground_window(q, ref.multi_vector_search(q, scenes, k=1)[0][0])) for q, *_ in QUERIES]


def salted(ref, seed, dim=24):
    ref.EMB_DIM, ref.hash = dim, lambda s: int.from_bytes(hashlib.blake2b(f"{seed}|{s}".encode(), digest_size=8).digest(), "big")


def run(ref, seed, der):  # per query right or wrong, and whether the name map broke
    diar = diarize(random.Random(seed), der)
    names = {m.group(1): tr["label"] for tr in diar if (m := re.search(r"i am (\w+)", tr["text"]))}
    wins = [iou(ask_diarized(ref, diar, names.get(n), t), g) >= 0.5 for _, n, t, g in QUERIES]
    return wins, len(set(names.values())) < len(NAMES)


def shipped_stats(ref, n):
    right, spans = 0, []
    for seed in range(SEEDS):
        salted(ref, seed)
        answers = shipped_answers(ref)
        right += sum(iou(a, g) >= 0.5 for a, (*_, g) in zip(answers, QUERIES))
        spans += [b - a for a, b in answers]
    return {"shipped": round(right / n, 3), "shipped_span": round(sum(spans) / len(spans), 1)}


def solve():
    ref, n = parity.load_reference(PHASE, LESSON, "main"), SEEDS * len(QUERIES)
    acc, worst, wide = dict.fromkeys(DERS, 0), [], 0
    for seed in range(SEEDS):
        salted(ref, seed)
        for der in DERS:
            wins, broken = run(ref, seed, der)
            acc[der] += sum(wins)
        worst.append((broken, wins.count(False)))  # at the largest DER
        salted(ref, seed, 256)
        wide += sum(run(ref, seed, 0.0)[0])
    return {**shipped_stats(ref, n), "acc": {d: round(v / n, 3) for d, v in acc.items()}, "wide": round(wide / n, 3),
            "intro_broken": sum(b for b, _ in worst) / SEEDS,
            "broken_share_of_misses": round(sum(m for b, m in worst if b) / max(1, sum(m for _, m in worst)), 2),
            "turns_with_the": sum(" the " in f" {tr['text']} " for tr in TURNS[len(NAMES):])}


def verify(r):
    return [
        practice.Check(
            "ANSWER: diarized per-speaker turns answer 77.9% at DER 0 (99.6% at EMB_DIM 256); the shipped index 0",
            (len(QUERIES), r["acc"], r["shipped"], r["wide"]) == (12, {0.0: 0.779, 0.112: 0.521, 0.17: 0.463, 0.202: 0.446}, 0.0, 0.996),
            f"IoU>=0.5 accuracy by DER, {SEEDS} seeds: {r['acc']}; DER 0 at EMB_DIM 256: {r['wide']}; shipped {r['shipped']}",
        ),
        practice.Check(
            "FINDING: a relabelled introduction takes out a speaker's whole set of answers",
            (r["intro_broken"], r["broken_share_of_misses"]) == (0.45, 0.49),
            f"DER 0.202: runs with a broken name map {r['intro_broken']:.0%}, their share of misses {r['broken_share_of_misses']:.0%}",
        ),
        practice.Check(
            "FINDING: the shipped grounding keys on 'the', which every topic turn contains",
            (r["shipped_span"], r["turns_with_the"]) == (22.5, 12),
            f"mean shipped window {r['shipped_span']} s vs 6 s gold turns; topic turns with 'the': {r['turns_with_the']}/12",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
