"""Exercise 3 -- verification cuts hallucinated instances by 58-69% but cannot fix a miss.

    Build a "counting strict" mode: the synthesizer extracts each counted instance with a timestamp and the user clicks to verify. Measure whether user-verification reduces hallucination.

Reading of the exercise: the lesson has no synthesizer, so a stand-in one
answers 400 counting questions (a 60 s scene with 1-8 gold events at known
times). It misses each event with probability `miss`, reports a found event
twice (a car seen in two frames) with probability `dup`, and adds up to 3
invented instances, each with probability `fake`. Free mode answers with the
number of instances. Strict mode lists each instance with its timestamp. The
simulated user clicks it and sees the frame. An instance with no gold event
within 1 s is rejected (5% of the time the user wrongly accepts it). A second
instance of an event already accepted is caught half the time. Hallucination
is the share of reported instances beyond one per real event. Three rate
settings are run: balanced, over-counting and under-counting. The lesson's
own `ground_window` is then run on its counting demo query.

**ANSWER: yes for hallucination, not always for the count.** Strict mode
cuts the hallucinated share (instances beyond one per real event: invented
or double-counted) from 19.4% to 7.3% (balanced), 33.3% to 14.0%
(over-counting) and 8.6% to 2.7% (under-counting). Exact-count accuracy
rises from 33.0% to 52.5% in the balanced setting and from 14.2% to 52.5%
when the model over-counts. It does not move when the model under-counts
(34.5% -> 33.8%): verification only removes instances and never recovers a
miss. There 85.9% of the wrong counts were already too low, and 96.6% are
after verification.

**FINDING: the lesson's counting query is grounded on "the".** In scene 2
("let me count the vehicles approaching") the only query token in the
transcript is "the", so `ground_window` returns 01:36-01:53: 17 s of a 64 s
scene (26.7%), keyed on a stopword. A strict mode that extracts instances
inside that window cannot count the cars in the other 73%.

**FINDING: the doc's sample output cites a window outside its own top
scene.** "Use It" prints top scene 3 at [01:32-01:54] and then a refined
window [00:12-00:58]. `ground_window` always stays inside the scene; that
holds on all 6 lesson scenes x 5 queries.

Structure: `synthesize` is the stand-in; `verify_clicks` is the user;
`run` scores one rate setting.
"""

from __future__ import annotations

import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "12-video-understanding-pipeline"
SETTINGS = {"balanced": {"miss": 0.10, "dup": 0.10, "fake": 0.15},
            "over": {"miss": 0.03, "dup": 0.25, "fake": 0.30},
            "under": {"miss": 0.25, "dup": 0.03, "fake": 0.05}}
TOL, USER_SLIP, DUP_CAUGHT = 1.0, 0.05, 0.5
QUERIES = ["how many cars pass through the intersection", "what happened first pour or stir", "plating of the dish",
           "ocean at sunset", "the chef"]


def fixture():
    rng = random.Random(0)
    return [sorted(rng.uniform(1, 59) for _ in range(rng.randint(1, 8))) for _ in range(400)]


def synthesize(rng, events, p):
    out = []
    for t in events:
        if rng.random() < p["miss"]:
            continue
        out.append(t + rng.uniform(-0.5, 0.5))
        if rng.random() < p["dup"]:
            out.append(t + rng.uniform(0.3, 0.9))
    return sorted(out + [rng.uniform(0, 60) for _ in range(3) if rng.random() < p["fake"]])


def nearest(events, t):
    i = min(range(len(events)), key=lambda j: abs(events[j] - t))
    return i if abs(events[i] - t) <= TOL else None


def verify_clicks(rng, events, instances):
    accepted, seen = [], set()
    for t in instances:
        i = nearest(events, t)
        if i is None and rng.random() >= USER_SLIP:
            continue
        if i is not None and i in seen and rng.random() < DUP_CAUGHT:
            continue
        seen.add(i)
        accepted.append(t)
    return accepted


def score(events_list, answers):
    exact = sum(len(a) == len(e) for e, a in zip(events_list, answers)) / len(answers)
    backed = sum(len({nearest(e, t) for t in a} - {None}) for e, a in zip(events_list, answers))
    fake = sum(map(len, answers)) - backed  # instances beyond one per real event: invented or double-counted
    under = sum(len(a) < len(e) for e, a in zip(events_list, answers))
    wrong = sum(len(a) != len(e) for e, a in zip(events_list, answers))
    return {"exact": round(exact, 3), "halluc": round(fake / max(1, sum(map(len, answers))), 3),
            "under_share": round(under / max(1, wrong), 3)}


def run(p, seed):
    rng, gold = random.Random(seed), fixture()
    free = [synthesize(rng, e, p) for e in gold]
    strict = [verify_clicks(rng, e, a) for e, a in zip(gold, free)]
    return {"free": score(gold, free), "strict": score(gold, strict)}


def grounding(ref):
    scene = ref.SAMPLE[2]
    start, end = ref.ground_window(QUERIES[0], scene)
    matched = sorted(set(ref.tokenize(QUERIES[0])) & set(ref.tokenize(scene.transcript)))
    inside = all(s.start_ms <= w[0] <= w[1] <= s.end_ms for s in ref.SAMPLE for w in
                 (ref.ground_window(q, s) for q in QUERIES))
    use_it = parity.doc_text(PHASE, LESSON).split("## Use It")[1]
    top = re.search(r"top scene: scene \d+ \[(\d\d:\d\d)-(\d\d:\d\d)\]", use_it).groups()
    refined = re.search(r"refined window: \[(\d\d:\d\d)-(\d\d:\d\d)\]", use_it).groups()
    return {"window": (ref.fmt_ms(start), ref.fmt_ms(end)), "matched": matched,
            "share": round((end - start) / (scene.end_ms - scene.start_ms), 3), "inside": inside,
            "doc_top": top, "doc_refined": refined}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"runs": {name: run(p, 1) for name, p in SETTINGS.items()}, **grounding(ref)}


def verify(result):
    runs = result["runs"]
    table = {n: (r["free"]["exact"], r["strict"]["exact"], r["free"]["halluc"], r["strict"]["halluc"]) for n, r in runs.items()}
    return [
        practice.Check(
            "ANSWER: verification cuts hallucinated instances in all 3 settings; the count does not improve on misses",
            table == {"balanced": (0.33, 0.525, 0.194, 0.073), "over": (0.142, 0.525, 0.333, 0.14),
                      "under": (0.345, 0.338, 0.086, 0.027)}
            and runs["under"]["strict"]["under_share"] > runs["under"]["free"]["under_share"],
            f"(exact free, exact strict, hallucinated free, hallucinated strict): {table}; share of wrong counts "
            f"that are too low, under-counting model: {runs['under']['free']['under_share']} -> "
            f"{runs['under']['strict']['under_share']}",
        ),
        practice.Check(
            "FINDING: the lesson's counting demo query is grounded on the stopword 'the'",
            (result["matched"], result["window"], result["share"]) == (["the"], ("01:36", "01:53"), 0.267),
            f"query tokens in scene 2's transcript: {result['matched']}; window {result['window']}, "
            f"{result['share']:.1%} of the scene",
        ),
        practice.Check(
            "FINDING: the doc's Use It window lies outside its own top scene; ground_window never does that",
            result["doc_top"] == ("01:32", "01:54") and result["doc_refined"] == ("00:12", "00:58")
            and result["inside"],
            f"doc: top scene {result['doc_top']}, refined {result['doc_refined']}; ground_window inside the scene "
            f"on 6 scenes x {len(QUERIES)} queries: {result['inside']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
