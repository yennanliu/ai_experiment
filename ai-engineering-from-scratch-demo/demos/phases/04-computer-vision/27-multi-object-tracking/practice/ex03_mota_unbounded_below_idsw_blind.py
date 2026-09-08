"""Exercise 3 — mota unbounded below idsw blind.

    **(Hard)** Integrate SAM 2's memory-based tracker (via `transformers`) as an
    alternative tracker backend. Run both SimpleTracker and SAM 2 on a 30-second
    clip of a crowd and compare ID-switch counts, manually labelling ground-truth
    IDs for 5 salient people.

Reading of the exercise: the second backend cannot be built -- `transformers`
and `sam2` both raise ModuleNotFoundError -- and the lesson ships no crowd clip
and no hand-labelled identities either, so the comparison has one arm. What is
left is worth more than the comparison would have been, because the quantity the
exercise picks to compare on ranks nothing: ID-switch count is blind to false
positives. A tracker that emits nine copies of every box scores exactly 0 ID
switches, the same as a flawless run, while MOTA on the same output is -7.000.
That negative is not a bug: MOTA subtracts (FN + FP + IDSW) / GT from 1 and FP
has no ceiling, so on this fixture MOTA is exactly 2 - k for k copies and falls
without bound -- an "accuracy" of -700%. The lesson's own worst case does not
reach that; at 30 objects it is +0.260. And an ID switch is not the only way to
lose an identity: an object that steps aside for two frames while the detector
misses it is fragmented once, keeps its track ID, and is scored 0 by
`count_id_switches` -- indistinguishable from perfect on the exercise's metric.

Structure: `missing` records the absent backends as measurements rather than
skipping. `assign` is one frame of CLEAR-MOT matching -- Hungarian on IoU with a
0.5 gate, exclusive, unlike `count_id_switches`'s per-object argmax -- and
`clearmot` accumulates FN, FP, IDSW, fragmentations and MOTA over a sequence.
`detour` is the fragmentation fixture: a stationary object that shifts 12px
aside for the two frames its detector misses it, far enough to break the 0.5
evaluation gate but not the 0.3 association gate, so the same track ID resumes.
The module-level `copies` is the false-positive arm, k identically-placed
hypotheses per real one; exercise 1's `drive` runs the lesson's tracker.
"""

from __future__ import annotations

import importlib
import pathlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "27-multi-object-tracking"

STACK, GATE, COUNTS, COPIES = ("transformers", "sam2"), 0.5, (3, 10, 30), (1, 2, 3, 5, 9)
SIDE, HOLE, SPAN = 12.0, (8, 10), 20
SIBLING = pathlib.Path(__file__).with_name("ex01_clipping_not_crowding_breaks_iou.py")

copies = lambda per_frame, k: [[(tid * 100 + j, box) for tid, box in tracks     # noqa: E731
                                for j in range(k)] for tracks in per_frame]
pull = lambda table, keys, field: [table[k][field] for k in keys]               # noqa: E731
round3 = lambda values: [round(v, 3) for v in values]                           # noqa: E731
linear = lambda: [2.0 - k for k in COPIES]                                      # noqa: E731
rate = lambda solve_, ref, np, per_frame, gt: dict(                             # noqa: E731
    clearmot(solve_, ref, np, per_frame, gt), lesson=ref.count_id_switches(per_frame, gt))


def missing() -> dict:
    found = {}
    for name in STACK:
        try:
            importlib.import_module(name)
            found[name] = "present"
        except ImportError as exc:              # pragma: no cover - neither is installed here
            found[name] = type(exc).__name__
    return found


def assign(solver, ref, np, tracks, truth) -> dict:
    if not tracks or not truth:
        return {}
    iou = ref.bbox_iou(np.array([b for _, b in truth]), np.array([b for _, b in tracks]))
    cost = np.where(iou < GATE, 1e6, 1.0 - iou)
    rows, columns = solver(cost)
    return {truth[r][0]: tracks[c][0] for r, c in zip(rows, columns) if cost[r, c] < 1.0}


def clearmot(solver, ref, np, per_frame, gt) -> dict:
    prev, live, tally = {}, {}, {"fn": 0, "fp": 0, "idsw": 0, "frag": 0, "gt": 0}
    for tracks, truth in zip(per_frame, gt):
        pairs = assign(solver, ref, np, tracks, truth)
        tally["gt"] += len(truth)
        tally["fp"] += len(tracks) - len(pairs)
        for gid, _ in truth:
            hit = pairs.get(gid)
            tally["fn"] += int(hit is None)
            tally["idsw"] += int(hit is not None and prev.get(gid, hit) != hit)
            tally["frag"] += int(hit is not None and live.get(gid) is False)
            prev[gid] = hit if hit is not None else prev.get(gid)
            live[gid] = hit is not None
    tally["mota"] = 1.0 - (tally["fn"] + tally["fp"] + tally["idsw"]) / tally["gt"]
    return tally


def detour() -> tuple:
    gt, dets = [], []
    for index in range(SPAN):
        shift = SIDE if HOLE[0] <= index < HOLE[1] else 0.0
        box = [100.0, 100.0 + shift, 120.0, 120.0 + shift]
        gt.append([(0, box)])
        dets.append([] if HOLE[0] <= index < HOLE[1] else [box])
    return dets, gt


def solve():
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:                  # pragma: no cover - T0 needs numpy and scipy
        raise practice.Skip(f"needs numpy/scipy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    drive = practice.load_module(SIBLING).drive
    fixture = {n: ref.synthetic_frames(num_frames=25, num_objects=n, seed=0) for n in COUNTS}
    runs = {n: drive(ref, fixture[n][0])[0] for n in COUNTS}
    dets, gt = detour()
    holed = drive(ref, dets, max_age=5)[0]
    return {"stack": missing(), "hole": dict(rate(linear_sum_assignment, ref, np, holed, gt),
                                             ids=len({t for f in holed for t, _ in f})),
            "score": {n: rate(linear_sum_assignment, ref, np, runs[n], fixture[n][1])
                      for n in COUNTS},
            "inflated": {k: rate(linear_sum_assignment, ref, np, copies(runs[3], k), fixture[3][1])
                         for k in COPIES}}


def verify(result):
    stack, score, inflated, hole = (result["stack"], result["score"],
                                    result["inflated"], result["hole"])
    curve, blind, cross = pull(inflated, COPIES, "mota"), pull(inflated, COPIES, "lesson"), score[10]
    return [
        practice.Check(
            "ANSWER: the second backend does not exist, so the comparison has one arm",
            list(stack.values()) == ["ModuleNotFoundError"] * len(STACK),
            f"importing the exercise's stack gives {stack}, and the lesson ships no 30-second crowd clip "
            f"and no hand-labelled identities either. The one backend that runs scores ID switches "
            f"{pull(score, COUNTS, 'lesson')} at {list(COUNTS)} objects, for MOTA "
            f"{round3(pull(score, COUNTS, 'mota'))} -- what follows measures that metric instead"),
        practice.Check(
            "FINDING: ID-switch count, the quantity the exercise compares on, ranks nothing",
            blind == [0] * len(COPIES) and curve[-1] < -5,
            f"emitting k identically-placed copies of every hypothesis leaves `count_id_switches` at "
            f"{blind} for k={list(COPIES)} -- 0 for a tracker outputting nine boxes per object, the "
            f"score a flawless run gets. MOTA on the same output reads {round3(curve)}"),
        practice.Check(
            "MECHANISM: MOTA is unbounded below -- it is 2 - k here, exactly",
            curve == linear(),
            f"MOTA = 1 - (FN + FP + IDSW)/GT and FP has no ceiling: on the 3-object fixture "
            f"({score[3]['gt']} ground-truth boxes) k copies give FP {pull(inflated, COPIES, 'fp')} at "
            f"FN and IDSW 0, so MOTA is {round3(curve)} = 2 - k for every k tried. At k={COPIES[-1]} "
            f"that is an 'accuracy' of {curve[-1]:.0%} -- what a sum of unbounded errors over a fixed "
            "count does"),
        practice.Check(
            "CONTROL: the lesson's own worst case stays positive, so the bound has to be forced",
            0.2 < score[30]["mota"] < 0.3 and score[30]["fp"] < score[30]["gt"],
            f"at 30 objects the tracker scores FN {score[30]['fn']} + FP {score[30]['fp']} + IDSW "
            f"{score[30]['idsw']} over {score[30]['gt']} ground-truth boxes, MOTA "
            f"{score[30]['mota']:.3f} -- degraded but positive. Below zero needs the three error terms "
            "to outnumber GT, which the fixture alone never does"),
        practice.Check(
            "FINDING: an ID switch and a fragmentation are different events",
            cross["idsw"] == 2 and cross["frag"] == 0 and hole["frag"] == 1 and hole["idsw"] == 0,
            f"the crossing at 10 objects scores IDSW {cross['idsw']}, FRAG {cross['frag']}, FN "
            f"{cross['fn']}, MOTA {cross['mota']:.3f}; an object stepping {SIDE:.0f}px aside for the "
            f"{HOLE[1] - HOLE[0]} frames its detector misses it scores IDSW {hole['idsw']}, FRAG "
            f"{hole['frag']}, FN {hole['fn']}, MOTA {hole['mota']:.3f} under {hole['ids']} track ID -- "
            "MOTA charges the misses and carries no fragmentation term"),
        practice.Check(
            "CONTROL: on the exercise's metric the fragmented run is indistinguishable from perfect",
            hole["lesson"] == score[3]["lesson"] == 0,
            f"`count_id_switches` returns {hole['lesson']} on the fragmented run -- what the clean "
            f"3-object run gets ({score[3]['lesson']}, MOTA {score[3]['mota']:.3f}) -- because the track "
            f"keeps its ID across the hole. Two backends compared on ID switches would tie while one "
            f"lost its object for {HOLE[1] - HOLE[0]} of {SPAN} frames"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
