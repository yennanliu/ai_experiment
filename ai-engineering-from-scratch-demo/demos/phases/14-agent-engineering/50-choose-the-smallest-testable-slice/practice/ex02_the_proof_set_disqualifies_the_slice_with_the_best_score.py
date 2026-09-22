"""Exercise 2 — the proof set disqualifies the slice with the best score.

    State the required proof set before scoring them.

Reading of the exercise: "before" is the whole instruction. The proof set
comes from the open assumptions the previous lesson ranked highest, and
writing it first is what makes the highest-scoring slice refusable.

**ANSWER: the required proof is `{numbers-trace, wrong-number-detected}` and
it rejects the slice scoring 2.4 in favour of the one scoring 2.0.**
`choose` filters by `required_proof <= set(item.proves)` before it compares
anything, so the cheapest slice -- which proves only that numbers trace --
never reaches the arithmetic. **2** of the **3** candidates are eligible and
the selected one is not the highest scorer.

**FINDING: the gate is a subset test, so proving more is free and proving
differently is fatal.** The largest slice proves **3** assumptions including
one nobody asked for and stays eligible; a slice proving
`numbers-trace` and `wrong-number-detected-late` proves **2** and is
rejected. Eligibility is string equality on names that live in two artifacts
and are checked in neither.

**FINDING: an empty proof set makes every slice eligible.** `set() <= anything`
is true, so calling `choose` with no required proof returns the highest
scorer -- here the cheapest slice at **2.4** -- and the run looks identical
to a considered decision. The gate that the lesson says matters more than the
arithmetic disappears silently when nobody fills it in.

**FINDING: no eligible candidate raises rather than reports.** Requiring a
proof no slice offers makes `decision` raise `ValueError`, so the document
that exists to record the choice cannot record "nothing qualifies" -- the
same shape Lesson 45's planner had when a dependency went missing.

Structure: `REQUIRED` is the proof set; `eligibility()` runs the gate over
each candidate.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "50-choose-the-smallest-testable-slice"
REQUIRED = {"numbers-trace", "wrong-number-detected"}
SLICES = [
    ("generate one lesson, publish nothing", 3, 3, 2, 1, True, ("numbers-trace",)),
    ("publish one lesson carrying a planted error", 4, 5, 3, 3, True,
     ("numbers-trace", "wrong-number-detected")),
    ("generate and publish every remaining lesson", 5, 5, 9, 5, False,
     ("numbers-trace", "wrong-number-detected", "review-time")),
]


def build(ref, rows=SLICES):
    return [ref.Slice(*row) for row in rows]


def eligibility(ref, items, required=REQUIRED):
    return [{"name": item.name[:22], "score": ref.score(item),
             "eligible": required <= set(item.proves)} for item in items]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    items = build(ref)
    rows = eligibility(ref, items)
    selected = ref.choose(items, REQUIRED)
    near_miss = ref.Slice("publish with a late detector", 4, 5, 3, 3, True,
                          ("numbers-trace", "wrong-number-detected-late"))
    try:
        ref.decision(items, {"human-review-time"})
        raised = ""
    except ValueError as error:
        raised = str(error)
    return {
        "required": sorted(REQUIRED),
        "eligible": [row["name"] for row in rows if row["eligible"]],
        "rejected": [row["name"] for row in rows if not row["eligible"]],
        "scores": [row["score"] for row in rows],
        "selected": selected.name[:22], "selected_score": ref.score(selected),
        "best_score": max(row["score"] for row in rows),
        "gate_first": inspect.getsource(ref.choose).index("required_proof")
        < inspect.getsource(ref.choose).index("max("),
        "superset_eligible": REQUIRED <= set(SLICES[2][6]),
        "near_miss_eligible": REQUIRED <= set(near_miss.proves),
        "empty_eligible": [row["name"] for row in eligibility(ref, items, set())
                           if row["eligible"]],
        "empty_choice": ref.choose(items, set()).name[:22],
        "raised": raised,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the proof set rejects the 2.4 slice in favour of the 2.0 one",
            all([result["required"] == ["numbers-trace", "wrong-number-detected"],
                 len(result["eligible"]) == 2, result["selected_score"] == 2.0,
                 result["best_score"] == 2.4,
                 result["rejected"] == ["generate one lesson, p"],
                 result["gate_first"] is True]),
            f"the gate leaves {len(result['eligible'])} of 3 candidates and selects "
            f"{result['selected']!r} at {result['selected_score']}, while the best score in "
            f"the field is {result['best_score']} and belongs to {result['rejected']}",
        ),
        practice.Check(
            "FINDING: proving more is free and proving differently is fatal",
            all([result["superset_eligible"] is True,
                 result["near_miss_eligible"] is False]),
            "the largest slice proves an extra assumption nobody asked for and stays "
            "eligible, while a slice proving 'wrong-number-detected-late' is rejected: "
            "eligibility is string equality across two artifacts and checked in neither",
        ),
        practice.Check(
            "FINDING: an empty proof set makes every slice eligible",
            all([len(result["empty_eligible"]) == 3,
                 result["empty_choice"] == "generate one lesson, p"]),
            f"set() is a subset of everything, so an unfilled gate leaves "
            f"{len(result['empty_eligible'])} eligible candidates and returns "
            f"{result['empty_choice']!r}, the cheapest -- a run that looks identical to a "
            "considered decision",
        ),
        practice.Check(
            "FINDING: no eligible candidate raises rather than reports",
            result["raised"] == "no slice proves the required assumptions",
            f"requiring a proof no slice offers raises {result['raised']!r}, so the "
            "document that exists to record the choice cannot record 'nothing qualifies'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
