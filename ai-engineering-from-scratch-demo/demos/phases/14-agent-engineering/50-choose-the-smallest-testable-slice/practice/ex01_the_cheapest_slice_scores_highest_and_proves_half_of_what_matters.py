"""Exercise 1 — the cheapest slice scores highest and proves half of what matters.

    Design three slices for the same outcome at different consequence levels.

Reading of the exercise: three slices for one outcome, differing in what they
put at risk. The outcome here is the one Lesson 49 mapped -- generating a
lesson's answers section instead of writing it -- and the consequence ladder
runs from a file nobody sees to a published page a reader may rely on.

**ANSWER: the three score 2.4, 2.0 and 0.526, and the cheapest one is the one
that proves least.** Generating a lesson without publishing it costs effort
**2** at consequence **1** and scores **2.4**; publishing one lesson with a
planted error costs **3** at consequence **3** and scores **2.0**; generating
and publishing everything remaining costs **9** at consequence **5**,
irreversible, and scores **0.526**. The ranking by score puts the weakest
evidence first.

**FINDING: the irreversible flag is a 4x swing on the same consequence.**
`score` multiplies consequence by **2** when a slice is irreversible and by
**0.5** when it is not, so the third slice's consequence of 5 contributes
**10** to its denominator where a reversible version would contribute
**2.5**. Marking it reversible lifts its score from **0.526** to **0.87**
without changing anything about the work.

**FINDING: consequence and effort are added, so a cheap catastrophe looks
like an expensive nuisance.** A one-day change that publishes a wrong number
to every reader (effort 1, consequence 5, irreversible) scores the same
denominator -- **11** -- as a nine-day change with no consequence at all
(effort 11, consequence 0). The formula the lesson calls "intentionally
simple" is simple in a direction that flatters the risky option.

**FINDING: nothing in the slice records which assumption each proof
discharges.** `Slice.proves` is a tuple of **2**, **2** and **3** strings and
`Assumption` in the previous lesson had no identifier, so the two artifacts
join by convention only. A misspelled proof name makes a slice silently
ineligible.

Structure: `SLICES` is the ladder; `scored()` runs the lesson's own scoring.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "50-choose-the-smallest-testable-slice"

# (name, outcome_value, uncertainty_reduced, effort, consequence, reversible, proves)
SLICES = [
    ("generate one lesson, publish nothing", 3, 3, 2, 1, True, ("numbers-trace",)),
    ("publish one lesson carrying a planted error", 4, 5, 3, 3, True,
     ("numbers-trace", "wrong-number-detected")),
    ("generate and publish every remaining lesson", 5, 5, 9, 5, False,
     ("numbers-trace", "wrong-number-detected", "review-time")),
]


def build(ref, rows=SLICES):
    return [ref.Slice(*row) for row in rows]


def scored(ref, items):
    return [(item.name[:22], ref.score(item)) for item in items]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    items = build(ref)
    scores = [ref.score(item) for item in items]
    last = SLICES[-1]
    reversible = ref.Slice(*last[:5], True, last[6])
    cheap_disaster = ref.Slice("one-day publish of a wrong number", 5, 5, 1, 5, False, ())
    slow_and_safe = ref.Slice("eleven days, no consequence", 5, 5, 11, 0, True, ())
    return {
        "slices": len(items), "scores": scores,
        "ranked": [name for name, _ in sorted(scored(ref, items), key=lambda row: -row[1])],
        "cheapest_proves": len(SLICES[0][6]),
        "most_proves": len(SLICES[-1][6]),
        "penalty": inspect.getsource(ref.score).count("2 if not item.reversible else 0.5"),
        "irreversible_score": ref.score(items[-1]),
        "reversible_score": ref.score(reversible),
        "denominators": (items[-1].effort + items[-1].consequence * 2,
                         items[-1].effort + items[-1].consequence * 0.5),
        "disaster_denominator": cheap_disaster.effort + cheap_disaster.consequence * 2,
        "safe_denominator": slow_and_safe.effort + slow_and_safe.consequence * 0.5,
        "proof_counts": [len(row[6]) for row in SLICES],
        "proof_type": ref.Slice.__annotations__["proves"],
        "misspelled_eligible": bool({"numbers-trace"} <= set(("numbers_trace",))),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the three score 2.4, 2.0 and 0.526, cheapest first",
            all([result["slices"] == 3, result["scores"] == [2.4, 2.0, 0.526],
                 result["cheapest_proves"] == 1, result["most_proves"] == 3,
                 result["ranked"][0] == "generate one lesson, p"]),
            f"the ladder scores {result['scores']} and ranks "
            f"{result['ranked'][0]!r} first, a slice proving "
            f"{result['cheapest_proves']} assumption against the "
            f"{result['most_proves']} the largest one proves",
        ),
        practice.Check(
            "FINDING: the irreversible flag is a 4x swing on the same consequence",
            all([result["penalty"] == 1, result["irreversible_score"] == 0.526,
                 result["reversible_score"] == 0.87,
                 result["denominators"] == (19, 11.5)]),
            f"consequence 5 contributes {result['denominators'][0] - 9} to the denominator "
            f"when irreversible and {result['denominators'][1] - 9} when not, so flipping "
            f"the flag lifts the score from {result['irreversible_score']} to "
            f"{result['reversible_score']} with no change to the work",
        ),
        practice.Check(
            "FINDING: consequence and effort are added, so a cheap catastrophe looks cheap",
            all([result["disaster_denominator"] == 11, result["safe_denominator"] == 11]),
            f"a one-day change that publishes a wrong number to every reader and an "
            f"eleven-day change with no consequence share a denominator of "
            f"{result['disaster_denominator']}: the formula is simple in a direction that "
            "flatters the risky option",
        ),
        practice.Check(
            "FINDING: nothing records which assumption each proof discharges",
            all([result["proof_counts"] == [1, 2, 3],
                 result["proof_type"] == "tuple[str, ...]",
                 result["misspelled_eligible"] is False]),
            f"proves is a {result['proof_type']} holding {result['proof_counts']} strings, "
            "joined to the assumption map by convention only -- an underscore instead of a "
            "hyphen makes a slice silently ineligible",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
