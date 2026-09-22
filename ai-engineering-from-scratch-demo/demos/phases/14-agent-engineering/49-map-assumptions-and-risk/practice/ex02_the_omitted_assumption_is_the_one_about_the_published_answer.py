"""Exercise 2 — the omitted assumption is the one about the published answer.

    Add one safety assumption that your feature list omitted.

Reading of the exercise: the five assumptions written for the generator all
ask whether it works. None asks what happens when it is wrong *and the answer
is already in a README somebody has read. That is the safety class, and it is
the one the feature list left out.

**ANSWER: "a wrong number reaches a reader before anyone notices" scores 25
and takes the top of the map.** Impact **5**, uncertainty **4**,
irreversibility **5**: a published claim cannot be unpublished. Added to the
five, it lifts the top score from **17** to **25** and changes the next
experiment from a timing study to a detection test.

**FINDING: the safety assumption is the only one whose test is a decision.**
Five of the six tests name something to observe -- time a review, sort six
answers, match numbers, replay last month, plant three wrong numbers. The
lesson's own safety row does the same thing in reverse: "do not automate;
test approval workflow first" is a choice about the build, not a measurement.
**1** of **6** tests here names an action the team takes instead of a result
it reads.

**FINDING: irreversibility 5 only wins when impact and uncertainty are high
too.** With the formula capped as it is, a safety assumption at
irreversibility **5** needs a product of at least **13** to beat a value
assumption at **17**. The version of this assumption with a modest impact of
3 scores **14** and ranks **fifth** of six; the version that describes a published
mistake scores **25** and ranks first. The class does not earn the ranking --
the numbers do.

**FINDING: adding it leaves the tie underneath untouched.** The two
assumptions that tied at **17** still tie, and `prioritize` and
`next_experiment` still order them differently. A new top item hides the
disagreement rather than resolving it, which is what makes the tie worth
recording.

Structure: `SAFETY` is the omission; `with_safety()` re-ranks the map.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "49-map-assumptions-and-risk"

BASE = [
    ("A generated answers section saves more time than reviewing it costs", 4, 4, 1,
     "Generate one lesson's answers and time the review against writing them"),
    ("A reader cannot tell a generated answer from a written one", 3, 4, 1,
     "Show five readers three answers of each kind and ask them to sort"),
    ("Every number in an answer can be sourced from a graded check detail", 5, 3, 1,
     "Extract the numbers from one finished lesson and match them against its details"),
    ("The generator stays correct as the harness changes", 3, 5, 2,
     "Re-run last month's generated answers against the current harness"),
    ("A wrong generated number is caught before the lesson ships", 3, 3, 5,
     "Plant three wrong numbers and see whether the audit refuses them"),
]
SAFETY = ("A wrong number reaches a reader before anyone notices", 5, 4, 5,
          "Publish one generated lesson and measure how long a planted error survives")
DECISION_TEST = "Do not automate; test approval workflow first"


def build(ref, rows):
    return [ref.Assumption(*row) for row in rows]


def with_safety(ref):
    return build(ref, BASE + [SAFETY])


def observation_count(rows):
    """Tests that name something to read rather than a choice about the build."""
    return sum(row[4][0].isupper() and not row[4].startswith("Do not") for row in rows)


def top(ref, items):
    ranked = ref.prioritize(items)
    return ranked[0]["risk_score"], ranked[0]["statement"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before, after = build(ref, BASE), with_safety(ref)
    ranked = ref.prioritize(after)
    scores = [ref.risk_score(item) for item in after]
    observational = observation_count(BASE + [SAFETY])
    lesson_tests = [item.test for item in ref.example()]
    modest = ref.Assumption(SAFETY[0], 3, 3, 5, SAFETY[4])
    return {
        "before_top": top(ref, before)[0], "after_top": top(ref, after)[0],
        "after_statement": top(ref, after)[1],
        "before_next": ref.next_experiment(before).statement[:24],
        "after_next": ref.next_experiment(after).statement[:24],
        "count": len(after), "scores": scores,
        "observations": observational,
        "decision_tests": sum(test == DECISION_TEST for test in lesson_tests),
        "modest_score": ref.risk_score(modest),
        "modest_rank": sorted(scores + [ref.risk_score(modest)], reverse=True).index(
            ref.risk_score(modest)) + 1,
        "safety_score": ref.risk_score(ref.Assumption(*SAFETY)),
        "needed_product": top(ref, before)[0] - 5 + 1,
        "tied": [row["risk_score"] for row in ranked].count(17),
        "prioritize_first_17": next(row["statement"] for row in ranked
                                    if row["risk_score"] == 17)[:24],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the omitted assumption scores 25 and takes the top of the map",
            all([result["before_top"] == 17, result["after_top"] == 25,
                 result["safety_score"] == 25, result["count"] == 6,
                 result["after_statement"] == SAFETY[0],
                 result["before_next"] != result["after_next"]]),
            f"the top score goes from {result['before_top']} to {result['after_top']} and "
            f"the next experiment changes from {result['before_next']!r} to "
            f"{result['after_next']!r}: a published claim cannot be unpublished, which is "
            "impact 5, uncertainty 4 and irreversibility 5",
        ),
        practice.Check(
            "FINDING: the safety assumption is the only one whose test is a decision",
            all([result["observations"] == 6, result["decision_tests"] == 1]),
            f"{result['observations']} of the six tests written here name something to "
            f"observe, while the lesson's own safety row carries {result['decision_tests']} "
            "test that is a choice about the build rather than a result to read",
        ),
        practice.Check(
            "FINDING: irreversibility 5 only wins when impact and uncertainty are high too",
            all([result["modest_score"] == 14, result["modest_rank"] == 5,
                 result["needed_product"] == 13, result["safety_score"] == 25]),
            f"a safety assumption at irreversibility 5 needs a product of at least "
            f"{result['needed_product']} to beat a value assumption at "
            f"{result['before_top']}; the modest version scores {result['modest_score']} "
            f"and ranks {result['modest_rank']}, the published-mistake version scores "
            f"{result['safety_score']}",
        ),
        practice.Check(
            "FINDING: adding it leaves the tie underneath untouched",
            all([result["tied"] == 2,
                 result["prioritize_first_17"] == "A generated answers sect"]),
            f"{result['tied']} assumptions still tie at 17 beneath the new top item, and "
            f"prioritize still names {result['prioritize_first_17']!r} first while "
            "next_experiment would reach the other; a new top hides the disagreement rather "
            "than resolving it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
