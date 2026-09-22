"""Exercise 5 — the platform slice stays ineligible at any score.

    Identify a reusable platform component that should wait until after the
    slice.

Reading of the exercise: the component is easy to name -- a general answer
generator for every phase rather than one lesson -- and the useful part is
showing that no arithmetic rescues it. The lesson's own build instruction
asks for exactly this test.

**ANSWER: the platform slice proves 0 of the 2 required assumptions and stays
unselected even when its score is the highest in the field.** Given plausible
numbers it scores **0.231**; inflated to value 9, uncertainty 9 and effort 1
it scores **18.0**, the best of **4** candidates -- and `choose` still
returns the one-lesson pilot, because the subset gate runs before the
comparison.

**FINDING: the platform is the only candidate whose proofs are empty, and
emptiness is not what disqualifies it.** A platform proving one required
assumption would be rejected just the same; the gate is
`required_proof <= set(item.proves)`, so **1** of **2** fails exactly like
**0** of **2**. Partial coverage buys nothing, which is what makes the proof
set worth arguing about before the slices are drawn.

**FINDING: the lesson names 5 false minimums and this field contains 3 of
them.** The platform minimum is the reusable framework; the UI-only minimum
is the unpublished generation run, which removes the publishing uncertainty;
the happy-path minimum is any version without the planted error. The two
that are absent -- infrastructure-only and demo -- are absent because the
outcome already names a reader.

**FINDING: the artifact records the rejection without the reason.**
`decision` returns the selected slice and the alternatives, each with a
score, so a reader sees the platform scoring **18.0** and losing with no
field saying which proof it lacked. The eligibility test is the most
important thing the document does not record.

Structure: `FIELD` is the candidate set; `inflate()` gives the platform the
best score in it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "50-choose-the-smallest-testable-slice"
REQUIRED = {"numbers-trace", "wrong-number-detected"}
PROOFS = ("numbers-trace", "wrong-number-detected")
PLATFORM = ("a general answer generator for every phase", 2, 1, 12, 2, True, ())
FIELD = [
    ("generate one lesson, publish nothing", 3, 3, 2, 1, True, ("numbers-trace",)),
    ("publish one lesson, numbers only, planted error", 3, 5, 2, 3, True, PROOFS),
    ("generate and publish every remaining lesson", 5, 5, 9, 5, False,
     PROOFS + ("review-time",)),
    PLATFORM,
]
FALSE_MINIMUMS = ["UI-only", "infrastructure-only", "happy-path", "demo", "platform"]
PRESENT = ["UI-only", "happy-path", "platform"]


def build(ref, rows=FIELD):
    return [ref.Slice(*row) for row in rows]


def inflate(ref):
    """The platform with the best score in the field and none of the proofs."""
    return ref.Slice(PLATFORM[0], 9, 9, 1, 0, True, ())


def partial(ref):
    """A platform that proves half of what is required."""
    return ref.Slice("half-covering platform", 9, 9, 1, 0, True, ("numbers-trace",))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    items = build(ref)
    boosted = build(ref, FIELD[:3]) + [inflate(ref)]
    chosen = ref.choose(items, REQUIRED)
    chosen_boosted = ref.choose(boosted, REQUIRED)
    document = ref.decision(boosted, REQUIRED)
    scores = {item.name[:18]: ref.score(item) for item in boosted}
    return {
        "platform_proves": len(PLATFORM[6]),
        "required": len(REQUIRED),
        "platform_score": ref.score(ref.Slice(*PLATFORM)),
        "boosted_score": ref.score(inflate(ref)),
        "best": max(scores.values()), "candidates": len(boosted),
        "chosen": chosen.name[:22], "chosen_boosted": chosen_boosted.name[:22],
        "partial_eligible": REQUIRED <= set(partial(ref).proves),
        "empty_eligible": REQUIRED <= set(inflate(ref).proves),
        "false_minimums": len(FALSE_MINIMUMS), "present": PRESENT,
        "alternatives": len(document["alternatives"]),
        "alternative_keys": sorted(document["alternatives"][0]),
        "records_gap": any("proof" in key and key != "proves"
                           for key in document["alternatives"][0]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the platform proves 0 of 2 and loses with the best score in the field",
            all([result["platform_proves"] == 0, result["required"] == 2,
                 result["platform_score"] == 0.231, result["boosted_score"] == 18.0,
                 result["best"] == 18.0, result["candidates"] == 4,
                 result["chosen_boosted"] == "publish one lesson, nu"]),
            f"the platform scores {result['platform_score']} with plausible numbers and "
            f"{result['boosted_score']} when inflated -- the best of "
            f"{result['candidates']} candidates -- and choose still returns "
            f"{result['chosen_boosted']!r}, because the subset gate runs first",
        ),
        practice.Check(
            "FINDING: emptiness is not what disqualifies it",
            all([result["partial_eligible"] is False,
                 result["empty_eligible"] is False]),
            "a platform proving one of the two required assumptions is rejected exactly "
            "like one proving none: the gate is a subset test, so partial coverage buys "
            "nothing and the proof set is what deserves the argument",
        ),
        practice.Check(
            "FINDING: the lesson names 5 false minimums and this field contains 3",
            all([result["false_minimums"] == 5, len(result["present"]) == 3,
                 "platform" in result["present"]]),
            f"{len(result['present'])} of the {result['false_minimums']} named shapes are "
            f"in this field ({result['present']}); infrastructure-only and demo are absent "
            "because the outcome already names a reader",
        ),
        practice.Check(
            "FINDING: the artifact records the rejection without the reason",
            all([result["alternatives"] == 3, result["records_gap"] is False,
                 "score" in result["alternative_keys"]]),
            f"decision returns {result['alternatives']} alternatives carrying "
            f"{result['alternative_keys']}, so a reader sees the platform scoring "
            f"{result['best']} and losing with no field naming the proof it lacked",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
