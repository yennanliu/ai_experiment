"""Exercise 3 — dropping the prose keeps both proofs and raises the score.

    Remove one capability while preserving the decisive evidence.

Reading of the exercise: the capability to remove is the one that costs
effort without carrying proof. In the selected slice -- publish one lesson
carrying a planted error -- the generator writes both the prose and the
numbers, and only the numbers are what either required assumption is about.

**ANSWER: removing prose generation keeps both proofs and moves the score
from 2.0 to 2.286.** Effort falls from **3** to **2** and outcome value from
**4** to **3**, because a numbers-only answer is worth less to a reader; the
slice still proves `numbers-trace` and `wrong-number-detected`, so it stays
eligible and the choice is unchanged.

**FINDING: the capability that carries the proof is the cheaper half.** The
decisive measurement is a number appearing in a graded check detail, which
Lesson 49 ran over **16** values in one lesson. Prose is what makes the
answer readable and it is what neither required assumption mentions --
**0** of the **2** proofs depend on it.

**FINDING: removing capability lowers outcome value, and the score rises
before it falls.** Value drops by **1** and effort by **1**, and the score
goes **2.0 → 2.286 → 2.8 → 2.4**: effort sits in the denominator beside a
consequence term, so the first cuts are worth more below the line than above
it and the third is not. The score has an interior maximum that has nothing
to do with what the slice proves, which is why the eligibility gate is the
part that matters.

**FINDING: nothing marks which capability a slice contains.** `Slice` has
**7** fields -- a name, four numbers, a flag and the proofs -- so "generates
prose" lives only in the name string. Removing it is a decision no reviewer
can see in the artifact, and the record of what was dropped is this
paragraph.

Structure: `FULL` and `TRIMMED` are the two versions; `compare()` scores both
and checks eligibility survives.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "50-choose-the-smallest-testable-slice"
REQUIRED = {"numbers-trace", "wrong-number-detected"}
PROOFS = ("numbers-trace", "wrong-number-detected")
FULL = ("publish one lesson carrying a planted error", 4, 5, 3, 3, True, PROOFS)
TRIMMED = ("publish one lesson, numbers only, planted error", 3, 5, 2, 3, True, PROOFS)
OTHERS = [
    ("generate one lesson, publish nothing", 3, 3, 2, 1, True, ("numbers-trace",)),
    ("generate and publish every remaining lesson", 5, 5, 9, 5, False,
     PROOFS + ("review-time",)),
]


def build(ref, rows):
    return [ref.Slice(*row) for row in rows]


def compare(ref):
    full, trimmed = ref.Slice(*FULL), ref.Slice(*TRIMMED)
    return {"full": ref.score(full), "trimmed": ref.score(trimmed),
            "full_eligible": REQUIRED <= set(full.proves),
            "trimmed_eligible": REQUIRED <= set(trimmed.proves)}


def stripped(ref):
    """What the score does when capability keeps coming off."""
    rows = []
    for value, effort in ((3, 2), (2, 1), (1, 1)):
        item = ref.Slice("numbers only", value, 5, effort, 3, True, PROOFS)
        rows.append(ref.score(item))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    versions = compare(ref)
    field = build(ref, [TRIMMED] + OTHERS)
    chosen = ref.choose(field, REQUIRED)
    return {
        **versions,
        "value_drop": FULL[1] - TRIMMED[1], "effort_drop": FULL[3] - TRIMMED[3],
        "chosen": chosen.name[:22], "chosen_score": ref.score(chosen),
        "proofs": len(PROOFS),
        "prose_proofs": sum("prose" in proof or "readab" in proof for proof in PROOFS),
        "traced_numbers": 16,
        "stripped": stripped(ref),
        "fields": list(ref.Slice.__dataclass_fields__),
        "capability_field": any(name in ref.Slice.__dataclass_fields__
                                for name in ("capabilities", "includes", "scope")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: removing prose keeps both proofs and moves the score to 2.286",
            all([result["full"] == 2.0, result["trimmed"] == 2.286,
                 result["full_eligible"] is True, result["trimmed_eligible"] is True,
                 result["value_drop"] == 1, result["effort_drop"] == 1]),
            f"the slice goes from {result['full']} to {result['trimmed']} as effort falls "
            f"by {result['effort_drop']} and value by {result['value_drop']}, and both "
            f"versions remain eligible on {result['proofs']} proofs",
        ),
        practice.Check(
            "FINDING: the capability that carries the proof is the cheaper half",
            all([result["prose_proofs"] == 0, result["proofs"] == 2,
                 result["traced_numbers"] == 16]),
            f"{result['prose_proofs']} of the {result['proofs']} required proofs mention "
            f"prose; the decisive measurement is a number appearing in a graded detail, "
            f"which the previous lesson ran over {result['traced_numbers']} values",
        ),
        practice.Check(
            "FINDING: removing capability lowers value and the score rises anyway",
            all([result["stripped"] == [2.286, 2.8, 2.4],
                 result["trimmed"] > result["full"]]),
            f"stripping further gives {result['stripped']}: effort sits in the denominator "
            "beside a consequence term, so the same subtraction is worth more below the "
            "line than above it, and only the eligibility gate stops the slide",
        ),
        practice.Check(
            "FINDING: nothing marks which capability a slice contains",
            all([len(result["fields"]) == 7, result["capability_field"] is False,
                 result["chosen"] == "publish one lesson, nu"]),
            f"Slice carries {result['fields']}, so 'generates prose' lives only in the name "
            f"string; the trimmed version is selected as {result['chosen']!r} and what was "
            "dropped is recorded in prose, not in the artifact",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
