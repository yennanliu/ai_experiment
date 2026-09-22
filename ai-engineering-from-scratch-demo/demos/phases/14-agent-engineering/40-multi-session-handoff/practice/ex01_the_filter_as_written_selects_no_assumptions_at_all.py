"""Exercise 1 — the filter as written selects no assumptions at all.

    Add an `assumptions_to_validate` field that surfaces every assumption the
    builder logged but the reviewer did not score above 1.

Reading of the exercise: the filter joins two artifacts -- the builder's
assumption list in `state` and the reviewer's score for the `assumptions`
dimension. Before writing it, check that the join can select anything.

**ANSWER: the field is empty for every possible input.** Lesson 39's
`score_assumptions` returns 2 whenever the list is non-empty and 1 only when
it is empty, so "assumptions the reviewer did not score above 1" is the empty
list when there is nothing to list and the empty list again when there is.
Swept over **0** to **5** logged assumptions the field has length 0 in **6**
of **6** cases.

**FINDING: the reviewer scores the dimension, never the assumption.**
`DimensionScore` has **3** fields -- name, score, note -- and the note is a
count: "3 assumptions recorded". There is no per-item verdict anywhere in the
review report, so no filter over it can name *which* assumption is weak. The
exercise asks for a join on a key the reviewer does not emit.

**FINDING: the packet reads 2 keys of the review report.** The module touches
`review["verdict"]` in `generate_handoff` and `review["total"]` in
`derive_risks`, and `WorkbenchSnapshot.review`
is typed `dict[str, object]`, so even the per-dimension list that Lesson 39
*does* write to `review_report.json` never reaches the packet. Widening the
field means widening the read, not just the payload.

**FINDING: the useful version validates against evidence, not against a
score.** Mark an assumption unvalidated when no command that ran and no file
that changed mentions any word in it: on a fixture of **3** assumptions the
filter selects **2**, leaving out the one the test run actually exercised.
That is a field with content, and it needs nothing from the reviewer.

Structure: `sweep()` runs the exercise's filter over assumption lists of every
size; `unvalidated()` is the evidence-based replacement.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "40-multi-session-handoff"
REVIEWER = "39-reviewer-agent"
STOP = {"the", "and", "with", "from", "only", "into", "that", "this", "must", "have"}
ASSUMPTIONS = ["signup accepts email and password only",
               "rate limit window stays at 60 seconds",
               "legacy oauth callers are out of scope"]


def as_written(rev, assumptions):
    """The exercise's filter: surface assumptions the reviewer scored at or below 1."""
    dimension = rev.score_assumptions(rev.ReviewerInputs(
        task_id="T-001", goal="add input validation to signup", diff_summary={"touched": []},
        state={"assumptions": assumptions}, feedback=[], verdict={}))
    return (list(assumptions) if dimension.score <= 1 else []), dimension.score


def sweep(rev):
    return [as_written(rev, [f"assumption {i}" for i in range(n)]) for n in range(6)]


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOP}


def unvalidated(assumptions, commands, files):
    """An assumption is unvalidated when nothing that ran or changed mentions it."""
    evidence = words(" ".join(commands) + " " + " ".join(files))
    return [a for a in assumptions if not (words(a) & evidence)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rev = parity.load_reference(PHASE, REVIEWER, "main")
    rows = sweep(rev)
    module = inspect.getsource(ref)
    commands = ["pytest tests/test_signup_email_password.py", "ruff check ."]
    files = ["app/signup.py", "tests/test_signup_email_password.py"]
    return {
        "sizes": [len(selected) for selected, _ in rows],
        "scores": [score for _, score in rows],
        "cases": len(rows),
        "dimension_fields": list(rev.DimensionScore.__dataclass_fields__),
        "note": rev.score_assumptions(rev.ReviewerInputs(
            "T", "g", {"touched": []}, {"assumptions": ASSUMPTIONS}, [], {})).note,
        "review_keys": sorted(set(re.findall(r"review\.get\(['\"](\w+)", module))),
        "review_type": ref.WorkbenchSnapshot.__annotations__["review"],
        "payload_fields": len(ref.HandoffPayload.__dataclass_fields__),
        "unvalidated": unvalidated(ASSUMPTIONS, commands, files),
        "assumptions": len(ASSUMPTIONS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the field is empty for every possible input",
            all([result["sizes"] == [0] * 6, result["scores"] == [1, 2, 2, 2, 2, 2],
                 result["cases"] == 6]),
            f"swept over 0 to 5 logged assumptions the field has sizes {result['sizes']} "
            f"while the dimension scores {result['scores']}: score 1 happens exactly when "
            "the list is empty, so the filter selects nothing in every case",
        ),
        practice.Check(
            "FINDING: the reviewer scores the dimension, never the assumption",
            all([result["dimension_fields"] == ["name", "score", "note"],
                 result["note"] == "3 assumptions recorded"]),
            f"DimensionScore carries {result['dimension_fields']} and the note is a count "
            f"({result['note']!r}), so nothing in the review report names which assumption "
            "is weak. The exercise joins on a key the reviewer does not emit",
        ),
        practice.Check(
            "FINDING: the packet reads 2 keys of the review report",
            all([result["review_keys"] == ["total", "verdict"],
                 result["review_type"] == "dict[str, object]",
                 result["payload_fields"] == 9]),
            f"the module reads {result['review_keys']} of a review typed "
            f"{result['review_type']}, so the per-dimension list Lesson 39 writes never "
            f"reaches the {result['payload_fields']}-field payload. Widening the field "
            "means widening the read",
        ),
        practice.Check(
            "FINDING: the useful version validates against evidence",
            all([len(result["unvalidated"]) == 2, result["assumptions"] == 3,
                 "rate limit window stays at 60 seconds" in result["unvalidated"]]),
            f"marking an assumption unvalidated when no command that ran and no file that "
            f"changed mentions it selects {len(result['unvalidated'])} of "
            f"{result['assumptions']} -- {result['unvalidated']} -- and leaves out the one "
            "the test run exercised. A field with content, needing nothing from the reviewer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
