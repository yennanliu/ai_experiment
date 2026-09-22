"""Exercise 4 — the next question asks for confirmation, and the frame has no field for doubt.

    Identify the earliest observation that would disprove the desired outcome.

Reading of the exercise: "disprove" and "show that it was achieved" are
different questions, and the lesson's artifact asks only the second one. The
earliest disproof is worth naming because it is usually much cheaper than the
proof.

**ANSWER: one graded answer failing against its lesson's code disproves the
outcome, and it costs 1 file where confirmation costs 15.** The outcome is
"a reader can run an answer and see it check its own claims"; a single answer
whose checks fail against the reference module falsifies it. Graded now, the
three most recent lessons return **15** passes and **0** failures, so the
observation has been looked for and not found -- which is the only form a
passing disproof takes.

**FINDING: `next_question` is confirmation-seeking by construction.** It reads
"What evidence would show that the desired outcome was achieved for the
{user}?" -- **1** templated question, asking for evidence *for*. `decision`
returns **4** keys and none of them is a falsifier, so the artifact that
travels to the next lesson carries the confirming question only.

**FINDING: the disproof has to name a file, not a feeling.** A failing check
reports its own detail line, so the observation is "this file, this check,
this measured value" rather than "readers seem confused". Across the
**15** answers there are **60** individual checks; the disproof is any one of
them turning red, which makes it **60** chances to be wrong rather than one
impression to argue about.

**FINDING: the cheap disproof and the expensive one disagree about when to
look.** Grading one file is **6.7%** of the work of grading fifteen, so the
earliest observation is available after the first lesson rather than at the
end of the phase. A frame that records only the confirming question invites
the opposite schedule.

Structure: `disproof()` runs the earliest observation; `proof()` runs the full
one, so the two costs can be compared.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "47-outcomes-before-output"
BASE = Path(__file__).resolve().parents[2]
LESSONS = ("43-frame-the-task-before-code", "44-plan-from-evidence",
           "45-delegate-with-isolation")


def answers():
    return sorted(path for lesson in LESSONS
                  for path in (BASE / lesson / "practice").glob("ex0*.py"))


def disproof(paths):
    """The earliest observation: the first graded answer that fails."""
    for path in paths:
        result = practice.grade_file(path)
        if result.status != "pass":
            return {"found": True, "file": path.name, "graded": 1}
    return {"found": False, "file": "", "graded": 1}


def proof(paths):
    """The confirming observation: every answer, every check."""
    results = [practice.grade_file(path) for path in paths]
    return {"files": len(results), "passed": sum(r.status == "pass" for r in results),
            "checks": sum(len(r.checks) for r in results),
            "failed": sum(1 for r in results for check in r.checks if not check.ok)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    paths = answers()
    early = disproof(paths[:1])
    full = proof(paths)
    frame = ref.OutcomeFrame(
        user="a reader working through a phase 14 lesson",
        situation="facing the exercises after reading the lesson",
        current_behavior="guesses at an answer with no way to check it",
        desired_outcome="a reader can run an answer and see it check its own claims",
        constraints=["answers import the lesson's code"],
        non_goals=["rewriting the lesson"], proposed_output="")
    document = ref.decision(frame)
    question = document["next_question"]
    return {
        "early_graded": early["graded"], "early_found": early["found"],
        "files": full["files"], "passed": full["passed"], "checks": full["checks"],
        "failed": full["failed"],
        "cost": round(early["graded"] / full["files"], 3),
        "question": question,
        "confirming": "would show that the desired outcome was achieved" in question,
        "keys": sorted(document),
        "falsifier_field": any(name in ref.OutcomeFrame.__dataclass_fields__
                               for name in ("falsifier", "disproof", "kill_criterion")),
        "templates": inspect.getsource(ref.decision).count("next_question"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one failing answer disproves it, at 1 file against 15",
            all([result["early_graded"] == 1, result["early_found"] is False,
                 result["files"] == 15, result["passed"] == 15, result["failed"] == 0]),
            f"the earliest observation grades {result['early_graded']} file and found "
            f"{'a failure' if result['early_found'] else 'none'}; confirmation grades "
            f"{result['files']} and returns {result['passed']} passes with "
            f"{result['failed']} failed checks",
        ),
        practice.Check(
            "FINDING: next_question is confirmation-seeking by construction",
            all([result["confirming"] is True, result["templates"] == 1,
                 result["keys"] == ["frame", "issues", "next_question", "status"],
                 result["falsifier_field"] is False]),
            f"the one templated question is {result['question']!r}; decision returns "
            f"{result['keys']} and the frame has no falsifier field, so the artifact that "
            "travels to the next lesson carries the confirming question only",
        ),
        practice.Check(
            "FINDING: the disproof has to name a file, not a feeling",
            all([result["checks"] == 60, result["files"] == 15]),
            f"the {result['files']} answers hold {result['checks']} individual checks, each "
            "reporting its own measured detail, so the observation is 'this file, this "
            "check, this value' -- sixty chances to be wrong rather than one impression",
        ),
        practice.Check(
            "FINDING: the cheap disproof and the expensive one disagree about when to look",
            all([result["cost"] == 0.067, result["files"] == 15]),
            f"grading one file is {result['cost']:.1%} of grading fifteen, so the earliest "
            "observation is available after the first lesson rather than at the end of the "
            "phase; a frame recording only the confirming question invites the opposite",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
