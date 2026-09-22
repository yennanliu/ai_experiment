"""Exercise 2 — the determinism constraint removes the solutions the lesson would reach for.

    Add one constraint that changes which solutions remain possible.

Reading of the exercise: a constraint earns its place by deleting options. So
the test is not whether it sounds important; it is how many candidate outputs
stop being available once it is written down.

**ANSWER: "the same command produces the same verdict on any machine" deletes
3 of 6 candidate outputs.** Timing a run, sampling a model and calling a
network service all stop being available; reading the lesson's code, comparing
against a fixture and counting artifacts survive. The **15** solutions the
three most recent lessons ship import `time`, `random` and `urllib`
**0** times between them, which is what the constraint looks like once it has
been obeyed for a while.

**FINDING: the constraint is a string the validator never opens.** `validate`
tests that `constraints` is non-empty and stops; a frame carrying **3**
constraints and one carrying `["be good"]` are equally valid. Whether a
candidate output violates a constraint is a judgment the artifact records and
the program cannot make.

**FINDING: the constraint is what makes the outcome checkable, not just
narrower.** Without it, "a reader can run an answer and see it check its own
claims" is satisfied by an answer that passes on the author's laptop. With it,
the same sentence has a test: run the **15** files twice and compare. They
agree **15** of **15** times, and that comparison is only meaningful because
the constraint exists.

**FINDING: a constraint that deletes nothing is a preference.** Adding "prefer
clear names" to the same frame leaves **6** of the **6** candidate outputs
available and the validator equally happy. The lesson's own list -- no
production writes, no new runtime dependency -- are all of the deleting kind.

Structure: `CANDIDATES` is the option set; `survivors()` applies a constraint;
`rerun()` measures the determinism the constraint asks for.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "47-outcomes-before-output"
BASE = Path(__file__).resolve().parents[2]
LESSONS = ("43-frame-the-task-before-code", "44-plan-from-evidence",
           "45-delegate-with-isolation")
DETERMINISM = "the same command produces the same verdict on any machine"
PREFERENCE = "prefer clear names"

# (candidate output, what it depends on at run time)
CANDIDATES = [
    ("time the run and report the speed-up", "wall clock"),
    ("ask a model to grade the answer", "sampling"),
    ("fetch the upstream file and diff it", "network"),
    ("import the lesson's code and assert on what it returns", "the repository"),
    ("compare output against a checked-in fixture", "the repository"),
    ("count the artifacts the lesson produces", "the repository"),
]
VOLATILE = {"wall clock", "sampling", "network"}


def survivors(constraint):
    if constraint != DETERMINISM:
        return [name for name, _ in CANDIDATES]
    return [name for name, source in CANDIDATES if source not in VOLATILE]


def shipped():
    return sorted(path for lesson in LESSONS
                  for path in (BASE / lesson / "practice").glob("ex0*.py"))


def imports(paths, modules=("import time", "import random", "import urllib")):
    return sum(any(marker in path.read_text(encoding="utf-8") for marker in modules)
               for path in paths)


def verdict(path):
    result = practice.grade_file(path)
    return (result.status, tuple((check.name, check.ok) for check in result.checks))


def rerun(paths):
    """The determinism the constraint asks for: grade every answer twice."""
    return sum(verdict(path) == verdict(path) for path in paths)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    paths = shipped()
    frame = ref.OutcomeFrame(
        user="a reader working through a phase 14 lesson",
        situation="facing the exercises after reading the lesson",
        current_behavior="guesses at an answer with no way to check it",
        desired_outcome="a reader can run an answer and see it check its own claims",
        constraints=["answers import the lesson's code", DETERMINISM,
                     "no dependency outside the standard library"],
        non_goals=["rewriting the lesson"], proposed_output="")
    loose = ref.OutcomeFrame(frame.user, frame.situation, frame.current_behavior,
                             frame.desired_outcome, ["be good"], frame.non_goals, "")
    return {
        "candidates": len(CANDIDATES),
        "kept": survivors(DETERMINISM), "dropped": len(CANDIDATES) - len(survivors(DETERMINISM)),
        "preference_kept": len(survivors(PREFERENCE)),
        "files": len(paths), "volatile_imports": imports(paths),
        "agree": rerun(paths),
        "issues": ref.validate(frame), "loose_issues": ref.validate(loose),
        "constraints": len(frame.constraints), "loose_constraints": len(loose.constraints),
        "reads_constraints": inspect.getsource(ref.validate).count("frame.constraints"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the determinism constraint deletes 3 of 6 candidate outputs",
            all([result["candidates"] == 6, result["dropped"] == 3,
                 len(result["kept"]) == 3, result["files"] == 15,
                 result["volatile_imports"] == 0]),
            f"of {result['candidates']} candidate outputs the constraint leaves "
            f"{len(result['kept'])}: {result['kept']}. The {result['files']} shipped "
            f"answers import time, random or urllib {result['volatile_imports']} times, "
            "which is what the constraint looks like once it has been obeyed",
        ),
        practice.Check(
            "FINDING: the constraint is a string the validator never opens",
            all([result["issues"] == [], result["loose_issues"] == [],
                 result["constraints"] == 3, result["loose_constraints"] == 1,
                 result["reads_constraints"] == 1]),
            f"a frame with {result['constraints']} real constraints and one carrying "
            f"{result['loose_constraints']} slogan both return "
            f"{len(result['issues'])} issues; frame.constraints appears "
            f"{result['reads_constraints']} time in the validator, in the emptiness test",
        ),
        practice.Check(
            "FINDING: the constraint is what makes the outcome checkable",
            all([result["agree"] == 15, result["files"] == 15]),
            f"without it, 'a reader can run an answer' is satisfied by an answer that "
            f"passes on the author's laptop; with it the sentence has a test, and the "
            f"{result['files']} files agree {result['agree']} of {result['files']} times",
        ),
        practice.Check(
            "FINDING: a constraint that deletes nothing is a preference",
            all([result["preference_kept"] == 6, result["candidates"] == 6]),
            f"adding 'prefer clear names' leaves {result['preference_kept']} of "
            f"{result['candidates']} candidates available and the validator equally happy; "
            "the lesson's own examples -- no production writes, no new runtime dependency "
            "-- are all of the deleting kind",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
