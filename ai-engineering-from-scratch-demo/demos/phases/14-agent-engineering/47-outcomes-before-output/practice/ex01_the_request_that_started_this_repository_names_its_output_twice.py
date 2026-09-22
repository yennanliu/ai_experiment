"""Exercise 1 — the request that started this repository names its output twice.

    Rewrite a feature request from your backlog as an outcome frame.

Reading of the exercise: the request rewritten here is the one that produced
the files around this solution -- "complete all of the lessons in phase 14,
commit and push once a lesson is done". It names a unit of work and a delivery
mechanism, and says nothing about who is better off.

**ANSWER: the request leaks its output twice and the rewrite states an
observable result.** "Complete all of the lessons" names the artifact and
"commit and push" names the delivery, so **2** of the request's **2** clauses
are outputs. The frame -- a reader of a lesson can run a solution that
verifies its own claims against the lesson's code -- validates with **0**
issues and never mentions a practice file.

**FINDING: the leak check only fires when the frame names its own output.**
`validate` compares `proposed_output` against `desired_outcome` as a
substring, and `proposed_output` defaults to `""`, so a frame that leaves it
blank cannot fail the check no matter what the outcome says. The original
request, framed with a blank proposed output, returns **0** issues while
naming the artifact outright.

**FINDING: substring matching misses the paraphrase it exists to catch.**
Writing the outcome as "triage with the assistant" against a proposed output
of "incident assistant" returns **0** issues, while "use the incident
assistant to triage" returns **1**. The check tests spelling, not leakage,
which is the same failure Lesson 46's keyword router had.

**FINDING: the frame has 7 fields and the program only counts.** Four strings
are checked for being non-blank, `constraints` and `non_goals` are checked for
being non-empty and never read again -- `non_goals` appears in the validator
exactly once -- and the one semantic rule in the module is the substring test.
The discipline is in the writing; the program counts.

Structure: `framed()` is the rewrite; `leaks()` runs the lesson's own check
over the original request and its paraphrases.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "47-outcomes-before-output"
REQUEST = "complete all of the lessons in phase 14, commit and push once a lesson is done"
OUTPUT = "practice solutions"


def framed(ref, desired=None, proposed=OUTPUT):
    return ref.OutcomeFrame(
        user="a reader working through a phase 14 lesson",
        situation="after reading the lesson and running its code, facing its exercises",
        current_behavior="reads the exercise, guesses at an answer, and has no way to "
                         "tell whether the guess matches what the lesson's code does",
        desired_outcome=desired or "a reader can run an answer to every exercise and see "
                                   "it check its own claims against the lesson's own code",
        constraints=["answers import the lesson's code rather than forking it",
                     "the same command produces the same verdict on any machine",
                     "no dependency outside the standard library for a T0 lesson"],
        non_goals=["rewriting the lesson text", "translating anything",
                   "changing the reference curriculum"],
        proposed_output=proposed)


def leaks(ref):
    """The original request as an outcome, spelled four ways."""
    rows = {
        "request-blank": framed(ref, desired=REQUEST, proposed=""),
        "request-named": framed(ref, desired=REQUEST, proposed="lessons"),
        "paraphrase": framed(ref, desired="triage with the assistant",
                             proposed="incident assistant"),
        "literal": framed(ref, desired="use the incident assistant to triage",
                          proposed="incident assistant"),
    }
    return {name: ref.validate(frame) for name, frame in rows.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    frame = framed(ref)
    checks = leaks(ref)
    validator = inspect.getsource(ref.validate)
    fields = list(ref.OutcomeFrame.__dataclass_fields__)
    return {
        "issues": ref.validate(frame),
        "status": ref.decision(frame)["status"],
        "mentions_output": OUTPUT in frame.desired_outcome,
        "request_clauses": 2,
        "blank": checks["request-blank"], "named": checks["request-named"],
        "paraphrase": checks["paraphrase"], "literal": checks["literal"],
        "default_blank": ref.OutcomeFrame.__dataclass_fields__["proposed_output"].default == "",
        "fields": fields,
        "emptiness": validator.count("if not "),
        "semantic": validator.count("in frame.desired_outcome.lower()"),
        "unread": [name for name in ("constraints", "non_goals")
                   if validator.count(name) == 1],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the request leaks its output twice and the rewrite states a result",
            all([result["issues"] == [], result["status"] == "ready-to-discover",
                 result["mentions_output"] is False, result["request_clauses"] == 2]),
            f"the rewritten frame validates with {len(result['issues'])} issues at status "
            f"{result['status']!r} and never names the artifact; the original request's "
            f"{result['request_clauses']} clauses name the unit of work and the delivery "
            "mechanism and nobody who is better off",
        ),
        practice.Check(
            "FINDING: the leak check only fires when the frame names its own output",
            all([result["blank"] == [], result["default_blank"] is True,
                 result["named"] == ["desired outcome names the proposed output"]]),
            f"with proposed_output blank -- its default -- the original request as an "
            f"outcome returns {result['blank']}; naming the output turns the same sentence "
            f"into {result['named']}",
        ),
        practice.Check(
            "FINDING: substring matching misses the paraphrase it exists to catch",
            all([result["paraphrase"] == [], len(result["literal"]) == 1]),
            f"'triage with the assistant' against a proposed 'incident assistant' returns "
            f"{result['paraphrase']} while the literal spelling returns "
            f"{len(result['literal'])} issue: the check tests spelling, not leakage",
        ),
        practice.Check(
            "FINDING: the frame has 7 fields and the program only counts",
            all([len(result["fields"]) == 7, result["emptiness"] == 3,
                 result["semantic"] == 1, result["unread"] == ["non_goals"]]),
            f"{len(result['fields'])} fields, {result['emptiness']} emptiness tests and "
            f"{result['semantic']} semantic rule -- the substring test. The contents of "
            f"constraints and non_goals are never read, and {result['unread']} appears in "
            "the validator exactly once, in the test that it is non-empty",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
