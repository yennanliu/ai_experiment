"""Exercise 2 — the questions that flow up are the fields nobody checks.

    Add a 5th role: "requirements analyst" that translates user wish into
    planner-ready spec. What communicative dehallucination requests should flow
    up to it?

Reading of the exercise: add the role, then find the answer by elimination --
a request only needs to travel upward if no role downstream can settle it, so
count which parts of the spec any role actually reads.

**ANSWER: `signature` and `description` -- the two Spec fields that **0**
roles check.** Of the four fields, `task_name` is read by the critic and the
verifier, `tests` by the verifier, and `signature` and `description` by
neither. Those two are exactly where a downstream agent has no oracle and no
recourse, so those are the requests that must go back up: *what are the
parameter types and the return type*, and *what does "sum of two integers"
mean at the boundaries* -- overflow, non-integers, whether it must reject
floats. Everything else can be settled without asking, because something
downstream can check it.

**FINDING: the unchecked fields are already shipping wrong answers.** An
artifact defined as `def add_two(*args): return sum(args)` contradicts the
spec's stated `add_two(a: int, b: int) -> int` and is approved by the critic
and passed by the verifier. Nothing in the pipeline compares the artifact
against `signature`, so the analyst's output would be the first thing in the
system with a claim that no role can enforce.

**FINDING: the executor cannot raise a request, because it never reads the
spec.** `executor_correct(spec)` and `executor_buggy(spec)` both return a
constant `Artifact`; the parameter is unused in **2** of **2** executors.
Communicative dehallucination is a *request* from the role that discovered the
gap, and the role most likely to discover one here never looks at the
requirements at all.

**FINDING: there is nowhere to put a question.** `Artifact` has **1** field,
`code`, and `CriticReport` and `VerifierReport` carry booleans and note lists
addressed to no one. A request that flows up needs a recipient and a subject;
adding the analyst therefore means adding a channel, not just a function.

Structure: `analyst()` is the fifth role; `readers()` counts which roles
consult which Spec field.
"""

from __future__ import annotations

import dataclasses
import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "08-role-specialization"
WISH = "A function that returns the sum of two integers."


def analyst(wish):
    """The fifth role: a wish becomes the fields a planner needs, plus its questions."""
    return {"task_name": "add_two",
            "signature": "add_two(a: int, b: int) -> int",
            "description": wish,
            "questions": ["are non-int arguments an error or coerced?",
                          "is the result required to fit in a machine int?"]}


def readers(ref):
    """Which roles read which Spec field."""
    sources = {"critic": inspect.getsource(ref.critic),
               "verifier": inspect.getsource(ref.verifier)}
    fields = [field.name for field in dataclasses.fields(ref.Spec)]
    return {field: [role for role, text in sources.items() if f"spec.{field}" in text]
            for field in fields}


def unused_parameter(function):
    """Does this executor look at the spec it is handed?"""
    source = inspect.getsource(function)
    body = source[source.index(":") + 1:]
    return "spec" not in body


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spec = ref.planner(WISH)
    seen = readers(ref)
    wrong = ref.Artifact(code="def add_two(*args):\n    return sum(args)\n")
    proposed = analyst(WISH)
    return {
        "fields": sorted(seen),
        "unread": sorted(field for field, roles in seen.items() if not roles),
        "read_by": {field: len(roles) for field, roles in seen.items()},
        "wrong_signature_critic": ref.critic(spec, wrong).approved,
        "wrong_signature_verifier": ref.verifier(spec, wrong).passed,
        "signature_compared": "signature" in inspect.getsource(ref.critic)
                              or "signature" in inspect.getsource(ref.verifier),
        "blind_executors": sum(unused_parameter(function) for function in
                               (ref.executor_correct, ref.executor_buggy)),
        "executors": 2,
        "artifact_fields": len(dataclasses.fields(ref.Artifact)),
        "questions": len(proposed["questions"]),
        "covers_unread": sorted(set(proposed) & {"signature", "description"}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: signature and description, the two fields no role checks",
            all([result["unread"] == ["description", "signature"],
                 result["read_by"]["task_name"] == 2,
                 result["read_by"]["tests"] == 1,
                 result["covers_unread"] == ["description", "signature"]]),
            f"of the {len(result['fields'])} Spec fields, task_name is read by "
            f"{result['read_by']['task_name']} roles and tests by "
            f"{result['read_by']['tests']}, while {' and '.join(result['unread'])} are "
            "read by none -- those are where a downstream agent has no oracle, so those "
            "are the requests that must travel up",
        ),
        practice.Check(
            "FINDING: the unchecked fields are already shipping wrong answers",
            all([result["wrong_signature_critic"], result["wrong_signature_verifier"],
                 not result["signature_compared"]]),
            "def add_two(*args) contradicts the spec's add_two(a: int, b: int) -> int "
            "and is approved by the critic and passed by the verifier, because nothing "
            "compares the artifact against signature -- the analyst's output would be "
            "the first claim in the system no role can enforce",
        ),
        practice.Check(
            "FINDING: the executor cannot raise a request, because it never reads the spec",
            result["blind_executors"] == result["executors"] == 2,
            f"both of the {result['executors']} executors return a constant Artifact "
            f"with the spec parameter unused in {result['blind_executors']} of them; a "
            "dehallucination request comes from the role that found the gap, and the "
            "role most likely to find one never looks",
        ),
        practice.Check(
            "FINDING: there is nowhere to put a question",
            all([result["artifact_fields"] == 1, result["questions"] == 2]),
            f"Artifact has {result['artifact_fields']} field and the two reports carry "
            f"booleans and notes addressed to no one, so the analyst's "
            f"{result['questions']} questions have no recipient -- adding the role means "
            "adding a channel, not just a function",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
