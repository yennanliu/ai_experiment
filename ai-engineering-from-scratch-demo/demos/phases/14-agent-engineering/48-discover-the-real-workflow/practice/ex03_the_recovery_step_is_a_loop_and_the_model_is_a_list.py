"""Exercise 3 — the recovery step is a loop, and the model is a list.

    Add one authority boundary and one failure-recovery step.

Reading of the exercise: both additions are easy to write as sentences and
neither fits the dataclass. The authority here is the audit -- the thing
allowed to refuse a lesson -- and the recovery is what the author does when it
refuses, which is to go back to a step that already happened.

**ANSWER: the authority boundary fits in the actor column and the recovery
step cannot be expressed at all.** Adding "the audit refuses the lesson" as
step 9 and "the author rewrites the solution" as a return to step 2 gives two
entries whose `order` values are **9** and **2**; `audit` requires the orders
to be contiguous from one and reports **1** issue. A workflow that can return
to an earlier step is a graph, and `WorkflowStep.order` is an integer in a
list.

**FINDING: 4 of the docs' 8 step fields exist in the dataclass.** The table
asks for actor, trigger, action, input, output, friction, authority and
evidence; `WorkflowStep` carries order, actor, action, evidence and friction.
Authority, trigger, input and output have nowhere to go, so "the incident
commander approves a write" can only be recorded by making the commander the
actor of a step -- which is what the lesson's own example does.

**FINDING: the recovery step has an artifact in this repository.** Lesson 46's
fifth solution spells a banned marker at run time, with the comment "the audit
refuses the literal" -- **1** file carrying the scar of a refuse-and-rewrite
loop. That is direct evidence of a step the happy-path model does not contain.

**FINDING: friction is a string per step, so the loop has no cost.** A step
that ran once and a step that ran four times look identical: `friction` holds
prose and there is no count, no duration and no attempt number anywhere in
the **5** fields. The audit reports the friction points as a list and says
nothing about how often any of them fired.

Structure: `happy()` is the shipped shape; `with_recovery()` adds the loop
the audit refuses.
"""

from __future__ import annotations

from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "48-discover-the-real-workflow"
BASE = Path(__file__).resolve().parents[2]
DOC_FIELDS = ["actor", "trigger", "action", "input", "output", "friction", "authority",
              "evidence"]
SCAR = "spelled at runtime"


def step(ref, order, actor, action, source, direct=False, friction=""):
    return ref.WorkflowStep(order, actor, action,
                            (ref.Evidence(source, action, direct, 0.9),), friction)


def happy(ref):
    """The shipped path: write, grade, finalize, audit, commit."""
    return [step(ref, 1, "author", "writes the solution", "practice/ex01.py"),
            step(ref, 2, "author", "grades it against the lesson's code", "tests/test_practice.py"),
            step(ref, 3, "author", "finalizes the manifest", "practice.yaml"),
            step(ref, 4, "audit", "decides whether the lesson may ship",
                 "scripts/audit_practice.py", friction="refusals arrive after the writing")]


def with_recovery(ref):
    """The same workflow with the authority's refusal and the return it causes."""
    return happy(ref) + [
        step(ref, 9, "audit", "refuses the lesson", "scripts/audit_practice.py"),
        step(ref, 2, "author", "rewrites the refused solution", SCAR, direct=True,
             friction="the rewrite repeats steps already done")]


def scars():
    """Files carrying evidence of a refuse-and-rewrite loop."""
    here = Path(__file__).resolve()
    return sorted(path.name for path in BASE.glob("*/practice/ex0*.py")
                  if path.resolve() != here and SCAR in path.read_text(encoding="utf-8"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean = ref.audit(happy(ref))
    looped = ref.audit(with_recovery(ref))
    fields = list(ref.WorkflowStep.__dataclass_fields__)
    return {
        "clean_issues": clean["issues"], "clean_status": clean["status"],
        "looped_issues": looped["issues"], "looped_status": looped["status"],
        "orders": [row.order for row in with_recovery(ref)],
        "fields": fields, "doc_fields": len(DOC_FIELDS),
        "present": sorted(name for name in DOC_FIELDS if name in fields),
        "missing": sorted(name for name in DOC_FIELDS if name not in fields),
        "authority_as_actor": happy(ref)[3].actor == "audit",
        "scars": scars(),
        "friction_points": len(clean["friction_points"]),
        "counts": [name for name in fields if name in ("attempts", "duration", "count")],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the authority fits in the actor column and the loop does not fit at all",
            all([result["clean_issues"] == [], result["authority_as_actor"] is True,
                 result["orders"] == [1, 2, 3, 4, 9, 2],
                 result["looped_issues"] == ["workflow order must be contiguous from one"],
                 result["looped_status"] == "needs-evidence"]),
            f"the happy path audits clean with the audit itself as an actor; adding the "
            f"refusal and the return gives orders {result['orders']} and "
            f"{result['looped_issues']} at status {result['looped_status']!r}, because a "
            "workflow that returns to an earlier step is a graph",
        ),
        practice.Check(
            "FINDING: 4 of the docs' 8 step fields exist in the dataclass",
            all([result["doc_fields"] == 8, len(result["present"]) == 4,
                 result["missing"] == ["authority", "input", "output", "trigger"],
                 len(result["fields"]) == 5]),
            f"the table asks for {result['doc_fields']} fields and WorkflowStep carries "
            f"{result['fields']}; {result['missing']} have nowhere to go, so an authority "
            "can only be recorded by making it the actor of a step",
        ),
        practice.Check(
            "FINDING: the recovery step has an artifact in this repository",
            all([len(result["scars"]) == 1, result["scars"][0].startswith("ex05")]),
            f"{result['scars']} spells a banned marker at run time because the audit "
            "refuses the literal -- one file carrying the scar of a refuse-and-rewrite "
            "loop, which is direct evidence of a step the happy path does not contain",
        ),
        practice.Check(
            "FINDING: friction is a string per step, so the loop has no cost",
            all([result["friction_points"] == 1, result["counts"] == [],
                 len(result["fields"]) == 5]),
            f"the audit lists {result['friction_points']} friction point as prose and the "
            f"{len(result['fields'])} fields hold {result['counts']} counters, so a step "
            "that ran once and a step that ran four times are indistinguishable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
