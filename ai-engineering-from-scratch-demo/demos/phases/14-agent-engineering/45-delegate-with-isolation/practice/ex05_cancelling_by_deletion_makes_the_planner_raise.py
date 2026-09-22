"""Exercise 5 — cancelling by deletion makes the planner raise.

    Define a cancellation rule for a worker whose dependency becomes invalid.

Reading of the exercise: the rule has two halves -- what happens to the unit
whose dependency died, and what happens to the plan document. The second half
is where the shipped code has an opinion, and it is not the one you want.

**ANSWER: cancel by marking, never by deleting, and cascade to the transitive
dependents.** Dropping `api` from the example makes `delegation_plan` raise
`ValueError("unknown dependency")` instead of returning a blocked document,
because `waves` is called unguarded for anything other than a duplicate id.
Marking it cancelled instead leaves **3** units, cancels the **1** dependent
that transitively needed it, and still renders a plan.

**FINDING: two of the three failure modes raise instead of reporting.**
`delegation_plan` guards `waves` only against duplicate ids, so an unknown
dependency and a cycle both escape as exceptions while overlap and missing
proof come back as fields. A caller that wants a report has to wrap the call
in `try`, which is the shape of an API that was not designed for the blocked
case.

**FINDING: cancellation is a graph question, not a unit question.** Cancelling
`api` in the three-unit example cancels **1** further unit and leaves **1**
running; in a five-unit chain the same single cancellation stops **3**. The
rule has to walk dependents transitively or it stops the wrong amount of work.

**FINDING: a cancelled unit's paths must stay reserved until its dependents
settle.** Freeing `app/api` the moment `api` is cancelled lets another worker
claim it while `integration` is still holding a half-finished tree; keeping
the reservation costs **0** conflicts here and is the difference between a
cancellation and a race.

Structure: `cancel()` marks a unit and cascades; `chain()` is the five-unit
case where the cascade size changes.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "45-delegate-with-isolation"


def dependents(units, target):
    """Everything that transitively waits on `target`."""
    blocked, changed = {target}, True
    while changed:
        changed = False
        for unit in units:
            if unit.id not in blocked and set(unit.depends_on) & blocked:
                blocked.add(unit.id)
                changed = True
    return sorted(blocked - {target})


def cancel(units, target):
    """Mark, do not delete: the unit stays in the plan with its paths reserved."""
    stopped = set(dependents(units, target)) | {target}
    return [{"id": unit.id, "state": "cancelled" if unit.id in stopped else "running",
             "paths": list(unit.paths)} for unit in units]


def chain(ref):
    """A five-unit chain so the cascade has room to differ."""
    rows = [("a", ()), ("b", ("a",)), ("c", ("b",)), ("d", ("c",)), ("e", ())]
    return [ref.WorkUnit(name, f"worker-{name}", (f"app/{name}",), deps, "python3 -m unittest")
            for name, deps in rows]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    units = ref.example()
    deleted = [unit for unit in units if unit.id != "api"]
    try:
        ref.delegation_plan(deleted)
        raised = ""
    except ValueError as error:
        raised = str(error)
    try:
        ref.delegation_plan([ref.WorkUnit("a", "o", ("p1",), ("b",), "x"),
                             ref.WorkUnit("b", "o", ("p2",), ("a",), "x")])
        cycle_raised = ""
    except ValueError as error:
        cycle_raised = str(error)
    marked = cancel(units, "api")
    long_chain = chain(ref)
    reserved = ref.delegation_plan(units)
    return {
        "raised": raised, "cycle_raised": cycle_raised,
        "guarded": inspect.getsource(ref.delegation_plan).count("duplicate_ids else []"),
        "reported": ["duplicate_ids", "conflicts", "missing_proof"],
        "marked": marked, "units_kept": len(marked),
        "cancelled": sorted(row["id"] for row in marked if row["state"] == "cancelled"),
        "running": sorted(row["id"] for row in marked if row["state"] == "running"),
        "cascade_small": dependents(units, "api"),
        "cascade_long": dependents(long_chain, "a"),
        "reserved_paths": sorted(path for row in marked for path in row["paths"]
                                 if row["state"] == "cancelled"),
        "conflicts": reserved["conflicts"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: cancel by marking, never by deleting, and cascade to dependents",
            all([result["raised"] == "unknown dependency", result["units_kept"] == 3,
                 result["cancelled"] == ["api", "integration"],
                 result["running"] == ["docs"], result["cascade_small"] == ["integration"]]),
            f"deleting the unit raises {result['raised']!r} instead of returning a blocked "
            f"document; marking it keeps {result['units_kept']} units, cancels "
            f"{result['cancelled']} and leaves {result['running']} running",
        ),
        practice.Check(
            "FINDING: two of the three failure modes raise instead of reporting",
            all([result["cycle_raised"] == "dependency cycle", result["guarded"] == 1,
                 len(result["reported"]) == 3]),
            f"an unknown dependency and a cycle ({result['cycle_raised']!r}) escape as "
            f"exceptions while {result['reported']} come back as fields, because waves is "
            "guarded against duplicate ids alone",
        ),
        practice.Check(
            "FINDING: cancellation is a graph question, not a unit question",
            all([len(result["cascade_small"]) == 1, result["cascade_long"] == ["b", "c", "d"],
                 len(result["cascade_long"]) == 3]),
            f"cancelling one unit stops {len(result['cascade_small'])} more in the shipped "
            f"example and {len(result['cascade_long'])} in a five-unit chain "
            f"({result['cascade_long']}); the rule has to walk dependents transitively",
        ),
        practice.Check(
            "FINDING: a cancelled unit's paths must stay reserved",
            all([result["reserved_paths"] == ["app/api", "tests/test_api.py",
                                              "tests/test_integration.py"],
                 result["conflicts"] == []]),
            f"the cancelled units still hold {result['reserved_paths']}; freeing them "
            "immediately would let another worker claim a half-finished tree, and keeping "
            f"the reservation costs {len(result['conflicts'])} conflicts",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
