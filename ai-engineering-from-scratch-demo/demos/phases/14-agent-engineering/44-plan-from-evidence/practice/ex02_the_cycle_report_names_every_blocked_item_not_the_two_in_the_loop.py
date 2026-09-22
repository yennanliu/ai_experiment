"""Exercise 2 — the cycle report names every blocked item, not the two in the loop.

    Create a cycle and explain the hidden product disagreement behind it.

Reading of the exercise: a cycle in a plan is rarely a modelling slip. It is
two people holding the same decision from opposite ends, written down. The
cycle worth building here is the one the lesson's own example exists to
prevent.

**ANSWER: `docs` and `implementation` waiting on each other is a disagreement
about who owns the public contract.** Documentation waits to learn which
status code shipped; implementation waits to be told which status code was
promised. Neither is wrong, and the deadlock is the org chart, not the graph.
The lesson's `contract` item is the resolution: **1** node that both depend on
turns a cycle of **2** into **3** waves.

**FINDING: the cycle report names the blocked set, not the cycle.** With a
third item depending on the loop, `validate` reports "dependency cycle among:
a, b, c" -- **3** names for a **2**-item cycle, because `execution_waves`
prints whatever is left in `remaining` when nothing is ready. A reader has to
re-derive which edge to cut.

**FINDING: a cycle costs every other check its output.** `validate` reports
**1** issue for the cyclic plan and `plan_document` returns `waves: []`, so
the **3** items that were schedulable lose their ordering too. The first
failure hides the rest of the report.

**FINDING: removing the contract node is what makes the cycle available.**
With `contract` in the graph the same two items schedule together in wave 2 --
**0** issues, **3** waves. Delete it and the only honest edges between docs and
implementation point both ways. The fix for a cycle is usually a missing node,
not a deleted edge.

Structure: `cyclic()` builds the disagreement; `blocked_by()` shows which
names the report adds beyond the loop.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "44-plan-from-evidence"


def item(ref, item_id, depends_on, proof="python3 -m unittest"):
    return ref.WorkItem(item_id, f"{item_id} work", (f"docs/api.md:{len(item_id)}",),
                        tuple(depends_on), proof)


def cyclic(ref):
    """Docs and implementation each waiting for the other to fix the contract."""
    return [item(ref, "implementation", ("docs",)), item(ref, "docs", ("implementation",))]


def blocked_by(ref):
    """The same cycle with one innocent item downstream of it."""
    return cyclic(ref) + [item(ref, "integration", ("docs",))]


def resolved(ref):
    """The lesson's shape: one contract node both sides depend on."""
    return [item(ref, "contract", ()), item(ref, "implementation", ("contract",)),
            item(ref, "docs", ("contract",)),
            item(ref, "integration", ("implementation", "docs"))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    two = cyclic(ref)
    three = blocked_by(ref)
    fixed = resolved(ref)
    document = ref.plan_document(three)
    named = document["issues"][0].split(":")[1].strip().split(", ")
    return {
        "two_issues": ref.validate(two),
        "three_issues": document["issues"], "named": named,
        "loop": sorted({"docs", "implementation"}),
        "extra": sorted(set(named) - {"docs", "implementation"}),
        "status": document["status"], "waves": document["waves"],
        "schedulable": len(three) - 2,
        "issue_count": len(document["issues"]),
        "fixed_issues": ref.validate(fixed), "fixed_waves": ref.execution_waves(fixed),
        "fixed_wave_two": ref.execution_waves(fixed)[1],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the cycle is a disagreement about who owns the public contract",
            all([len(result["two_issues"]) == 1,
                 "dependency cycle" in result["two_issues"][0],
                 result["fixed_issues"] == [], len(result["fixed_waves"]) == 3,
                 result["fixed_wave_two"] == ["docs", "implementation"]]),
            f"docs and implementation waiting on each other gives {result['two_issues']}; "
            f"adding one contract node both depend on gives {len(result['fixed_waves'])} "
            f"waves with {result['fixed_wave_two']} running together and "
            f"{len(result['fixed_issues'])} issues",
        ),
        practice.Check(
            "FINDING: the cycle report names the blocked set, not the cycle",
            all([result["named"] == ["docs", "implementation", "integration"],
                 result["extra"] == ["integration"], len(result["loop"]) == 2]),
            f"with one innocent item downstream, the report names {result['named']} -- "
            f"{len(result['named'])} names for a {len(result['loop'])}-item cycle, because "
            f"the scheduler prints whatever is left when nothing is ready. "
            f"{result['extra']} is blocked, not looping",
        ),
        practice.Check(
            "FINDING: a cycle costs every other check its output",
            all([result["issue_count"] == 1, result["waves"] == [],
                 result["status"] == "blocked", result["schedulable"] == 1]),
            f"plan_document returns {result['issue_count']} issue, status "
            f"{result['status']!r} and waves {result['waves']}, so the "
            f"{result['schedulable']} item that was schedulable loses its ordering too",
        ),
        practice.Check(
            "FINDING: removing the contract node is what makes the cycle available",
            all([result["fixed_wave_two"] == ["docs", "implementation"],
                 result["fixed_issues"] == []]),
            f"with the contract node present the same two items schedule together in wave "
            f"two ({result['fixed_wave_two']}) and validate returns "
            f"{result['fixed_issues']}; without it the only honest edges point both ways. "
            "The fix for a cycle is usually a missing node",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
