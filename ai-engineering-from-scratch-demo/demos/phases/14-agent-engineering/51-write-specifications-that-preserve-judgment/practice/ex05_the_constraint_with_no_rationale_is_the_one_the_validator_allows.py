"""Exercise 5 — the constraint with no rationale is the one the validator allows.

    Remove a constraint that has no evidence or risk rationale.

Reading of the exercise: find a constraint carrying no reason, check that
nothing is lost by removing it, and remove it. The shipped specification has
exactly one candidate, and it is a candidate precisely because the validator
permits it.

**ANSWER: "changing alert routing" is a non-goal with no rationale anywhere,
and removing it changes nothing the document can measure.** The
specification carries **2** non-goals and neither shares a single word with
any decision's rationale, so the unjustified one is identified by reading
rather than by matching: automatic remediation is what the locked
production-authority decision exists to exclude, and alert routing is argued
for nowhere. Deleting it leaves **1** non-goal and the contract still
compiles `executable` with **0** issues.

**FINDING: only decisions can carry a reason, and only some of those are
asked for one.** `Decision` has a `rationale` field; `invariants`,
`examples`, `non_goals` and `proof` are lists of bare strings -- **4** of the
**6** surfaces cannot hold a justification at all. Removing an unjustified
constraint is therefore a judgment made outside the artifact, every time.

**FINDING: which constraint is defended cannot be computed from the
document.** **0** of the **2** non-goals shares vocabulary with a rationale,
so no mechanical check separates the one the locked decision exists to
exclude from the one nothing argues for. Deleting the defended one instead
leaves that locked decision standing with nothing to exclude and still
reports **0** issues: both deletions look identical to `validate`, and only
one of them loses an argument.

**FINDING: an empty surface is the one thing the validator does catch.**
Removing both non-goals makes `validate` report `non_goals is empty` -- **1**
issue -- so the schema's opinion is about presence, not content. A
specification is allowed to carry constraints nobody can defend, as long as
it carries some.

Structure: `without()` removes a named non-goal; `anchored()` reports which
constraints another surface argues for.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "51-write-specifications-that-preserve-judgment"
CANDIDATE = "changing alert routing"
ANCHORED = "automatic remediation"


def without(ref, spec, removed):
    return ref.Specification(spec.outcome, list(spec.invariants), list(spec.examples),
                             [goal for goal in spec.non_goals if goal != removed],
                             list(spec.decisions), list(spec.proof))


def anchored(spec):
    """Non-goals that another surface argues for, by word overlap with a rationale."""
    reasons = " ".join(item.rationale.lower() for item in spec.decisions)
    rows = {}
    for goal in spec.non_goals:
        words = {word for word in goal.lower().split() if len(word) > 4}
        rows[goal] = bool(words & set(reasons.split()))
    return rows


def surfaces_with_rationale(ref):
    fields = ref.Specification.__dataclass_fields__
    return [name for name in fields if name == "decisions"], list(fields)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spec = ref.example()
    trimmed = without(ref, spec, CANDIDATE)
    lost_anchor = without(ref, spec, ANCHORED)
    empty = ref.Specification(spec.outcome, list(spec.invariants), list(spec.examples),
                              [], list(spec.decisions), list(spec.proof))
    carriers, fields = surfaces_with_rationale(ref)
    links = anchored(spec)
    return {
        "non_goals": len(spec.non_goals), "after": len(trimmed.non_goals),
        "removed": CANDIDATE,
        "status": ref.compile_contract(trimmed)["status"],
        "issues": ref.validate(trimmed),
        "anchored": [goal for goal, linked in links.items() if linked],
        "unanchored": [goal for goal, linked in links.items() if not linked],
        "carriers": carriers, "surfaces": len(fields),
        "without_rationale": len(fields) - len(carriers) - 1,
        "lost_anchor_issues": ref.validate(lost_anchor),
        "locked_left": [item.question for item in lost_anchor.decisions
                        if item.mode == "locked"],
        "empty_issues": ref.validate(empty),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the unjustified non-goal comes out and nothing measurable changes",
            all([result["non_goals"] == 2, result["after"] == 1,
                 result["removed"] == CANDIDATE, result["status"] == "executable",
                 result["issues"] == [],
                 result["unanchored"] == [ANCHORED, CANDIDATE]]),
            f"the specification carries {result['non_goals']} non-goals, neither of which "
            f"shares any vocabulary with a decision's rationale, so {result['removed']!r} "
            f"is identified as unjustified by reading rather than by matching; removing it "
            f"leaves {result['after']} and the contract still compiles "
            f"{result['status']!r} with {len(result['issues'])} issues",
        ),
        practice.Check(
            "FINDING: only decisions can carry a reason",
            all([result["carriers"] == ["decisions"], result["surfaces"] == 6,
                 result["without_rationale"] == 4]),
            f"of the {result['surfaces']} surfaces, {result['carriers']} holds a rationale "
            f"field and {result['without_rationale']} are lists of bare strings, so "
            "removing an unjustified constraint is a judgment made outside the artifact",
        ),
        practice.Check(
            "FINDING: which constraint is defended cannot be computed from the document",
            all([result["anchored"] == [], len(result["unanchored"]) == 2,
                 result["lost_anchor_issues"] == [],
                 len(result["locked_left"]) == 1]),
            f"{len(result['anchored'])} of the {result['non_goals']} non-goals shares "
            "vocabulary with any rationale, although one of them is exactly what the locked "
            "production-authority decision exists to exclude; deleting that one instead "
            f"leaves {len(result['locked_left'])} locked decision with nothing to exclude "
            f"and still reports {len(result['lost_anchor_issues'])} issues",
        ),
        practice.Check(
            "FINDING: an empty surface is the one thing the validator does catch",
            result["empty_issues"] == ["non_goals is empty"],
            f"removing both non-goals reports {result['empty_issues']}, so the schema's "
            "opinion is about presence rather than content: a specification may carry "
            "constraints nobody can defend as long as it carries some",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
