"""Exercise 1 — the first row is two different deployments wearing one label.

    Create all four rows of the human/model matrix and write one legitimate
    use case for each.

Reading of the exercise: a use case is only legitimate if the matrix row it
claims is the row the runtime actually produces, so each of the four is built
as a real `InvocationPolicy` and routed with a real human request and a real
model request. Doing that four times makes the first row fail the test twice
-- once as each of the two deployments its own label names.

**ANSWER: four policies, four use cases, and the routed behaviour matches
every row.** `(False, False)` is a nightly evaluation harness, `(True,
False)` a destructive migration a person must name, `(False, True)` a
formatting convention nobody asks for by name, and `(True, True)` an ordinary
report. **8** requests, **8** decisions agreeing with the row.

**FINDING: the first row is two different deployments wearing one label.**
`programmatic-only or unavailable` is an or, and the matrix cannot say which.
The same `(False, False)` policy activates the skill for a harness caller
when the name is in `harness_allowlist` and denies it when the allowlist is
empty -- **1** row, **2** opposite outcomes, and the row prints identically
either way.

**FINDING: an agent is not a model here, and the matrix has no column for
it.** `allow_agent` is a separate flag and `CorePolicyAdapter` reads it on a
separate branch, so a policy with `allow_model=False, allow_agent=True`
prints in the "no model activation" row while an agent activates the skill.
The matrix reads **2** of the policy's **6** actor flags.

**FINDING: `active_policy` marks exactly one row, and marks none without a
policy.** It compares a 2-tuple, so the four rows are exhaustive and
disjoint by construction -- which is the matrix's real content, and is also
why it cannot express the three actors it leaves out.

Structure: `behaviour()` routes one human and one model request under a
policy, so a row's claim and the runtime's answer are the same evidence.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "25-skill-invocation-and-routing"
SKILL = "release-readiness"
DESCRIPTION = "Review merged pull request summaries and report release readiness."
USE_CASES = {
    (False, False): "a nightly evaluation harness runs it and nobody else may",
    (True, False): "a destructive migration a person has to name out loud",
    (False, True): "a formatting convention nobody thinks to ask for by name",
    (True, True): "an ordinary report, asked for either way",
}


def policy_for(ref, human, model, **extra):
    return ref.InvocationPolicy(allow_human=human, allow_model=model, allow_agent=model,
                                model_threshold=0.15, **extra)


def behaviour(ref, skills, policy):
    """What the runtime does with one human and one model request under this policy."""
    adapter = ref.CorePolicyAdapter(policy)
    human = ref.route_request(skills, ref.InvocationRequest(
        ref.Actor.HUMAN, "", explicit_name=SKILL), adapter)
    model = ref.route_request(skills, ref.InvocationRequest(
        ref.Actor.MODEL, "report release readiness for merged pull requests"), adapter)
    return human.activated, model.activated


def flags_read(ref, name):
    """Which policy.allow_* fields this function loads -- the matrix's real width."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ref, name))))
    return sorted({node.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute) and node.attr.startswith("allow_")})


def row_for(rows, human, model):
    return next(row for row in rows if row["human"] == human and row["model"] == model)


def harness_call(ref, skills, policy):
    adapter = ref.CorePolicyAdapter(policy)
    return ref.route_request(skills, ref.InvocationRequest(
        ref.Actor.HARNESS, "nightly evaluation", explicit_name=SKILL), adapter).activated


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    skills = (ref.SkillMetadata(SKILL, DESCRIPTION),)
    matched, marked = [], []
    for (human, model), use_case in USE_CASES.items():
        policy = policy_for(ref, human, model)
        rows = ref.build_invocation_matrix(policy)
        matched.append(behaviour(ref, skills, policy) == (human, model))
        marked.append(sum(row["active_policy"] for row in rows))

    dark = policy_for(ref, False, False)
    lit = policy_for(ref, False, False, harness_allowlist=(SKILL,))
    rows_dark = ref.build_invocation_matrix(dark)
    rows_lit = ref.build_invocation_matrix(lit)

    split = ref.InvocationPolicy(allow_human=False, allow_model=False, allow_agent=True,
                                 model_threshold=0.15)
    agent = ref.route_request(skills, ref.InvocationRequest(
        ref.Actor.AGENT, "report release readiness for merged pull requests"),
        ref.CorePolicyAdapter(split)).activated
    flags = [name for name in vars(ref.InvocationPolicy)["__dataclass_fields__"]
             if name.startswith("allow_")]
    return {
        "use_cases": len(USE_CASES), "matched": matched, "requests": 2 * len(USE_CASES),
        "marked": marked, "rows": len(rows_dark),
        "meaning": row_for(rows_dark, False, False)["meaning"],
        "dark_harness": harness_call(ref, skills, dark),
        "lit_harness": harness_call(ref, skills, lit),
        "same_row": row_for(rows_dark, False, False) == row_for(rows_lit, False, False),
        "agent_activated": agent, "agent_row": row_for(
            ref.build_invocation_matrix(split), False, False)["meaning"],
        "flags": sorted(flags),
        "matrix_reads": flags_read(ref, "build_invocation_matrix"),
        "unmarked": sum(row["active_policy"] for row in ref.build_invocation_matrix(None)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four policies, four use cases, and the routed behaviour matches each row",
            all([result["use_cases"] == 4, result["matched"] == [True] * 4,
                 result["rows"] == 4, result["marked"] == [1] * 4]),
            f"{result['use_cases']} rows built as real policies and exercised with "
            f"{result['requests']} requests agree with the row every time: "
            f"{list(USE_CASES.values())[0]!r} for (False, False) up to "
            f"{list(USE_CASES.values())[3]!r} for (True, True)",
        ),
        practice.Check(
            "FINDING: the first row is two different deployments wearing one label",
            all([result["meaning"] == "programmatic-only or unavailable",
                 result["dark_harness"] is False, result["lit_harness"] is True,
                 result["same_row"]]),
            f"{result['meaning']!r} is an or, and the row cannot say which. The same "
            f"(False, False) policy denies a harness caller with an empty allowlist and "
            "activates it with the name listed -- one row, two opposite outcomes, and the "
            "row itself is byte-identical in both cases",
        ),
        practice.Check(
            "FINDING: an agent is not a model here, and the matrix has no column for it",
            all([result["agent_activated"], result["agent_row"].startswith("programmatic"),
                 len(result["flags"]) == 6,
                 result["matrix_reads"] == ["allow_human", "allow_model"]]),
            f"allow_agent is a separate flag on a separate adapter branch, so a policy with "
            f"allow_model=False and allow_agent=True prints in the "
            f"{result['agent_row']!r} row while the agent activates the skill. The matrix "
            f"reads {result['matrix_reads']} -- {len(result['matrix_reads'])} of the "
            f"{len(result['flags'])} actor flags {result['flags']}",
        ),
        practice.Check(
            "FINDING: active_policy marks exactly one row, and marks none without a policy",
            all([result["marked"] == [1] * 4, result["unmarked"] == 0]),
            f"the flag compares a 2-tuple, so the four rows are exhaustive and disjoint by "
            f"construction -- {result['marked']} marked across the four policies and "
            f"{result['unmarked']} with no policy at all. That is the matrix's real content, "
            "and the same reason it cannot express the three actors it leaves out",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
