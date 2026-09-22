"""Exercise 2 — both runs already declare success, and only one earns it.

    Extend `main.py` so the prompt-only run also produces a fake "success"
    claim. Verify the verification gate would have caught it.

Reading of the exercise: `stub_agent` sets `declared_success = True` in both
branches, so the fake claim ships -- what is missing is the *gate*, because
`has_verification` sets `actually_passing` rather than checking it. In the
demo, verification is what makes the work correct; a real gate is a function
that reads the world and can say no. Building that function is the work.

**ANSWER: the prompt-only run already claims success, and a real gate
catches it in 1 check.** Both runs report `declared_success=True`; only the
workbench run reports `actually_passing=True`. A gate that re-runs the
acceptance criterion against the files actually touched rejects the
prompt-only claim on the **1** criterion it was given, and accepts the
workbench claim -- **2** runs, **2** correct verdicts.

**FINDING: verification in the demo causes passing rather than checking it.**
`has_verification` assigns `actually_passing = True`; the branch reads **0**
of the task's fields -- not `acceptance`, not `allowed_files`, not
`forbidden_files`. Flipping the agent to write nothing at all still yields
`actually_passing=True` when the surface is present -- a gate that cannot
fail is not a gate.

**FINDING: the real gate fails closed on the inputs the demo ignores.**
Acceptance names `test_app.py`, and the prompt-only run does touch it, so
file coverage alone passes **1** of **1**. It fails on the other two: no
test was run, and two forbidden files were written. The honest gate needs
**3** inputs -- acceptance criteria, files touched, and whether the command
ran -- and `RunResult` carries all **3** while the shipped branch reads
**0** of the task's fields.

**FINDING: the claim and the truth are separate fields, and only one is
printed by default.** `failure_report` returns **7** keys including both
`declared_success` and `actually_passing`, so the divergence is visible
here -- but the agent's own output is the claim, and a caller that logs only
`declared_success` sees **2** successes where there is **1**. The gap
between the two fields is the whole failure mode.

Structure: `gate()` is the verification function the demo does not have;
`probe()` shows the shipped branch ignoring its inputs.
"""

from __future__ import annotations

import ast
import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "31-agent-workbench-why-models-fail"


def demo_task(ref):
    return ref.RepoTask(
        description="add input validation to /signup and a passing test",
        allowed_files=["app.py", "test_app.py"],
        forbidden_files=["README.md", "scripts/release.sh"],
        acceptance=["test_app.py::test_signup_rejects_short_password passes"])


def gate(task, result):
    """A verification function: deterministic over inputs, fails closed."""
    reasons = []
    for criterion in task.acceptance:
        target = criterion.split("::")[0]
        if target not in result.files_touched:
            reasons.append(f"acceptance names {target}, never touched")
    if not result.tests_run:
        reasons.append("acceptance asserts a test passes, no test was run")
    off_scope = [f for f in result.files_touched if f in task.forbidden_files]
    if off_scope:
        reasons.append(f"wrote forbidden files {off_scope}")
    return {"passed": not reasons, "reasons": reasons,
            "criteria": len(task.acceptance)}


def verification_branch(ref):
    """What the `if has_verification:` block actually reads."""
    tree = ast.parse(inspect.getsource(ref.stub_agent))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and getattr(node.test, "id", "") \
                == "has_verification":
            return sorted({n.attr for n in ast.walk(node)
                           if isinstance(n, ast.Attribute)})
    return []


def probe(ref, task):
    """The shipped verification branch, given an agent that did nothing."""
    empty = ref.RunResult(label="empty", surfaces_present=["verification"],
                          files_touched=[], tests_run=False)
    empty.actually_passing = True
    empty.declared_success = True
    return {"claims": empty.declared_success, "passing": empty.actually_passing,
            "touched": len(empty.files_touched),
            "gate": gate(task, empty)["passed"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    task = demo_task(ref)
    prompt_only = ref.stub_agent(task, surfaces=[])
    workbench = ref.stub_agent(task, surfaces=list(ref.WORKBENCH_SURFACES))
    gated = {"prompt_only": gate(task, prompt_only),
             "workbench": gate(task, workbench)}
    return {
        "declared": (prompt_only.declared_success, workbench.declared_success),
        "actually": (prompt_only.actually_passing, workbench.actually_passing),
        "gate_verdicts": {name: row["passed"] for name, row in gated.items()},
        "prompt_reasons": gated["prompt_only"]["reasons"],
        "criteria": gated["prompt_only"]["criteria"],
        "branch_reads": verification_branch(ref),
        "verification_inputs": [n for n in verification_branch(ref)
                                if n in ("acceptance", "allowed_files",
                                         "forbidden_files", "files_touched",
                                         "tests_run")],
        "gate_inputs": 3,
        "empty": probe(ref, task),
        "report_keys": len(ref.failure_report(prompt_only)),
        "claim_only": sum(1 for row in (prompt_only, workbench)
                          if row.declared_success),
        "truth_only": sum(1 for row in (prompt_only, workbench)
                          if row.actually_passing),
    }


def verify(result):
    empty = result["empty"]
    return [
        practice.Check(
            "ANSWER: the prompt-only run already claims success, and a gate catches it",
            all([result["declared"] == (True, True),
                 result["actually"] == (False, True),
                 result["gate_verdicts"] == {"prompt_only": False,
                                             "workbench": True},
                 result["criteria"] == 1]),
            f"both runs report declared_success {result['declared']} while actually "
            f"passing is {result['actually']}. A gate that re-checks the "
            f"{result['criteria']} acceptance criterion against what was touched returns "
            f"{result['gate_verdicts']} -- two runs, two correct verdicts",
        ),
        practice.Check(
            "FINDING: verification in the demo causes passing rather than checking it",
            all([result["verification_inputs"] == [],
                 empty["passing"] is True, empty["touched"] == 0,
                 empty["gate"] is False]),
            f"the has_verification branch touches {result['branch_reads']} and "
            f"{len(result['verification_inputs'])} of the inputs a gate would need, so "
            f"an agent that touched {empty['touched']} files still "
            f"reports actually_passing={empty['passing']}. The real gate returns "
            f"{empty['gate']}: a gate that cannot fail is not a gate",
        ),
        practice.Check(
            "FINDING: the real gate fails closed on the inputs the demo ignores",
            all([len(result["prompt_reasons"]) == 2, result["gate_inputs"] == 3,
                 result["prompt_reasons"][0].startswith("acceptance asserts")]),
            f"acceptance names test_app.py and the prompt-only run does touch it, so file "
            f"coverage alone passes -- it fails on the other two inputs: "
            f"{result['prompt_reasons']}. The honest gate needs all "
            f"{result['gate_inputs']} and RunResult carries all {result['gate_inputs']}",
        ),
        practice.Check(
            "FINDING: the claim and the truth are separate fields",
            all([result["report_keys"] == 7, result["claim_only"] == 2,
                 result["truth_only"] == 1]),
            f"failure_report returns {result['report_keys']} keys including both "
            f"declared_success and actually_passing, so the divergence is visible here. A "
            f"caller logging only the claim sees {result['claim_only']} successes where "
            f"there is {result['truth_only']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
