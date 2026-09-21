"""Exercise 4 — scope catches it first, and verification never does.

    Re-run the script with a different stub agent that hallucinates an extra
    file write. Which surface catches it first?

Reading of the exercise: "first" is an ordering question, so the surfaces
have to be put in the order a run actually reaches them -- scope before the
write, feedback during, verification at task close, review after. A stub
that writes one file nobody asked for is then run past each gate in turn and
the first refusal is the answer.

**ANSWER: scope catches it before the write happens; verification never
sees it.** A stub that adds `config/prod.yaml` to an otherwise correct run
is refused by scope at position **1** of **4**. Feedback records the write
without judging it, the shipped verification branch reads **0** of the task's
fields and returns `actually_passing=True`, and review flags it at position
**4** -- after the file exists.

**FINDING: the shipped `failure_report` cannot see this write at all.**
`off_scope_writes` compares against the hardcoded set `{"app.py",
"test_app.py"}` rather than `task.allowed_files`, so it happens to catch
`config/prod.yaml` here -- and on a task whose allowed files are
`["src/api.py"]` it reports **1** legitimate write as off-scope and the real
violation as fine. The check is correct for exactly one task.

**FINDING: prevention and detection are different positions, not different
strengths.** Scope refuses **1** write and the repo is unchanged; review
finds the same write and the repo has **3** files where **2** were allowed.
Both are "caught", and only one is recoverable without a revert -- which is
why the ordering, not the count, is the useful output of this exercise.

**FINDING: a hallucinated write is invisible to every surface that only
reads the agent's own report.** The stub declares success either way, so
scope, feedback and review all have to read the *filesystem* rather than the
`RunResult`. Of the shipped report's **7** keys, **1** (`off_scope_writes`)
is derived from observation and **6** are the agent's own account.

Structure: `GATES` is the four surfaces in execution order; `first_catch()`
walks them until one refuses.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "31-agent-workbench-why-models-fail"
HALLUCINATED = "config/prod.yaml"


def demo_task(ref, allowed=None):
    return ref.RepoTask(
        description="add input validation to /signup and a passing test",
        allowed_files=allowed or ["app.py", "test_app.py"],
        forbidden_files=["README.md", "scripts/release.sh"],
        acceptance=["test_app.py::test_signup_rejects_short_password passes"])


def hallucinating_stub(ref, task, surfaces):
    """The shipped stub plus one file nobody asked for."""
    result = ref.stub_agent(task, surfaces=surfaces)
    result.files_touched = [*result.files_touched, HALLUCINATED]
    result.notes.append(f"also wrote {HALLUCINATED}")
    return result


def scope_gate(task, result):
    extra = [f for f in result.files_touched if f not in task.allowed_files]
    return not extra, extra


def feedback_gate(task, result):
    del task
    return True, [f"logged {len(result.files_touched)} writes"]


def verification_gate(task, result):
    del task
    return result.actually_passing, []


def review_gate(task, result):
    extra = [f for f in result.files_touched if f not in task.allowed_files]
    return not extra, extra


GATES = (("scope", scope_gate), ("feedback", feedback_gate),
         ("verification", verification_gate), ("review", review_gate))


def first_catch(task, result):
    for position, (name, gate) in enumerate(GATES, start=1):
        ok, detail = gate(task, result)
        if not ok:
            return {"surface": name, "position": position, "detail": detail}
    return {"surface": None, "position": None, "detail": []}


def repo_state(task, result, stop_at):
    """Files on disk once the run reaches `stop_at`, given writes happen after scope."""
    if stop_at == "scope":
        return len(task.allowed_files)
    return len(set(result.files_touched))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    task = demo_task(ref)
    run = hallucinating_stub(ref, task, list(ref.WORKBENCH_SURFACES))
    caught = first_catch(task, run)
    other = demo_task(ref, allowed=["src/api.py"])
    other_run = ref.RunResult(label="other", files_touched=["src/api.py",
                                                           HALLUCINATED])
    shipped = ref.failure_report(other_run)
    report_keys = list(ref.failure_report(run))
    return {
        "gates": len(GATES), "caught": caught,
        "verification_verdict": verification_gate(task, run)[0],
        "review_position": [i for i, (n, _) in enumerate(GATES, start=1)
                            if n == "review"][0],
        "shipped_flags": ref.failure_report(run)["off_scope_writes"],
        "other_flags": shipped["off_scope_writes"],
        "other_allowed": other.allowed_files,
        "scope_repo": repo_state(task, run, "scope"),
        "review_repo": repo_state(task, run, "review"),
        "allowed": len(task.allowed_files),
        "report_keys": len(report_keys),
        "observed_keys": [k for k in report_keys if k == "off_scope_writes"],
        "declared": run.declared_success,
    }


def verify(result):
    caught = result["caught"]
    return [
        practice.Check(
            "ANSWER: scope catches it at position 1 and verification never does",
            all([caught["surface"] == "scope", caught["position"] == 1,
                 caught["detail"] == [HALLUCINATED], result["gates"] == 4,
                 result["verification_verdict"] is True,
                 result["review_position"] == 4]),
            f"a stub that adds {HALLUCINATED!r} is refused by "
            f"{caught['surface']!r} at position {caught['position']} of "
            f"{result['gates']}. The shipped verification branch returns "
            f"{result['verification_verdict']} regardless, and review would flag it at "
            f"position {result['review_position']} -- after the file exists",
        ),
        practice.Check(
            "FINDING: the shipped failure_report is correct for exactly one task",
            all([result["shipped_flags"] == [HALLUCINATED],
                 result["other_flags"] == ["src/api.py", HALLUCINATED],
                 result["other_allowed"] == ["src/api.py"]]),
            f"off_scope_writes compares against a hardcoded set rather than "
            f"task.allowed_files, so on this task it reports "
            f"{result['shipped_flags']} and on a task allowing "
            f"{result['other_allowed']} it reports {result['other_flags']} -- the "
            "legitimate write flagged alongside the real violation",
        ),
        practice.Check(
            "FINDING: prevention and detection are different positions",
            all([result["scope_repo"] == 2, result["review_repo"] == 3,
                 result["allowed"] == 2]),
            f"stopped at scope the repo has {result['scope_repo']} files, which is the "
            f"{result['allowed']} that were allowed; stopped at review it has "
            f"{result['review_repo']}. Both surfaces catch the same write and only one "
            "leaves nothing to revert",
        ),
        practice.Check(
            "FINDING: only one of the report's keys is an observation",
            all([result["report_keys"] == 7,
                 result["observed_keys"] == ["off_scope_writes"],
                 result["declared"] is True]),
            f"the stub declares success ({result['declared']}) either way, so of the "
            f"report's {result['report_keys']} keys exactly "
            f"{len(result['observed_keys'])} is derived from what the filesystem shows "
            "and the rest are the agent's own account of itself",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
