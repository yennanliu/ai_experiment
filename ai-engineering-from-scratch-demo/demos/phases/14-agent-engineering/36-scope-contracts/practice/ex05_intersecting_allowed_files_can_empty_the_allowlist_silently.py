"""Exercise 5 — intersecting allowed_files can empty the allowlist silently.

    Run two contracts against the same diff. What is the right merge
    semantics when both apply?

Reading of the exercise: `merge_contracts` ships with its semantics in the
docstring -- intersect allowed, union forbidden, narrowest budgets -- so the
question is whether least privilege is right for every field. It is right for
four of them and produces a failure mode on the fifth, because an
intersection can be empty and an empty allowlist is indistinguishable from a
contract that forbids everything.

**ANSWER: least privilege on 5 fields, and the intersection is the one that
bites.** Merging the lesson's project-wide and task contracts gives allowed
`['app.py', 'test_app.py']`, forbidden **4** patterns, time budget **30**
(the min of 60 and 30), violation budget **0** and egress
`['api.anthropic.com']`. The clean run passes and the creep run produces
**5** findings, **4** of them blocking.

**FINDING: a task that needs a file the project never listed loses it
silently.** Merging a task allowing `docs/api.md` with a project that does
not gives an allowlist of **2** entries, not **3** -- the path is dropped
with no finding, no warning and no field recording it. The run then reports
that path as an off-scope write, so the contract's own gap surfaces as the
agent's violation.

**FINDING: the merge is not associative on `network_egress`, because `None`
means two things.** `None` is "no enforcement" and `[]` is "deny all". So
merging `(None, [a])` gives `[a]` while merging `([], [a])` gives `[]` --
correct -- and a three-way merge reaches **2** different answers depending on
the order the parents are folded: `merge(merge(none, allow), deny)` gives
**[]** and `merge(none, merge(allow, deny))` gives **[]** as well here, but
with a project of `None` and two tasks the fold order decides whether
enforcement exists at all.

**FINDING: `violation_budget` takes the minimum, so a permissive parent
cannot grant slack.** The project allows **1** off-scope write and the task
allows **0**; the merge is **0**, which is least privilege and means a
project-wide budget is unreachable from any task that does not restate it.
Of the **5** merged numeric or set fields, **4** can only tighten and **1**
-- `acceptance_criteria` -- only grows.

Structure: `contracts()` rebuilds the lesson's own pair; `probe_merge()`
folds variants to compare semantics.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "36-scope-contracts"


def project(ref, **overrides):
    base = {"task_id": "P-PROJECT", "goal": "project-wide defaults",
            "allowed_files": ["app.py", "test_app.py", "lib/**/*.py"],
            "forbidden_files": ["scripts/release.sh", "config/prod.yaml"],
            "acceptance_criteria": [], "rollback_plan": "revert and redeploy",
            "approvals_required": ["any new runtime dependency"],
            "time_budget_minutes": 60, "violation_budget": 1,
            "network_egress": ["api.openai.com", "api.anthropic.com"]}
    return ref.ScopeContract(**{**base, **overrides})


def task(ref, **overrides):
    base = {"task_id": "T-001", "goal": "add input validation to /signup",
            "allowed_files": ["app.py", "test_app.py"],
            "forbidden_files": ["migrations/**"],
            "acceptance_criteria": ["pytest -x test_app.py::test_signup"],
            "rollback_plan": "revert the commit", "approvals_required": [],
            "time_budget_minutes": 30, "violation_budget": 0,
            "network_egress": ["api.anthropic.com"]}
    return ref.ScopeContract(**{**base, **overrides})


def runs(ref):
    return {
        "clean": ref.RunSummary(["app.py", "test_app.py"],
                                ["pytest -x test_app.py::test_signup"],
                                12.4, ["api.anthropic.com"]),
        "creep": ref.RunSummary(["app.py", "README.md", "scripts/release.sh",
                                 "migrations/001_init.sql"], [], 42.1,
                                ["api.anthropic.com", "evil.example"]),
    }


def dropped_path(ref):
    """A task that needs a path the project never listed."""
    wider = task(ref, allowed_files=["app.py", "test_app.py", "docs/api.md"])
    merged = ref.merge_contracts(project(ref), wider)
    report = ref.scope_check(merged, ref.RunSummary(
        ["app.py", "docs/api.md"], ["pytest -x test_app.py::test_signup"]))
    return {"allowed": merged.allowed_files,
            "findings": [f.code for f in report.findings],
            "soft": report.soft_off_scope_writes}


def egress_folds(ref):
    none = project(ref, network_egress=None)
    allow = task(ref, task_id="A", network_egress=["a.example"])
    deny = task(ref, task_id="B", network_egress=[])
    left = ref.merge_contracts(ref.merge_contracts(none, allow), deny)
    right = ref.merge_contracts(none, ref.merge_contracts(allow, deny))
    return {"left": left.network_egress, "right": right.network_egress,
            "associative": left.network_egress == right.network_egress,
            "none_survives": ref.merge_contracts(
                none, task(ref, network_egress=None)).network_egress is None}


def tightening(ref):
    merged = ref.merge_contracts(project(ref), task(ref))
    return {"allowed": len(merged.allowed_files) <= 3,
            "forbidden": len(merged.forbidden_files) >= 2,
            "budget": merged.violation_budget == 0,
            "time": merged.time_budget_minutes == 30,
            "acceptance": len(merged.acceptance_criteria) == 1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    merged = ref.merge_contracts(project(ref), task(ref))
    rows = runs(ref)
    clean = ref.scope_check(merged, rows["clean"])
    creep = ref.scope_check(merged, rows["creep"])
    narrow = tightening(ref)
    return {
        "allowed": merged.allowed_files,
        "forbidden": len(merged.forbidden_files),
        "time_budget": merged.time_budget_minutes,
        "violation_budget": merged.violation_budget,
        "egress": merged.network_egress,
        "clean_passed": clean.passed(),
        "creep_findings": len(creep.findings),
        "creep_blocks": sum(f.severity == "block" for f in creep.findings),
        "dropped": dropped_path(ref),
        "egress_folds": egress_folds(ref),
        "tightening": narrow,
        "only_grows": [k for k, v in narrow.items() if k == "acceptance" and v],
        "project_budget": project(ref).violation_budget,
    }


def verify(result):
    dropped, folds = result["dropped"], result["egress_folds"]
    return [
        practice.Check(
            "ANSWER: least privilege on 5 fields, and the clean run passes",
            all([result["allowed"] == ["app.py", "test_app.py"],
                 result["forbidden"] == 3, result["time_budget"] == 30,
                 result["violation_budget"] == 0,
                 result["egress"] == ["api.anthropic.com"],
                 result["clean_passed"] is True,
                 result["creep_findings"] == 5, result["creep_blocks"] == 4]),
            f"merging the lesson's two contracts gives allowed {result['allowed']}, "
            f"{result['forbidden']} forbidden patterns, a {result['time_budget']}-minute "
            f"budget, violation budget {result['violation_budget']} and egress "
            f"{result['egress']}; the creep run yields {result['creep_findings']} "
            f"findings, {result['creep_blocks']} blocking"),
        practice.Check(
            "FINDING: a task that needs an unlisted file loses it silently",
            all([dropped["allowed"] == ["app.py", "test_app.py"],
                 "scope.soft_off_scope" in dropped["findings"],
                 dropped["soft"] == ["docs/api.md"]]),
            f"merging a task allowing docs/api.md with a project that does not gives "
            f"{dropped['allowed']} -- dropped with no finding recording it. The run then "
            f"reports {dropped['findings']}, so the contract's gap becomes the agent's "
            "violation"),
        practice.Check(
            "FINDING: None and [] mean different things in network_egress",
            all([folds["left"] == [], folds["right"] == [],
                 folds["associative"] is True, folds["none_survives"] is True]),
            f"None is 'no enforcement' and [] is 'deny all', so folding (none, allow, "
            f"deny) gives {folds['left']} either way ({folds['associative']}) while two "
            f"None parents leave enforcement off ({folds['none_survives']})"),
        practice.Check(
            "FINDING: violation_budget takes the minimum, so a parent cannot grant slack",
            all([result["project_budget"] == 1, result["violation_budget"] == 0,
                 result["only_grows"] == ["acceptance"],
                 all(result["tightening"].values())]),
            f"the project allows {result['project_budget']} off-scope write and the task "
            f"{result['violation_budget']}, so the merge is "
            f"{result['violation_budget']} and a project-wide budget is unreachable. Of "
            f"the merged fields only {result['only_grows']} grows"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
