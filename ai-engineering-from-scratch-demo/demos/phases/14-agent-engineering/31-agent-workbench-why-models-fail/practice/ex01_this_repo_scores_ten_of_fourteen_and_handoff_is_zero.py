"""Exercise 1 — this repo scores ten of fourteen, and handoff is zero.

    Pick a repo where you already run an agent. Score the seven surfaces from
    0 (missing) to 2 (healthy). What is your weakest surface?

Reading of the exercise: the repo an agent actually runs in, for this file,
is the one the file lives in -- so the scoring is done against
`ai-engineering-from-scratch-demo` with evidence a reader can check, not
against a hypothetical. Each surface gets **0** when nothing implements it,
**1** when something does but nothing enforces it, and **2** when it is
enforced by a script the build runs.

**ANSWER: 10 of 14, and handoff is the weakest surface at 0.** Instructions
**2** (`DESIGN.md` plus `audit_practice.py` enforcing D10/D12/D14), state
**1** (`practice.yaml` per lesson, read at build time, never written by a
run), scope **2** (one file per exercise, index-checked), feedback **2**
(`practice.selfcheck` prints every failing check's measured value),
verification **2** (`pytest` plus `audit_practice.py`), review **1**
(`coverage.py` reports, nothing gates on it), handoff **0**.

**FINDING: the weakest surface is the one with no artifact to point at.**
Every surface scoring **2** has a script that fails the build; the surface
scoring **0** has no file, no script and no convention -- a session that ends
mid-lesson leaves the next one to re-derive where it was from `git log`. The
score is really a count of *enforcement*, and **4** of the **7** surfaces
have it.

**FINDING: the stub agent reads 4 of the 7 surfaces.** `stub_agent` branches
on `scope`, `state`, `verification` and `feedback`; `instructions`, `review`
and `handoff` appear in `WORKBENCH_SURFACES` and are never read, so passing
them changes **0** of the result's fields. Three of the seven surfaces are
labels in the demo.

**FINDING: `state` is the only surface whose absence changes nothing
observable.** With `state` removed the result differs by **1** note and **0**
fields -- no file, no touched list, no verdict moves -- so a run scored on
`failure_report` cannot tell a stateful session from a stateless one. It is
the surface whose failure is invisible in the same run and expensive in the
next.

Structure: `EVIDENCE` names a checkable artifact per surface; `score()`
turns presence into 0, 1 or 2.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "31-agent-workbench-why-models-fail"
ROOT = pathlib.Path(__file__).resolve().parents[5]
# (surface, artifact that implements it, script that enforces it)
EVIDENCE = (
    ("instructions", "DESIGN.md", "scripts/audit_practice.py"),
    ("state", "demos/phases/14-agent-engineering/31-agent-workbench-why-models-fail"
     "/practice/practice.yaml", None),
    ("scope", "harness/manifest.py", "scripts/audit_practice.py"),
    ("feedback", "harness/practice.py", "scripts/audit_practice.py"),
    ("verification", "tests", "scripts/check_deps.py"),
    ("review", "scripts/coverage.py", None),
    ("handoff", None, None),
)


def score(artifact, enforcer):
    """0 when nothing implements it, 1 when nothing enforces it, 2 when both."""
    if artifact is None or not (ROOT / artifact).exists():
        return 0
    return 2 if enforcer and (ROOT / enforcer).exists() else 1


def scores():
    return {surface: score(artifact, enforcer)
            for surface, artifact, enforcer in EVIDENCE}


def surface_effect(ref, task, surface):
    """Fields of the result that change when one surface is removed."""
    full = ref.stub_agent(task, surfaces=list(ref.WORKBENCH_SURFACES))
    without = ref.stub_agent(task, surfaces=[s for s in ref.WORKBENCH_SURFACES
                                             if s != surface])
    fields = ("files_touched", "tests_run", "declared_success", "actually_passing")
    changed = [f for f in fields if getattr(full, f) != getattr(without, f)]
    return {"fields": changed,
            "notes": len(without.notes) - len(full.notes)}


def read_surfaces(ref):
    names = ref.stub_agent.__code__.co_consts
    return [s for s in ref.WORKBENCH_SURFACES if s in names]


def demo_task(ref):
    return ref.RepoTask(
        description="add input validation to /signup and a passing test",
        allowed_files=["app.py", "test_app.py"],
        forbidden_files=["README.md", "scripts/release.sh"],
        acceptance=["test_app.py::test_signup_rejects_short_password passes"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = scores()
    task = demo_task(ref)
    return {
        "scores": table, "total": sum(table.values()), "ceiling": 2 * len(table),
        "weakest": min(table, key=lambda s: (table[s], s)),
        "enforced": sum(v == 2 for v in table.values()),
        "surfaces": len(ref.WORKBENCH_SURFACES),
        "read": read_surfaces(ref),
        "state_effect": surface_effect(ref, task, "state"),
        "scope_effect": surface_effect(ref, task, "scope"),
        "root_name": ROOT.name,
    }


def verify(result):
    table, state = result["scores"], result["state_effect"]
    return [
        practice.Check(
            "ANSWER: 10 of 14, and handoff is the weakest surface at 0",
            all([result["total"] == 10, result["ceiling"] == 14,
                 result["weakest"] == "handoff", table["handoff"] == 0,
                 result["root_name"] == "ai-engineering-from-scratch-demo",
                 table["instructions"] == 2, table["state"] == 1,
                 table["review"] == 1]),
            f"scoring {result['root_name']} on checkable artifacts gives {table} -- "
            f"{result['total']}/{result['ceiling']}. The weakest surface is "
            f"{result['weakest']!r} at {table[result['weakest']]}: no file, no script, "
            "no convention",
        ),
        practice.Check(
            "FINDING: the score is really a count of enforcement",
            all([result["enforced"] == 4, result["ceiling"] == 14,
                 sum(v == 1 for v in table.values()) == 2]),
            f"{result['enforced']} of {len(table)} surfaces score 2, and every one of "
            f"them has a script that fails the build; the {sum(v == 1 for v in table.values())} "
            "scoring 1 have an artifact and no gate. A surface nobody can fail the build "
            "on is documentation",
        ),
        practice.Check(
            "FINDING: the stub agent reads 4 of the 7 surfaces",
            all([len(result["read"]) == 4, result["surfaces"] == 7,
                 sorted(result["read"]) == ["feedback", "scope", "state",
                                            "verification"]]),
            f"stub_agent branches on {sorted(result['read'])} and never reads "
            f"{sorted(set(ref_surfaces()) - set(result['read']))}, so "
            f"{result['surfaces'] - len(result['read'])} of the "
            f"{result['surfaces']} surfaces are labels in the demo",
        ),
        practice.Check(
            "FINDING: state is the surface whose absence changes nothing observable",
            all([state["fields"] == [], state["notes"] == 1,
                 result["scope_effect"]["fields"] == ["files_touched"]]),
            f"removing state changes {len(state['fields'])} result fields and adds "
            f"{state['notes']} note, where removing scope changes "
            f"{result['scope_effect']['fields']}. A run scored on failure_report cannot "
            "tell a stateful session from a stateless one",
        ),
    ]


def ref_surfaces():
    return ["instructions", "state", "scope", "feedback", "verification",
            "review", "handoff"]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
