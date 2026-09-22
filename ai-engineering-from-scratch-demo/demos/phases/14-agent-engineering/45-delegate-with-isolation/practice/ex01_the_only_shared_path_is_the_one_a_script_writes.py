"""Exercise 1 — the only shared path is the one a script writes.

    Decompose a real change into two independent work units and one
    integrator.

Reading of the exercise: "real" means the decomposition has to survive the
repository's own history. The change decomposed here is one this repository
performs every day -- ship the practice solutions for two lessons -- and the
test of independence is whether the two units' path sets are disjoint in the
commits that actually happened.

**ANSWER: two lesson directories decompose cleanly and the integrator owns the
generated file.** `43-frame-the-task-before-code/practice` and
`44-plan-from-evidence/practice` overlap in **0** paths, schedule as
`[['lesson-43', 'lesson-44'], ['integration']]`, and the plan validates
`ready`. The one path neither worker may own is the repository's top-level
`README.md`: **6** of the last **6** lesson commits touch it, because
`scripts/coverage.py` rewrites it from the manifests.

**FINDING: the contract in the docs has 6 fields and `WorkUnit` has 5.**
`goal` and `handoff` are missing -- the two that carry what the unit is for
and what came back. `id`, `owner`, `paths`, `depends_on` and `proof` are what
a worker is actually given, so "files changed, decisions made, remaining risk"
has nowhere to land in the artifact the integrator reads.

**FINDING: a unit that owns no paths validates ready.** `delegation_plan`
checks duplicate ids, path overlap and empty proofs; **0** checks require a
unit to own anything. A work unit with `paths=()` is either a read-only
worker -- exercise 3 -- or a decomposition error, and the plan cannot tell
them apart.

**FINDING: overlap is computed per path pair, and never within a unit.**
`conflicts` walks every pair of paths across every pair of *distinct* units,
so one worker claiming the phase directory produces **2** findings against a
neighbour that owns two nested paths -- one mistake, two lines -- while those
two nested paths overlap each other and nothing reports it. The count in the
report is not the number of decisions the integrator has to make.

Structure: `units()` builds the decomposition; `history()` measures which
paths the real commits touch together.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "45-delegate-with-isolation"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
BASE = "demos/phases/14-agent-engineering"
UNITS = [
    ("lesson-43", "worker-a", (f"{BASE}/43-frame-the-task-before-code/practice",),
     (), "uv run pytest demos/phases/14-agent-engineering/43-frame-the-task-before-code"),
    ("lesson-44", "worker-b", (f"{BASE}/44-plan-from-evidence/practice",),
     (), "uv run pytest demos/phases/14-agent-engineering/44-plan-from-evidence"),
    ("integration", "integrator", ("README.md",), ("lesson-43", "lesson-44"),
     "uv run python scripts/audit_practice.py 14-agent-engineering"),
]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def history(count=6):
    """The last lesson commits, and how many README files each one touches."""
    shas = git("log", "--format=%H", "-n", str(count), "--", BASE).split()
    rows = []
    for sha in shas:
        files = [line for line in git("show", "--name-only", "--format=", sha).splitlines()
                 if line.strip()]
        rows.append({"sha": sha[:7], "files": len(files),
                     "root_readme": any(line.endswith("-demo/README.md") for line in files)})
    return rows


def units(ref, rows=UNITS):
    return [ref.WorkUnit(*row) for row in rows]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plan = ref.delegation_plan(units(ref))
    commits = history()
    greedy = list(UNITS)
    greedy[0] = (UNITS[0][0], UNITS[0][1],
                 (UNITS[0][2][0], f"{UNITS[0][2][0]}/tests"), (), UNITS[0][4])
    greedy[1] = ("lesson-44", "worker-b", (BASE,), (), UNITS[1][4])
    pathless = units(ref) + [ref.WorkUnit("research", "reader", (), (), "fact table")]
    return {
        "status": plan["status"], "conflicts": plan["conflicts"],
        "waves": plan["waves"], "units": len(plan["units"]),
        "commits": len(commits),
        "root_touched": sum(row["root_readme"] for row in commits),
        "doc_fields": 6, "unit_fields": list(ref.WorkUnit.__dataclass_fields__),
        "missing_fields": [name for name in ("goal", "handoff")
                           if name not in ref.WorkUnit.__dataclass_fields__],
        "pathless_status": ref.delegation_plan(pathless)["status"],
        "pathless_conflicts": len(ref.delegation_plan(pathless)["conflicts"]),
        "greedy_conflicts": ref.delegation_plan(units(ref, greedy))["conflicts"],
        "self_overlap": ref.paths_overlap(UNITS[0][2][0], f"{UNITS[0][2][0]}/tests"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two lesson directories decompose cleanly; the integrator owns the "
            "generated file",
            all([result["status"] == "ready", result["conflicts"] == [],
                 result["waves"] == [["lesson-43", "lesson-44"], ["integration"]],
                 result["units"] == 3, result["root_touched"] == result["commits"] == 6]),
            f"the two workers overlap in {len(result['conflicts'])} paths and schedule as "
            f"{result['waves']}; the top-level README is touched by "
            f"{result['root_touched']} of the last {result['commits']} lesson commits "
            "because scripts/coverage.py rewrites it, so it belongs to the integrator",
        ),
        practice.Check(
            "FINDING: the documented contract has 6 fields and WorkUnit has 5",
            all([len(result["unit_fields"]) == 5,
                 result["missing_fields"] == ["goal", "handoff"]]),
            f"the dataclass carries {result['unit_fields']} and the docs table asks for "
            f"{result['doc_fields']}; {result['missing_fields']} are missing, so what the "
            "unit is for and what came back have nowhere to live",
        ),
        practice.Check(
            "FINDING: a unit that owns no paths validates ready",
            all([result["pathless_status"] == "ready", result["pathless_conflicts"] == 0]),
            f"adding a unit with paths=() leaves the plan {result['pathless_status']!r} with "
            f"{result['pathless_conflicts']} conflicts: a read-only worker and a "
            "decomposition error look the same to the planner",
        ),
        practice.Check(
            "FINDING: overlap is computed per path pair, and never within a unit",
            all([len(result["greedy_conflicts"]) == 2, result["self_overlap"] is True]),
            f"a worker claiming the phase directory produces "
            f"{len(result['greedy_conflicts'])} findings against a neighbour that owns two "
            "nested paths -- one mistake, two lines -- while those two nested paths overlap "
            f"each other ({result['self_overlap']}) and nothing reports it, because "
            "conflicts only compares distinct units",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
