"""Exercise 4 — the merge gate finds the file no contract claims.

    Add a merge gate that checks the final changed-file set against all unit
    contracts.

Reading of the exercise: the planner checks contracts against each other
before the work. The merge gate checks the work against the contracts
afterwards, which is a different question and needs a different input: the
actual changed-file set, from git.

**ANSWER: on a real checkout the gate flags 2 of 6 changed files -- one
unclaimed and one written by the wrong owner.** Three workers edit a temp
repository; `git status --porcelain` reports **6** paths. `docs/api.md` and
`app/api/routes.py` match their owners, `scripts/release.sh` is claimed by no
contract, and `tests/test_api.py` is edited by the api worker although the
docs unit owns it. The planner said `ready` before any of this
happened.

**FINDING: the gate needs the owner, not just the union of paths.** Checking
membership in the union catches **1** of the **2** problems: an unclaimed
file. Attributing each change to the worker that made it is what catches the
second, and the artifact already has an `owner` field to compare against.

**FINDING: `paths_overlap` is blind to the glob a real contract wants to
use.** `paths_overlap("app/**", "app/api/routes.py")` is **False**, so a
contract written with globs passes the pre-work check and then owns nothing at
merge time. The lesson's directory-prefix form works and the glob form
silently does not.

**FINDING: the gate is the first thing in the lesson that reads the
repository.** `delegation_plan` takes **1** argument -- the list of contracts
-- and the module never imports `subprocess`, so every check before the merge
is a check on the plan's internal consistency. A plan can be perfectly
consistent and describe work nobody did.

Structure: `worktree()` builds the changed-file set; `merge_gate()` compares
it against the contracts.
"""

from __future__ import annotations

import inspect
import subprocess
import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "45-delegate-with-isolation"
# (path, the worker that actually wrote it)
EDITS = [("app/api/routes.py", "worker-api"), ("tests/test_api.py", "worker-api"),
         ("docs/api.md", "worker-docs"), ("scripts/release.sh", "worker-api"),
         ("app/api/schemas.py", "worker-api"), ("docs/errors.md", "worker-docs")]


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True).stdout.strip()


def worktree():
    """A checkout the three workers have already edited."""
    root = Path(tempfile.mkdtemp(prefix="merge-"))
    git(root, "init", "-q")
    git(root, "config", "user.email", "a@b.co")
    git(root, "config", "user.name", "tester")
    for name, _ in EDITS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("before\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "baseline")
    for name, _ in EDITS:
        (root / name).write_text("after\n")
    return root


def owns(ref, unit, path):
    return any(ref.paths_overlap(owned, path) for owned in unit.paths)


def merge_gate(ref, units, changed):
    """Each changed path against every contract: who claimed it, who wrote it."""
    rows = []
    for path, author in changed:
        claimants = [unit for unit in units if owns(ref, unit, path)]
        rows.append({"path": path, "author": author,
                     "claimed_by": [unit.owner for unit in claimants],
                     "unclaimed": not claimants,
                     "wrong_owner": bool(claimants) and author not in
                     [unit.owner for unit in claimants]})
    return rows


def contracts(ref):
    return [
        ref.WorkUnit("api", "worker-api", ("app/api",), (), "python3 -m unittest tests.test_api"),
        ref.WorkUnit("docs", "worker-docs", ("docs", "tests/test_api.py"), (),
                     "python3 scripts/check_links.py"),
        ref.WorkUnit("integration", "reviewer", ("tests/test_integration.py",),
                     ("api", "docs"), "python3 -m unittest"),
    ]


def union_misses(ref, units, changed):
    """What a union-of-paths check alone would flag."""
    union = {path for unit in units for path in unit.paths}
    return sorted(path for path in changed
                  if not any(ref.paths_overlap(owned, path) for owned in union))


def changed_paths(root):
    return sorted(line.split(maxsplit=1)[1]
                  for line in git(root, "status", "--porcelain").splitlines() if line.strip())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    units = contracts(ref)
    changed = changed_paths(worktree())
    authored = dict(EDITS)
    rows = merge_gate(ref, units, [(path, authored[path]) for path in changed])
    plan = ref.delegation_plan(units)
    flagged = [row for row in rows if row["unclaimed"] or row["wrong_owner"]]
    return {
        "changed": len(changed), "rows": len(rows),
        "flagged": sorted(row["path"] for row in flagged),
        "unclaimed": sorted(row["path"] for row in rows if row["unclaimed"]),
        "wrong_owner": sorted(row["path"] for row in rows if row["wrong_owner"]),
        "union_only": union_misses(ref, units, changed),
        "plan_status": plan["status"], "plan_conflicts": plan["conflicts"],
        "glob_blind": ref.paths_overlap("app/**", "app/api/routes.py"),
        "prefix_works": ref.paths_overlap("app/api", "app/api/routes.py"),
        "plan_inputs": len(inspect.signature(ref.delegation_plan).parameters),
        "runs_git": "subprocess" in Path(ref.__file__).read_text(),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the gate flags 2 of 6 changed files",
            all([result["changed"] == 6, result["rows"] == 6,
                 result["unclaimed"] == ["scripts/release.sh"],
                 result["wrong_owner"] == ["tests/test_api.py"],
                 result["plan_status"] == "ready"]),
            f"git reports {result['changed']} changed paths; the gate flags "
            f"{result['flagged']} -- {result['unclaimed']} claimed by no contract and "
            f"{result['wrong_owner']} written by a worker that does not own it -- "
            f"while the planner said {result['plan_status']!r} beforehand",
        ),
        practice.Check(
            "FINDING: the gate needs the owner, not just the union of paths",
            all([result["union_only"] == ["scripts/release.sh"],
                 len(result["flagged"]) == 2]),
            f"membership in the union of owned paths catches {result['union_only']}, "
            f"{len(result['union_only'])} of the {len(result['flagged'])} problems; "
            "attributing each change to the worker that made it catches the other, and the "
            "contract already carries an owner field",
        ),
        practice.Check(
            "FINDING: paths_overlap is blind to the glob a real contract wants",
            all([result["glob_blind"] is False, result["prefix_works"] is True]),
            f"paths_overlap('app/**', 'app/api/routes.py') is {result['glob_blind']} while "
            f"the directory-prefix form is {result['prefix_works']}: a contract written "
            "with globs passes the pre-work check and owns nothing at merge time",
        ),
        practice.Check(
            "FINDING: the gate is the first thing in the lesson that reads the repository",
            all([result["runs_git"] is False, result["plan_inputs"] == 1,
                 result["plan_conflicts"] == []]),
            f"delegation_plan takes {result['plan_inputs']} argument -- the list of "
            f"contracts -- and the module never imports subprocess ({result['runs_git']}), "
            "so every check before the merge is a check on the plan's internal "
            "consistency. A consistent plan can describe work nobody did",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
