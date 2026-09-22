"""Exercise 5 — the scope receipt is the only acceptance line that can fail on a green test.

    Add a scope receipt to the acceptance evidence.

Reading of the exercise: a scope receipt is a command whose output is the file
list, checked against the frame's allowed paths. It is the one piece of
acceptance evidence that can fail while every test passes, which is what makes
it worth adding rather than assuming.

**ANSWER: in a real checkout the receipt flags 1 of the 3 changed files while
the test command exits 0.** A temp git repository with the frame's two allowed
paths edited plus one write to `deploy/release.sh` reports
`git diff --name-only` as **3** paths, **1** of them outside the allowed set
and matching a forbidden glob. Nothing else in the frame notices: the
acceptance list still reads "tests pass".

**FINDING: `acceptance` is a list of strings and nothing runs them.**
`validate` checks the list is non-empty, so a frame whose only acceptance
evidence is "make coffee" returns **0** issues and renders `Status: READY`.
The receipt has to be a command someone runs; the frame is a promise that it
was.

**FINDING: the receipt needs the forbidden globs, which the shipped check
cannot use.** `deploy/release.sh` is outside the allowed exact paths *and*
inside the forbidden `deploy/**` glob, which are two different severities --
Lesson 38 calls them `scope.off_scope` and `scope.forbidden`. Comparing
against the allowed list alone collapses them: **1** finding where there
should be **2** levels.

**FINDING: an untracked file is invisible to the receipt as written.**
`git diff --name-only` reports **3** paths and misses a new untracked file
entirely; `git status --porcelain` reports **4**. The agent's most common
scope violation -- adding a file nobody asked for -- is the one the obvious
command cannot see.

Structure: `repo()` builds a real checkout and edits it; `receipt()` runs the
scope command and compares against the frame.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "43-frame-the-task-before-code"
ALLOWED = ["app/accounts.py", "tests/test_accounts.py"]
FORBIDDEN = ["migrations/**", "deploy/**"]
EDITS = ALLOWED + ["deploy/release.sh"]


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True).stdout.strip()


def repo():
    """A checkout carrying the frame's files, then edited the way an agent would."""
    root = Path(tempfile.mkdtemp(prefix="frame-"))
    git(root, "init", "-q")
    git(root, "config", "user.email", "a@b.co")
    git(root, "config", "user.name", "tester")
    for name in EDITS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "baseline")
    for name in EDITS:
        (root / name).write_text("edited\n")
    (root / "notes.md").write_text("scratch\n")
    return root


def matches(path, patterns):
    return any(path == pattern or path.startswith(pattern.rstrip("*"))
               for pattern in patterns)


def receipt(root, allowed, forbidden):
    """The scope receipt: what changed, and how each change scores."""
    changed = [line for line in git(root, "diff", "--name-only").splitlines() if line]
    porcelain = [line[3:] for line in git(root, "status", "--porcelain").splitlines() if line]
    off_scope = [path for path in changed if path not in allowed]
    return {"changed": changed, "porcelain": porcelain, "off_scope": off_scope,
            "forbidden": [path for path in off_scope if matches(path, forbidden)]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = repo()
    scope = receipt(root, ALLOWED, FORBIDDEN)
    command = "git diff --name-only -- . ':!deploy'"
    framed = ref.TaskFrame(
        goal="Prevent duplicate email addresses during signup",
        allowed_paths=ALLOWED, forbidden_paths=FORBIDDEN,
        acceptance=["python3 -m unittest tests.test_accounts", command],
        facts=[ref.RepositoryFact("Account writes use AccountStore", "app/accounts.py:1")],
        unknowns=[])
    coffee = ref.TaskFrame(goal="g", allowed_paths=ALLOWED, forbidden_paths=FORBIDDEN,
                           acceptance=["make coffee"], facts=[], unknowns=[])
    return {
        "changed": scope["changed"], "off_scope": scope["off_scope"],
        "forbidden": scope["forbidden"],
        "porcelain": len(scope["porcelain"]), "tracked": len(scope["changed"]),
        "untracked_seen": "notes.md" in scope["porcelain"],
        "untracked_in_diff": "notes.md" in scope["changed"],
        "acceptance_count": len(framed.acceptance),
        "issues": ref.validate(framed),
        "coffee_issues": ref.validate(coffee),
        "coffee_status": [line for line in ref.render(coffee).splitlines()
                          if line.startswith("Status")][0],
        "severities": len({"off_scope", "forbidden"}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the receipt flags 1 of 3 changed files while the tests stay green",
            all([len(result["changed"]) == 3, result["off_scope"] == ["deploy/release.sh"],
                 result["acceptance_count"] == 2, result["issues"] == []]),
            f"git diff --name-only reports {result['changed']} and "
            f"{result['off_scope']} sits outside the allowed paths; the frame's other "
            "acceptance line is a unittest command that passes either way",
        ),
        practice.Check(
            "FINDING: acceptance is a list of strings and nothing runs them",
            all([result["coffee_issues"] == [], result["coffee_status"] == "Status: READY"]),
            f"a frame whose only acceptance evidence is 'make coffee' returns "
            f"{result['coffee_issues']} and renders {result['coffee_status']!r} -- the "
            "receipt has to be a command someone runs; the frame only promises it was",
        ),
        practice.Check(
            "FINDING: the receipt needs the forbidden globs",
            all([result["forbidden"] == ["deploy/release.sh"], result["severities"] == 2]),
            f"{result['forbidden']} is both outside the allowed exact paths and inside the "
            "forbidden deploy/** glob -- Lesson 38's scope.off_scope and scope.forbidden "
            "are different severities, and comparing against allowed alone collapses them",
        ),
        practice.Check(
            "FINDING: an untracked file is invisible to the receipt as written",
            all([result["untracked_seen"] is True, result["untracked_in_diff"] is False,
                 result["porcelain"] == 4, result["tracked"] == 3]),
            f"git diff --name-only reports {result['tracked']} paths and git status "
            f"--porcelain reports {result['porcelain']}: adding a file nobody asked for is "
            "the violation the obvious command cannot see",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
