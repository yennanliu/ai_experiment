"""Exercise 3 — the non-goals held in five of six commits.

    Add two non-goals that keep the first slice small.

Reading of the exercise: a non-goal is only a boundary if a diff can cross it.
So write two that this repository's own history can be checked against, then
check it.

**ANSWER: "do not translate anything" and "do not touch the tooling" keep the
slice to one directory per lesson, and the second one held in 5 of 6
commits.** Over the six lesson commits ending at the one that shipped Lesson 46, every file lands inside a
lesson's `practice/` directory or is the generated top-level `README.md` --
except one commit, which also edited `scripts/scaffold_practice.py`. That is
**1** crossing, in the place a non-goal is supposed to make visible.

**FINDING: the crossing was correct, and the non-goal is what makes it a
decision.** The scaffolder raised `FileNotFoundError` on lessons that ship
English docs only, so shipping anything at all required the edit. Without the
non-goal it is an invisible extra file in a diff; with it, it is a line in
the commit message and a choice somebody made.

**FINDING: the translation non-goal excludes more than it looks like.** The
reference phase carries **42** Chinese documents and every generated manifest
carries a `zh:` block per exercise; declaring translation a non-goal removes
that surface from the slice entirely rather than deferring it file by file.

**FINDING: `non_goals` is a list of strings and the validator counts it.**
`validate` reports "non-goals are empty" and nothing else, so a frame with
**2** concrete non-goals and one carrying `["nothing"]` are equally valid.
The check that matters -- does the diff stay inside them -- is the one
Exercise 5 of Lesson 43 built out of `git status`, and it lives in the gate,
not in the frame.

Structure: `crossings()` reads the real commits; `frames()` compares a
concrete non-goal list against a vacuous one.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "47-outcomes-before-output"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
BASE = "demos/phases/14-agent-engineering"
INSIDE = re.compile(rf"^{re.escape(BASE)}/\d\d-[a-z0-9-]+/practice/")
NON_GOALS = ["do not translate anything", "do not touch the tooling"]
REFERENCE = Path("/Users/jliu/ai-engineering-from-scratch/phases/14-agent-engineering")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


ANCHOR = "a992f27"  # the lesson-46 commit: a fixed window, not a sliding one


def commits(count=6):
    return git("log", "--format=%H", "-n", str(count), ANCHOR, "--", BASE).split()


def crossings(count=6):
    """Files in the recent lesson commits that fall outside the declared slice."""
    rows = []
    prefix = ROOT.name + "/"
    for sha in commits(count):
        files = [line[len(prefix):] if line.startswith(prefix) else line
                 for line in git("show", "--name-only", "--format=", sha).splitlines()
                 if line.strip()]
        outside = [name for name in files
                   if not INSIDE.match(name) and name != "README.md"]
        rows.append({"sha": sha[:7], "files": len(files), "outside": outside})
    return rows


def frames(ref, non_goals):
    return ref.OutcomeFrame(
        user="a reader working through a phase 14 lesson",
        situation="facing the exercises after reading the lesson",
        current_behavior="guesses at an answer with no way to check it",
        desired_outcome="a reader can run an answer and see it check its own claims",
        constraints=["answers import the lesson's code"], non_goals=non_goals,
        proposed_output="")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = crossings()
    crossed = [row for row in rows if row["outside"]]
    concrete = frames(ref, NON_GOALS)
    vacuous = frames(ref, ["nothing"])
    zh_docs = len(list(REFERENCE.glob("*/docs/zh.md"))) if REFERENCE.exists() else 0
    manifests = sorted((ROOT / "demos" / "phases" / "14-agent-engineering").glob(
        "*/practice/practice.yaml"))
    return {
        "commits": len(rows), "held": len(rows) - len(crossed),
        "crossed": [row["sha"] for row in crossed],
        "outside": sorted({name for row in crossed for name in row["outside"]}),
        "zh_docs": zh_docs,
        "zh_blocks": sum(path.read_text(encoding="utf-8").count("    zh: |")
                         for path in manifests[:42]),
        "concrete_issues": ref.validate(concrete), "vacuous_issues": ref.validate(vacuous),
        "empty_issues": ref.validate(frames(ref, [])),
        "non_goals": len(NON_GOALS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the non-goals held in 5 of 6 commits",
            all([result["commits"] == 6, result["held"] == 5,
                 len(result["crossed"]) == 1,
                 result["outside"] == ["scripts/scaffold_practice.py"]]),
            f"across the {result['commits']} lesson commits ending at the anchor, every file lands inside a "
            f"practice/ directory or the generated README, except {result['crossed']} which "
            f"also edited {result['outside']} -- one crossing, in the place a non-goal makes "
            "visible",
        ),
        practice.Check(
            "FINDING: the crossing was correct, and the non-goal makes it a decision",
            all([result["outside"] == ["scripts/scaffold_practice.py"],
                 len(result["crossed"]) == 1]),
            "the scaffolder raised FileNotFoundError on lessons shipping English docs only, "
            "so shipping at all required the edit; without the non-goal it is an invisible "
            "extra file in a diff, with it a line in the commit message",
        ),
        practice.Check(
            "FINDING: the translation non-goal excludes more than it looks like",
            all([result["zh_docs"] == 42, result["zh_blocks"] > 200]),
            f"the reference phase carries {result['zh_docs']} Chinese documents and the "
            f"generated manifests carry {result['zh_blocks']} zh blocks; declaring "
            "translation a non-goal removes the surface rather than deferring it file by "
            "file",
        ),
        practice.Check(
            "FINDING: non_goals is a list of strings and the validator counts it",
            all([result["concrete_issues"] == [], result["vacuous_issues"] == [],
                 result["empty_issues"] == ["non-goals are empty"],
                 result["non_goals"] == 2]),
            f"{result['non_goals']} concrete non-goals and a single word 'nothing' both "
            f"return {len(result['vacuous_issues'])} issues; only an empty list is refused "
            f"({result['empty_issues']}), and the check that matters lives in the gate",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
