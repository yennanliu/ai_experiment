"""Exercise 3 — the non-goals held in five of six commits.

    Add two non-goals that keep the first slice small.

Reading of the exercise: a non-goal is only a boundary if a diff can cross it.
So write two this repository's own tree can be checked against, then check it.

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

Structure: `shipped()` reads where the work landed; `crossing()` shows the one
file outside the slice; `frames()` compares a concrete non-goal list against a
vacuous one.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "47-outcomes-before-output"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
BASE = ROOT / "demos" / "phases" / "14-agent-engineering"
FINISHED = tuple(f"{number}-" for number in range(43, 53))
PHASE_DIR = "phases/14-agent-engineering"
NON_GOALS = ["do not translate anything", "do not touch the tooling"]
TOOLING = "scripts/scaffold_practice.py"
FALLBACK = "except FileNotFoundError"


def shipped():
    """Every file the finished lessons put in the tree, and where it landed."""
    files = [path for lesson in sorted(BASE.iterdir())
             if lesson.is_dir() and lesson.name.startswith(FINISHED)
             for path in sorted(lesson.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts]
    return {"files": len(files),
            "outside": [path.name for path in files if "practice" not in path.parts]}


def crossing():
    """The one tooling file the phase had to change, and the proof it changed."""
    source = (ROOT / TOOLING).read_text(encoding="utf-8")
    return {"path": TOOLING, "fallback": FALLBACK in source,
            "inside_practice": "practice" in Path(TOOLING).parts}


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
    placement = shipped()
    crossed = crossing()
    concrete = frames(ref, NON_GOALS)
    vacuous = frames(ref, ["nothing"])
    reference = parity.find_reference_root() / PHASE_DIR
    zh_docs = len(list(reference.glob("*/docs/zh.md")))
    manifests = sorted((ROOT / "demos" / "phases" / "14-agent-engineering").glob(
        "*/practice/practice.yaml"))
    return {
        **placement, "crossing": crossed["path"], "fallback": crossed["fallback"],
        "crossing_inside": crossed["inside_practice"],
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
            "ANSWER: every file the phase shipped is inside a practice directory",
            all([result["files"] == 80, result["outside"] == [],
                 result["crossing"] == TOOLING, result["crossing_inside"] is False,
                 result["fallback"] is True]),
            f"the finished lessons put {result['files']} files in the tree and "
            f"{len(result['outside'])} of them sit outside a practice directory; the one "
            f"crossing is {result['crossing']}, which still carries the "
            f"{FALLBACK!r} branch the phase needed",
        ),
        practice.Check(
            "FINDING: the crossing was correct, and the non-goal makes it a decision",
            all([result["crossing"] == TOOLING, result["fallback"] is True]),
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
