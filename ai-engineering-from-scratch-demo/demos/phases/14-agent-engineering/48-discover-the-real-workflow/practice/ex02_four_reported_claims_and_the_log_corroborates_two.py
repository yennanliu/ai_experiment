"""Exercise 2 — four reported claims, and the log corroborates two.

    Interview a user and mark every claim that still lacks direct evidence.

Reading of the exercise: the interview available here is the instruction that
started this work -- a person stating how the work should proceed. Each
sentence is a claim about the workflow, and the repository can be asked
whether it happened.

**ANSWER: 2 of the 4 claims are corroborated by the log and 2 remain
reported.** "One commit per completed lesson" holds -- the **8** most recent
lesson commits carry **8** distinct lesson numbers. "Push after each lesson"
holds: the branch is an ancestor of its remote. "The pull request is the place
this lands" and "the reviewer reads the answers first" have **0** artifacts in
this repository, and stay marked as reported.

**FINDING: marking a claim costs one field the dataclass already has.**
`Evidence.direct` distinguishes the corroborated pair from the reported pair,
and the audit's ratio moves from **0.0** to **0.5** when the corroboration is
recorded. The mechanism exists; what the interview adds is the discipline of
entering the claim before looking for the artifact.

**FINDING: a corroborated claim is narrower than the sentence it came from.**
"Commit and push once a lesson is completed" is corroborated as "one commit
whose subject names one lesson", which is what the log can show. Whether the
lesson was *completed* is a separate check -- the audit of that lesson -- and
conflating the two is how a reported claim quietly becomes a requirement.

**FINDING: the audit reports a ratio and never says which claims are weak.**
`audit` returns **5** keys: a status, the issues, the ratio, the friction
points and the steps. A reader gets one number for the whole workflow and has
to walk `steps` to find the **2** unsupported claims, which is the opposite of
the lesson's advice to keep uncertain claims visible.

Structure: `CLAIMS` is the interview; `corroborate()` asks the log about each
one.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "48-discover-the-real-workflow"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
BASE = "demos/phases/14-agent-engineering"

# (claim as stated, how the log would corroborate it)
CLAIMS = [
    ("one commit per completed lesson", "commit-subjects"),
    ("push after each lesson rather than at the end", "remote-ancestry"),
    ("the pull request is where this lands", "none"),
    ("a reviewer reads the answers before the code", "none"),
]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def lesson_numbers(count=8):
    subjects = git("log", "--format=%s", "-n", str(count), "--", BASE).splitlines()
    return [match.group(1) for subject in subjects
            if (match := re.search(r"lesson (\d+)", subject))]


def pushed():
    """Whether the branch this work lives on is an ancestor of its remote."""
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    remotes = git("branch", "-r", "--contains", "HEAD~1")
    return branch, bool(remotes.strip())


def corroborate():
    numbers = lesson_numbers()
    branch, on_remote = pushed()
    found = {"commit-subjects": len(numbers) == len(set(numbers)) and len(numbers) == 8,
             "remote-ancestry": on_remote, "none": False}
    return [{"claim": claim, "via": via, "direct": found[via]} for claim, via in CLAIMS], \
        numbers, branch


def steps(ref, rows):
    return [ref.WorkflowStep(index, "author", row["claim"],
                             (ref.Evidence(row["via"], row["claim"], row["direct"], 0.8),))
            for index, row in enumerate(rows, 1)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, numbers, branch = corroborate()
    report = ref.audit(steps(ref, rows))
    reported = [ref.WorkflowStep(step.order, step.actor, step.action,
                                 (ref.Evidence(item.source, item.observation, False, 0.8),))
                for step in steps(ref, rows) for item in step.evidence]
    return {
        "claims": len(rows), "corroborated": sum(row["direct"] for row in rows),
        "supported": [row["claim"] for row in rows if row["direct"]],
        "reported": [row["claim"] for row in rows if not row["direct"]],
        "lessons": len(numbers), "distinct": len(set(numbers)),
        "branch": branch,
        "ratio": report["direct_evidence_ratio"],
        "unmarked_ratio": ref.audit(reported)["direct_evidence_ratio"],
        "keys": sorted(report),
        "names_weak": any("claim" in key or "weak" in key for key in report),
        "status": report["status"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 2 of the 4 claims are corroborated and 2 remain reported",
            all([result["claims"] == 4, result["corroborated"] == 2,
                 result["lessons"] == 8, result["distinct"] == 8,
                 result["reported"] == ["the pull request is where this lands",
                                        "a reviewer reads the answers before the code"]]),
            f"{result['corroborated']} of {result['claims']} claims are backed by the log: "
            f"{result['lessons']} recent lesson commits carry {result['distinct']} distinct "
            f"lesson numbers and the branch is on its remote. {result['reported']} have no "
            "artifact in this repository",
        ),
        practice.Check(
            "FINDING: marking a claim costs one field the dataclass already has",
            all([result["ratio"] == 0.5, result["unmarked_ratio"] == 0.0]),
            f"recording the corroboration moves the ratio from {result['unmarked_ratio']} "
            f"to {result['ratio']}; the mechanism exists, and what the interview adds is "
            "entering the claim before looking for the artifact",
        ),
        practice.Check(
            "FINDING: a corroborated claim is narrower than the sentence it came from",
            all([result["supported"][0] == "one commit per completed lesson",
                 result["distinct"] == 8]),
            "'commit and push once a lesson is completed' is corroborated only as 'one "
            f"commit whose subject names one lesson' ({result['distinct']} distinct); "
            "whether the lesson was completed is the lesson's own audit, and conflating "
            "the two turns a reported claim into a requirement",
        ),
        practice.Check(
            "FINDING: the audit reports a ratio and never says which claims are weak",
            all([len(result["keys"]) == 5, result["names_weak"] is False,
                 result["status"] == "grounded"]),
            f"audit returns {result['keys']} -- one number for the whole workflow at status "
            f"{result['status']!r} -- so a reader has to walk steps to find the "
            f"{len(result['reported'])} unsupported claims",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
