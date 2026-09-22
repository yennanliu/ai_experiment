"""Exercise 2 — four reported claims, and the log corroborates two.

    Interview a user and mark every claim that still lacks direct evidence.

Reading of the exercise: the interview available here is the instruction that
started this work -- a person stating how the work should proceed. Each
sentence is a claim about the workflow, and the repository can be asked
whether its artifacts show it.

**ANSWER: 2 of the 4 claims are corroborated by the tree and 2 remain
reported.** "Every exercise ships exactly one solution file" holds -- **10**
finished lessons declare **50** code exercises and carry **50** files. "Every
finished lesson carries a written answers section" holds: **10** of **10**.
"The pull request is the place this lands" and "the reviewer reads the answers
first" have **0** artifacts in this repository, and stay marked as reported.

**FINDING: marking a claim costs one field the dataclass already has.**
`Evidence.direct` distinguishes the corroborated pair from the reported pair,
and the audit's ratio moves from **0.0** to **0.5** when the corroboration is
recorded. The mechanism exists; what the interview adds is the discipline of
entering the claim before looking for the artifact.

**FINDING: a corroborated claim is narrower than the sentence it came from.**
"Complete all of the lessons" is corroborated as "every code exercise has a
file", which is what the tree can show. Whether the answer inside the file is
*right* is a separate check -- that lesson's own audit -- and conflating the
two is how a reported claim quietly becomes a requirement.

**FINDING: the audit reports a ratio and never says which claims are weak.**
`audit` returns **5** keys: a status, the issues, the ratio, the friction
points and the steps. A reader gets one number for the whole workflow and has
to walk `steps` to find the **2** unsupported claims, which is the opposite of
the lesson's advice to keep uncertain claims visible.

Structure: `CLAIMS` is the interview; `corroborate()` asks the tree about each
one.
"""

from __future__ import annotations

import sys
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "48-discover-the-real-workflow"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
sys.path.insert(0, str(ROOT / "demos"))
BASE = ROOT / "demos" / "phases" / "14-agent-engineering"
FINISHED = tuple(f"{number}-" for number in range(43, 53))

# (claim as stated, how the repository would corroborate it)
CLAIMS = [
    ("every exercise ships exactly one solution file", "manifest-and-files"),
    ("every finished lesson carries a written answers section", "readme-sections"),
    ("the pull request is where this lands", "none"),
    ("a reviewer reads the answers before the code", "none"),
]


def lessons():
    return sorted(path for path in BASE.iterdir()
                  if path.is_dir() and path.name.startswith(FINISHED))


def counted():
    """One file per code exercise, and one answers section per lesson."""
    from harness import yamlite
    exercises = files = sections = 0
    for lesson in lessons():
        manifest = yamlite.loads(
            (lesson / "practice" / "practice.yaml").read_text(encoding="utf-8"))
        exercises += sum(row["kind"] == "code" for row in manifest["exercises"])
        files += len(list((lesson / "practice").glob("ex0*.py")))
        readme = (lesson / "practice" / "README.md").read_text(encoding="utf-8")
        sections += "## Answers" in readme
    return {"lessons": len(lessons()), "exercises": exercises, "files": files,
            "sections": sections}


def corroborate():
    rows = counted()
    found = {"manifest-and-files": rows["exercises"] == rows["files"],
             "readme-sections": rows["sections"] == rows["lessons"],
             "none": False}
    return [{"claim": claim, "via": via, "direct": found[via]}
            for claim, via in CLAIMS], rows


def steps(ref, rows):
    return [ref.WorkflowStep(index, "author", row["claim"],
                             (ref.Evidence(row["via"], row["claim"], row["direct"], 0.8),))
            for index, row in enumerate(rows, 1)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, counts = corroborate()
    report = ref.audit(steps(ref, rows))
    reported = [ref.WorkflowStep(step.order, step.actor, step.action,
                                 (ref.Evidence(item.source, item.observation, False, 0.8),))
                for step in steps(ref, rows) for item in step.evidence]
    return {
        "claims": len(rows), "corroborated": sum(row["direct"] for row in rows),
        "supported": [row["claim"] for row in rows if row["direct"]],
        "reported": [row["claim"] for row in rows if not row["direct"]],
        **counts,
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
                 result["lessons"] == 10, result["exercises"] == result["files"] == 50,
                 result["sections"] == 10,
                 result["reported"] == ["the pull request is where this lands",
                                        "a reviewer reads the answers before the code"]]),
            f"{result['corroborated']} of {result['claims']} claims are backed by the "
            f"tree: {result['lessons']} finished lessons declare {result['exercises']} code "
            f"exercises against {result['files']} solution files and carry "
            f"{result['sections']} answers sections. {result['reported']} have no artifact "
            "in this repository",
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
            all([result["supported"][0] == "every exercise ships exactly one solution file",
                 result["exercises"] == result["files"]]),
            "'complete all of the lessons' is corroborated only as 'every code exercise has "
            f"a file' ({result['files']} of {result['exercises']}); whether the answer in "
            "the file is right is the lesson's own audit, and conflating the two turns a "
            "reported claim into a requirement",
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
