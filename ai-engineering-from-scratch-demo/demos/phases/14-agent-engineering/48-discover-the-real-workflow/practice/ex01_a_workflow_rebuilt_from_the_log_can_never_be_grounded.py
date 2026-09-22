"""Exercise 1 — a workflow rebuilt from the log can never be grounded.

    Reconstruct one workflow from a log without interviewing anyone.

Reading of the exercise: the log available here is the trail of artifacts the
work leaves in the tree -- a manifest, a test module, a generated README.
Reconstructing the workflow from them is the whole exercise, and running the
lesson's audit over the result says something the exercise did not anticipate.

**ANSWER: the artifacts yield an 8-step workflow whose direct-evidence ratio
is 0.0, and the audit calls it `needs-evidence`.** Every step is supported by
an artifact -- a manifest on disk, a generated README, a practice directory --
and `Evidence.direct` is false for all **8**, so `audit` reports
`needs-evidence` on a reconstruction that is entirely correct. "Grounded"
requires a kind of evidence that a log, by definition, does not contain.

**FINDING: the evidence ladder has 4 rungs and the dataclass has a boolean.**
The docs rank direct behaviour, artifact, reported behaviour and inference;
`Evidence.direct` collapses that to true or false, so an artifact and an
inference are stored identically. **3** of the **4** rungs land on `False`
and the ratio that drives the status cannot tell them apart.

**FINDING: the reconstruction is checkable, which is what artifacts buy.**
Each step names a path: **8** of the **8** open in the tree, so every claim in
the reconstruction can be re-checked by someone who doubts it. An interview produces claims; a log produces
claims with addresses.

**FINDING: confidence is stored and never used.** `audit` checks that each
confidence sits in `[0, 1]` and then computes a ratio that ignores the
numbers entirely, so **8** evidence items at 0.99 and **8** at 0.01 produce
the same `direct_evidence_ratio` and the same status.

Structure: `reconstruct()` builds the steps from the log; `receipts()` checks
that each one points at something real.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "48-discover-the-real-workflow"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
BASE = "demos/phases/14-agent-engineering"
SAMPLE = "47-outcomes-before-output"

# (actor, action, source, observation, friction) -- every source is a path or a commit
STEPS = [
    ("author", "scaffolds the manifest from the lesson's exercise list",
     f"{BASE}/{SAMPLE}/practice/practice.yaml", "manifest exists with one entry per exercise",
     "the scaffolder fails on lessons with no Chinese doc"),
    ("author", "writes one solution file per exercise",
     f"{BASE}/{SAMPLE}/practice/ex01_the_request_that_started_this_repository_names_its_output_twice.py",
     "five ex-prefixed files sit beside the manifest", ""),
    ("author", "runs each solution until its checks pass",
     f"{BASE}/{SAMPLE}/practice/tests/test_practice.py", "a test module grades every file",
     "expected values are only known after the first run"),
    ("author", "fills the manifest's verifies threshold",
     f"{BASE}/{SAMPLE}/practice/practice.yaml", "no scaffold placeholder survives", ""),
    ("author", "regenerates the lesson README",
     f"{BASE}/{SAMPLE}/practice/README.md", "the file carries generated markers", ""),
    ("author", "writes the answers section by hand",
     f"{BASE}/{SAMPLE}/practice/README.md", "prose follows the generated block",
     "the numbers must match the graded output"),
    ("author", "ships the lesson as one directory", f"{BASE}/{SAMPLE}/practice",
     "one practice directory per lesson", ""),
    ("author", "regenerates the repository coverage table", "README.md",
     "the top-level README changes in the same commit", ""),
]


def reconstruct(ref):
    """Eight steps, each supported by an artifact rather than an observation."""
    return [ref.WorkflowStep(index, actor, action,
                             (ref.Evidence(source, observation, False, 0.9),), friction)
            for index, (actor, action, source, observation, friction)
            in enumerate(STEPS, 1)]


def receipts():
    """Every source is a path in the tree, so every receipt can be opened."""
    sources = [source for _, _, source, _, _ in STEPS]
    return len(sources), sum((ROOT / source).exists() for source in sources)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    steps = reconstruct(ref)
    report = ref.audit(steps)
    sources, resolving = receipts()
    confident = [ref.WorkflowStep(step.order, step.actor, step.action,
                                  tuple(ref.Evidence(item.source, item.observation,
                                                     item.direct, 0.01)
                                        for item in step.evidence), step.friction)
                 for step in steps]
    auditor = inspect.getsource(ref.audit)
    return {
        "steps": len(steps), "issues": report["issues"],
        "ratio": report["direct_evidence_ratio"], "status": report["status"],
        "friction": len(report["friction_points"]),
        "direct_type": ref.Evidence.__annotations__["direct"],
        "rungs": 4, "on_false": 3,
        "sources": sources, "resolving": resolving,
        "low_ratio": ref.audit(confident)["direct_evidence_ratio"],
        "low_status": ref.audit(confident)["status"],
        "confidence_used": auditor.count("confidence"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the log yields 8 steps at ratio 0.0 and status needs-evidence",
            all([result["steps"] == 8, result["issues"] == [], result["ratio"] == 0,
                 result["status"] == "needs-evidence", result["friction"] == 3]),
            f"the reconstruction has {result['steps']} steps and "
            f"{len(result['issues'])} ordering issues, and every one is supported by an "
            f"artifact, so the ratio is {result['ratio']} and the audit says "
            f"{result['status']!r} about a reconstruction that is correct",
        ),
        practice.Check(
            "FINDING: the ladder has 4 rungs and the dataclass has a boolean",
            all([result["direct_type"] == "bool", result["rungs"] == 4,
                 result["on_false"] == 3]),
            f"Evidence.direct is a {result['direct_type']}, so {result['on_false']} of the "
            f"{result['rungs']} documented rungs -- artifact, reported, inference -- store "
            "identically, and the ratio that drives the status cannot tell them apart",
        ),
        practice.Check(
            "FINDING: the reconstruction is checkable, which is what artifacts buy",
            all([result["sources"] == 8, result["resolving"] == 8]),
            f"every one of the {result['sources']} steps names a path in the tree and "
            f"{result['resolving']} of them open -- an interview produces claims; an "
            "artifact produces claims with addresses",
        ),
        practice.Check(
            "FINDING: confidence is stored and never used",
            all([result["low_ratio"] == result["ratio"],
                 result["low_status"] == result["status"],
                 result["confidence_used"] == 2]),
            f"dropping every confidence to 0.01 leaves the ratio at {result['low_ratio']} "
            f"and the status at {result['low_status']!r}; confidence appears "
            f"{result['confidence_used']} times in audit, both in the range check",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
