"""Exercise 2 — the earliest layer for this incident is not one of the five.

    Name the earliest layer that can prevent each recurrence.

Reading of the exercise: "earliest" means the layer at which the failure
stops being possible, not the layer that noticed. For the two real signals
one of those layers is on the lesson's list and the other is not.

**ANSWER: the complaint's earliest layer is `policy`, and the incident's is
the fixture the solution reads -- which is none of the 5 destinations.** The
pull-request complaint is a permission the repository owner grants, and the
router agrees. The incident was a solution measuring "the last six commits":
the test suite caught it, but the layer that makes it impossible is the
solution pinning its window to a fixed commit, and **0** of the router's
**5** destinations describe a fixture.

**FINDING: the five durable artifacts do not exist in this repository.**
`promote` assigns one path per destination -- `evaluations/regression-suite.json`,
`policies/authority-boundaries.json` and three others -- and **0** of the **5**
are present in the tree. The ratchet promotes every signal into a filesystem
nobody has created.

**FINDING: the layer that caught it and the layer that owns it are one step
apart, and the artifact records only one.** The test suite is where the
incident surfaced; the solutions are where it was fixed -- **3** of them now
pin a commit. `RatchetAction` has **7** fields and none distinguishes
detection from prevention, so a reader cannot tell whether a control stops a
failure or merely reports it.

**FINDING: routing by keyword puts the cheapest layer last.** The router
tests `evaluation`, `policy`, `context`, `runtime` and then falls through to
`backlog`, so a signal that matches nothing becomes a backlog item -- the
most expensive destination and the only one with no verification a machine
can run. Of the **5** evidence strings, **0** name a command.

Structure: `LAYERS` is the hand assignment; `artifacts()` checks the
destinations against the tree.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "54-build-the-feedback-ratchet"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
BASE = ROOT / "demos" / "phases" / "14-agent-engineering"

SIGNALS = [
    ("commit 448f198", "a later commit changed a measurement three solutions depended on",
     4, 3, "practice-maintainer", 90),
    ("project memory", "permission denied opening the pull request from the agent",
     2, 2, "repository-owner", 180),
]
# hand-assigned earliest layer, and whether the router can name it
LAYERS = [("the solution's own fixture", False), ("policy", True)]


def build(ref, row):
    return ref.Signal(*row)


def artifacts(ref):
    """Each destination's durable artifact, and whether it exists here."""
    rows = {}
    for destination in ("evaluation", "policy", "context", "runtime", "backlog"):
        signal = ref.Signal("s", {"evaluation": "regression", "policy": "unsafe",
                                  "context": "missing context", "runtime": "timeout",
                                  "backlog": "new need"}[destination], 3, 1, "o", 30)
        action = ref.promote(signal)
        rows[action.destination] = (action.durable_artifact,
                                    (ROOT / action.durable_artifact).exists())
    return rows


FINISHED = tuple(f"{number}-" for number in range(43, 53))


def anchored():
    """Solutions in the finished lessons that pin their git window to a fixed commit."""
    return sorted(path.parents[1].name[:2] for path in BASE.glob("*/practice/ex0*.py")
                  if path.parents[1].name.startswith(FINISHED)
                  and "ANCHOR = " in path.read_text(encoding="utf-8"))


def evidence(ref):
    """One verification string per destination, and how many name a command."""
    rows = {}
    for text in ("regression", "unsafe", "missing context", "timeout", "new need"):
        action = ref.promote(ref.Signal("s", text, 3, 1, "o", 30))
        rows[action.destination] = action.verification_evidence
    return {"evidences": len(rows),
            "commands": sum(value.startswith(("python", "uv ", "pytest"))
                            for value in rows.values())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    actions = [ref.promote(build(ref, row)) for row in SIGNALS]
    paths = artifacts(ref)
    return {
        **evidence(ref),
        "routed": [action.destination for action in actions],
        "hand": [name for name, _ in LAYERS],
        "nameable": sum(ok for _, ok in LAYERS),
        "destinations": len(paths),
        "existing": sum(exists for _, exists in paths.values()),
        "artifact_names": sorted(path for path, _ in paths.values()),
        "anchored": anchored(),
        "action_fields": list(ref.RatchetAction.__dataclass_fields__),
        "detection_field": any(name in ref.RatchetAction.__dataclass_fields__
                               for name in ("detected_by", "prevents", "layer_kind")),
        "fallthrough": inspect.getsource(ref.destination).strip().endswith(
            'return "backlog"'),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: policy for the complaint, and a fixture for the incident",
            all([result["routed"] == ["backlog", "policy"],
                 result["hand"] == ["the solution's own fixture", "policy"],
                 result["nameable"] == 1, result["destinations"] == 5]),
            f"the router answers {result['routed']} and the earliest layers are "
            f"{result['hand']}; {result['nameable']} of the two is nameable among the "
            f"{result['destinations']} destinations, because none of them describes a "
            "fixture",
        ),
        practice.Check(
            "FINDING: the five durable artifacts do not exist in this repository",
            all([result["existing"] == 0, result["destinations"] == 5,
                 "policies/authority-boundaries.json" in result["artifact_names"]]),
            f"{result['existing']} of the {result['destinations']} assigned paths are "
            f"present in the tree -- {result['artifact_names'][:2]} and three more -- so "
            "every signal is promoted into a filesystem nobody has created",
        ),
        practice.Check(
            "FINDING: detection and prevention are one step apart and only one is recorded",
            all([result["anchored"] == ["45", "47", "48"],
                 len(result["action_fields"]) == 7,
                 result["detection_field"] is False]),
            f"the test suite caught the incident and the fix lives in the solutions -- "
            f"lessons {result['anchored']} now pin a commit -- while RatchetAction's "
            f"{len(result['action_fields'])} fields cannot say whether a control prevents a "
            "failure or merely reports it",
        ),
        practice.Check(
            "FINDING: routing by keyword puts the cheapest layer last",
            all([result["fallthrough"] is True, result["commands"] == 0,
                 result["evidences"] == 5]),
            f"the router falls through to backlog, the most expensive destination, and "
            f"{result['commands']} of the {result['evidences']} verification strings name a "
            "command a machine could run",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
