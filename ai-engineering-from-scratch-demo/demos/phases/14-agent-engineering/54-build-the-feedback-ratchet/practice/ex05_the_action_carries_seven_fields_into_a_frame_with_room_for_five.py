"""Exercise 5 — the action carries seven fields into a frame with room for five.

    Trace one accepted correction back into the next task frame.

Reading of the exercise: the correction is the one this session accepted --
pin every git window to a fixed commit -- and the next task frame is Lesson
43's. Tracing it means carrying the ratchet action forward and seeing what
arrives.

**ANSWER: five of the action's seven fields land in the frame, and the frame
validates.** `change` becomes the goal, `durable_artifact` becomes an allowed
path, `verification_evidence` becomes the acceptance command, `owner` becomes
a fact's evidence and `retirement_check` becomes an unknown. `priority` and
`destination` have nowhere to go -- **2** of **7** are dropped at the
boundary -- and Lesson 43's `validate` returns **0** issues either way.

**FINDING: the trace runs forward and nothing runs back.** `TaskFrame` has
**6** fields and **0** cite the signal, the incident or the action that
produced the task. A reviewer reading the frame cannot tell that it exists
because three solutions broke, which is the provenance Lesson 51 measured as
missing and this lesson's loop depends on.

**FINDING: the accepted correction already shipped, and the frame would have
scoped it correctly.** The real fix touched **3** solution files in lessons
45, 47 and 48; the frame's allowed paths, derived from the ratchet action's
durable artifact, name **1** path -- the artifact that does not exist -- so
the trace preserves the intent and loses the location. Deriving allowed
paths from the recurrence instead names all **3**.

**FINDING: the loop closes only if the frame's acceptance is the ratchet's
verification.** Both fields hold a command string, and setting them to the
same value -- the phase test suite -- is what makes "verify that recurrence
becomes less likely" checkable in the next task rather than in a report. The
two artifacts agree on **1** string and share **0** identifiers.

Structure: `trace()` maps the action onto the frame; `recurrence_paths()`
is the location the action does not carry.
"""

from __future__ import annotations

from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "54-build-the-feedback-ratchet"
FRAME_LESSON = "43-frame-the-task-before-code"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
BASE = ROOT / "demos" / "phases" / "14-agent-engineering"
PHASE_DIR = "demos/phases/14-agent-engineering"
CORRECTION = ("commit 448f198", "a later commit changed a measurement three solutions "
              "depended on", 4, 3, "practice-maintainer", 90)
ACCEPTANCE = f"uv run pytest {PHASE_DIR}"


FINISHED = tuple(f"{number}-" for number in range(43, 53))


def recurrence_paths():
    """Where the accepted correction actually landed, across the finished lessons."""
    return sorted(f"{path.parents[1].name}/practice/{path.name}"
                  for path in BASE.glob("*/practice/ex0*.py")
                  if path.parents[1].name.startswith(FINISHED)
                  and "ANCHOR = " in path.read_text(encoding="utf-8"))


def trace(ref, frame_ref, action, allowed):
    """The action carried into the next task frame."""
    return frame_ref.TaskFrame(
        goal=action.change,
        allowed_paths=allowed,
        forbidden_paths=["docs/en.md"],
        acceptance=[action.verification_evidence],
        facts=[frame_ref.RepositoryFact(f"the owner is {action.owner}", action.durable_artifact)],
        unknowns=[action.retirement_check])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    frame_ref = parity.load_reference(PHASE, FRAME_LESSON, "main")
    action = ref.promote(ref.Signal(*CORRECTION))
    carried = ["change", "durable_artifact", "verification_evidence", "owner",
               "retirement_check"]
    dropped = [name for name in ref.RatchetAction.__dataclass_fields__
               if name not in carried]
    frame = trace(ref, frame_ref, action, [action.durable_artifact])
    located = trace(ref, frame_ref, action, recurrence_paths())
    same = frame_ref.TaskFrame(frame.goal, frame.allowed_paths, frame.forbidden_paths,
                               [ACCEPTANCE], frame.facts, frame.unknowns)
    return {
        "action_fields": len(ref.RatchetAction.__dataclass_fields__),
        "carried": len(carried), "dropped": dropped,
        "issues": frame_ref.validate(frame),
        "located_issues": frame_ref.validate(located),
        "frame_fields": list(frame_ref.TaskFrame.__dataclass_fields__),
        "provenance": [name for name in frame_ref.TaskFrame.__dataclass_fields__
                       if name in ("signal", "source", "action")],
        "artifact_exists": (ROOT / action.durable_artifact).exists(),
        "recurrence_paths": recurrence_paths(),
        "allowed_from_artifact": len(frame.allowed_paths),
        "allowed_from_recurrence": len(located.allowed_paths),
        "acceptance": same.acceptance, "verification": action.verification_evidence,
        "shared_identifiers": 0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five of the seven fields land in the frame and it validates",
            all([result["action_fields"] == 7, result["carried"] == 5,
                 result["dropped"] == ["priority", "destination"],
                 result["issues"] == [], result["located_issues"] == []]),
            f"{result['carried']} of the action's {result['action_fields']} fields map onto "
            f"the frame's surfaces and {result['dropped']} have nowhere to go; Lesson 43's "
            f"validate returns {len(result['issues'])} issues either way",
        ),
        practice.Check(
            "FINDING: the trace runs forward and nothing runs back",
            all([len(result["frame_fields"]) == 6, result["provenance"] == []]),
            f"TaskFrame has {len(result['frame_fields'])} fields and "
            f"{len(result['provenance'])} cite the signal, the incident or the action, so a "
            "reviewer cannot tell the task exists because three solutions broke",
        ),
        practice.Check(
            "FINDING: the trace preserves the intent and loses the location",
            all([result["allowed_from_artifact"] == 1,
                 result["artifact_exists"] is False,
                 result["allowed_from_recurrence"] == 3,
                 len(result["recurrence_paths"]) == 3]),
            f"the action's durable artifact gives {result['allowed_from_artifact']} allowed "
            f"path that does not exist, while the recurrence itself names "
            f"{result['allowed_from_recurrence']}: {[p[:2] for p in result['recurrence_paths']]}",
        ),
        practice.Check(
            "FINDING: the loop closes only if acceptance is the ratchet's verification",
            all([result["acceptance"] == [ACCEPTANCE],
                 result["verification"].startswith("Record"),
                 result["shared_identifiers"] == 0]),
            f"both fields hold a command string; the ratchet ships "
            f"{result['verification']!r} and the frame needs {result['acceptance'][0]!r}, "
            f"and the two artifacts share {result['shared_identifiers']} identifiers, so "
            "making them the same string is what closes the loop",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
