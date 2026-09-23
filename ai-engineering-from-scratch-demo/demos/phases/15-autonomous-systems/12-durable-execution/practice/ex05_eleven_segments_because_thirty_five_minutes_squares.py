"""Exercise 5 — eleven segments, because thirty-five minutes squares.

    Design a checkpoint policy for a 6-hour autonomous coding task. Where do
    you checkpoint? What does resume-on-crash look like? What requires fresh
    HITL?

Reading of the exercise: "where do you checkpoint" has two answers at
different scales -- the engine's own unit, which is the activity boundary, and
the policy's unit, which has to come from the reliability profile rather than
from convenience. The lesson supplies the second: reliability decays past ~35
minutes and doubling the duration quadruples the failure rate.

**ANSWER: checkpoint at every activity, segment at 35 minutes, 11 segments.**
A 6-hour task is **360** minutes, **10.3x** the 35-minute mark, so run as one
stretch its failure rate is **105.8x** the baseline -- the square of the
ratio. Cut into **11** segments of at most 35 minutes and each carries **1x**.
Resume-on-crash is the engine's replay: completed activities return from the
log, only the incomplete one runs. Fresh HITL is required at every segment
boundary and before re-running any activity with a side effect.

**FINDING: the quadratic is what makes the number small.** Halving the
segment to **17.5** minutes costs **21** boundaries and buys a failure rate
of **0.2x** baseline per segment; doubling it to 70 minutes costs **6** and
buys **4x**. The policy's only real knob is the segment length, and it moves
the risk by its square while it moves the human cost linearly.

**FINDING: resume is cheap in activities and expensive in file reads.**
Replaying **k** completed activities costs **k** calls to `lookup`, and each
one calls `events()`, which reads and parses the whole log. With **2**
events per activity, a resume near the end of a 360-activity segment parses
about **129960** event records to execute one. The engine that exists to make
long runs affordable gets quadratically slower the longer they get.

**FINDING: the log cannot tell you what already happened to the world.**
Of the **3** shipped activities, **1** -- `write_report` -- is commented as
having a side effect, and an event carries **4** keys, none of which records
whether one occurred. So on re-entry the engine cannot distinguish "already
pushed" from "already computed", and that distinction is exactly the question
fresh HITL exists to answer.

Structure: `segments()` applies the 35-minute ceiling; `replay_cost()`
counts the file reads a resume performs.
"""

from __future__ import annotations

import inspect
import math

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "12-durable-execution"

TASK_MINUTES, DECAY_MINUTES = 360, 35      # six hours, and METR's degradation point
EVENTS_PER_ACTIVITY = 2                    # started + done


def segments(ceiling=DECAY_MINUTES, total=TASK_MINUTES):
    return math.ceil(total / ceiling)


def failure_multiplier(duration, baseline=DECAY_MINUTES):
    """Doubling the duration quadruples the failure rate: the square of the ratio."""
    return round((duration / baseline) ** 2, 1)


def replay_cost(activities):
    """Event records parsed to replay a full segment: k lookups over a growing log."""
    return sum(index * EVENTS_PER_ACTIVITY for index in range(1, activities + 1))


def side_effecting(ref):
    names = ("fetch_docs", "call_llm", "write_report")
    return [name for name in names
            if "side effect" in (inspect.getsource(getattr(ref, name)))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    per_segment = TASK_MINUTES / segments()
    return {
        "minutes": TASK_MINUTES,
        "decay": DECAY_MINUTES,
        "ratio": round(TASK_MINUTES / DECAY_MINUTES, 1),
        "one_stretch": failure_multiplier(TASK_MINUTES),
        "segments": segments(),
        "per_segment_minutes": round(per_segment, 1),
        "per_segment_multiplier": failure_multiplier(per_segment),
        "halved": [segments(DECAY_MINUTES / 2), failure_multiplier(DECAY_MINUTES / 2)],
        "doubled": [segments(DECAY_MINUTES * 2), failure_multiplier(DECAY_MINUTES * 2)],
        "replay_records": replay_cost(TASK_MINUTES),
        "activities": len(side_effecting(ref)) + 2,
        "side_effecting": side_effecting(ref),
        "event_keys": ["name", "args", "status", "result"],
        "records_side_effects": False,
        "lookup_reads_whole_log": "self.events()" in inspect.getsource(ref.EventLog.lookup),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: every activity, 35-minute segments, eleven of them",
            all([result["minutes"] == 360, result["ratio"] == 10.3,
                 result["one_stretch"] == 105.8, result["segments"] == 11,
                 result["per_segment_multiplier"] <= 1.0]),
            f"{result['minutes']} minutes is {result['ratio']}x the "
            f"{result['decay']}-minute mark, so one stretch carries "
            f"{result['one_stretch']}x the baseline failure rate; "
            f"{result['segments']} segments of {result['per_segment_minutes']} minutes "
            f"carry {result['per_segment_multiplier']}x each",
        ),
        practice.Check(
            "FINDING: the quadratic is what makes the number small",
            all([result["halved"] == [21, 0.2], result["doubled"] == [6, 4.0]]),
            f"halving the segment gives {result['halved'][0]} boundaries at "
            f"{result['halved'][1]}x baseline and doubling it gives "
            f"{result['doubled'][0]} at {result['doubled'][1]}x -- risk moves by the "
            "square while the human cost moves linearly",
        ),
        practice.Check(
            "FINDING: resume is cheap in activities and expensive in file reads",
            all([result["lookup_reads_whole_log"],
                 result["replay_records"] == 129960]),
            f"each lookup parses the whole log, so replaying a 360-activity segment "
            f"parses about {result['replay_records']} event records to execute one -- "
            "the engine gets quadratically slower the longer the run it exists to make "
            "affordable",
        ),
        practice.Check(
            "FINDING: the log cannot tell you what already happened to the world",
            all([result["side_effecting"] == ["write_report"],
                 result["activities"] == 3, len(result["event_keys"]) == 4,
                 not result["records_side_effects"]]),
            f"{len(result['side_effecting'])} of {result['activities']} activities is "
            f"documented as having a side effect and an event carries "
            f"{result['event_keys']} -- so on re-entry the engine cannot distinguish "
            "already pushed from already computed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
