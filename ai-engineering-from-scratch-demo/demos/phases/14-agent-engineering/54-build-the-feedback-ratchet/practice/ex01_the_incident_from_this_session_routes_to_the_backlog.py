"""Exercise 1 — the incident from this session routes to the backlog.

    Turn one incident and one user complaint into ratchet actions.

Reading of the exercise: both inputs are real. The incident happened while
these solutions were being written -- a later commit changed a number three
earlier solutions measured, and the test suite went red. The complaint is the
one recorded in this project's memory: the agent cannot open the pull request
it is told to open.

**ANSWER: both promote, and the incident lands in the backlog while the
complaint lands in policy.** The incident's observation contains none of the
router's **14** keywords, so `destination` falls through to `backlog`; the
complaint contains "permission" and routes to `policy`. The layer that
actually caught the incident was the test suite -- `evaluation` -- so the
router is wrong about the one signal that had already been resolved when it
was written.

**FINDING: priority multiplies severity by frequency, so the worst signal
ranks last.** In the lesson's own example a production write at severity
**5** and frequency **1** scores **5**, below a false positive at severity
**3** seen **4** times, which scores **12**. The backlog sorts descending, so
the only signal about authority is the last row.

**FINDING: the router reads the wording, and the wording is written after the
fix.** "A later commit changed a measurement three solutions depended on"
describes the cause; adding the word "regression" -- true, and how a reviewer
would phrase it -- moves the same signal from `backlog` to `evaluation`
without changing a fact. **2** of the **5** destinations are reachable from
one honest rewording of this incident.

**FINDING: `promote` accepts an owner it never checks.** `Signal.owner`
becomes `RatchetAction.owner` verbatim, so `"platform"`, `""` and
`"somebody"` all produce actions. Of the **7** fields on the action, the one
that makes it an action rather than an observation is the one with no
validation.

Structure: `SIGNALS` carries the two real inputs; `routed()` compares the
router's answer with the layer that actually owned each cause.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "54-build-the-feedback-ratchet"

# (source, observation, severity, frequency, owner, expires, the layer that owned it)
SIGNALS = [
    ("commit 448f198", "a later commit changed a measurement three solutions depended on",
     4, 3, "practice-maintainer", 90, "evaluation"),
    ("project memory", "permission denied opening the pull request from the agent",
     2, 2, "repository-owner", 180, "policy"),
]
REWORDED = "a regression: a later commit changed a measurement three solutions depended on"


def build(ref, row):
    return ref.Signal(row[0], row[1], row[2], row[3], row[4], row[5])


def routed(ref):
    rows = []
    for row in SIGNALS:
        action = ref.promote(build(ref, row))
        rows.append({"observation": row[1][:28], "routed": action.destination,
                     "owned": row[6], "priority": action.priority,
                     "agrees": action.destination == row[6]})
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = routed(ref)
    router = inspect.getsource(ref.destination)
    shipped = ref.backlog(ref.example())
    reworded = ref.destination(ref.Signal(SIGNALS[0][0], REWORDED, 4, 3, "x", 90))
    blank_owner = ref.promote(ref.Signal("s", "unavailable dependency", 2, 1, "", 30))
    return {
        "signals": len(rows), "routed": [row["routed"] for row in rows],
        "owned": [row["owned"] for row in rows],
        "agrees": sum(row["agrees"] for row in rows),
        "destinations": router.count("return"),
        "priorities": [action.priority for action in shipped],
        "worst_last": shipped[-1].destination,
        "worst_priority": shipped[-1].priority,
        "reworded": reworded,
        "owner_fields": [name for name in ref.RatchetAction.__dataclass_fields__
                         if name == "owner"],
        "action_fields": len(ref.RatchetAction.__dataclass_fields__),
        "blank_owner": blank_owner.owner,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the incident routes to backlog and the complaint to policy",
            all([result["signals"] == 2, result["routed"] == ["backlog", "policy"],
                 result["owned"] == ["evaluation", "policy"], result["agrees"] == 1]),
            f"the two real signals route to {result['routed']} against the layers that "
            f"actually owned them, {result['owned']} -- {result['agrees']} of "
            f"{result['signals']} agree, and the miss is the incident whose cause the test "
            "suite had already caught",
        ),
        practice.Check(
            "FINDING: priority multiplies severity by frequency, so the worst ranks last",
            all([result["priorities"] == [12, 12, 5], result["worst_last"] == "policy",
                 result["worst_priority"] == 5]),
            f"the lesson's own example sorts to {result['priorities']}: a production write "
            f"at severity 5 seen once scores {result['worst_priority']} and lands last, "
            f"below a false positive seen four times, so the only {result['worst_last']} "
            "row is the bottom of the backlog",
        ),
        practice.Check(
            "FINDING: the router reads the wording, and the wording is written after the fix",
            all([result["reworded"] == "evaluation", result["routed"][0] == "backlog",
                 result["destinations"] == 5]),
            f"adding the word 'regression' -- true, and how a reviewer would phrase it -- "
            f"moves the same incident from {result['routed'][0]!r} to "
            f"{result['reworded']!r} without changing a fact",
        ),
        practice.Check(
            "FINDING: promote accepts an owner it never checks",
            all([result["blank_owner"] == "", result["action_fields"] == 7,
                 result["owner_fields"] == ["owner"]]),
            f"a signal with an empty owner promotes to an action whose owner is "
            f"{result['blank_owner']!r}; of the {result['action_fields']} fields on the "
            "action, the one that makes it an action rather than an observation is the one "
            "with no validation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
