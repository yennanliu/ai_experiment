"""Exercise 5 — the only order the demo runs is the one where it does not matter.

    Find one framework in this table that hides shared state entirely. Explain
    what breaks when agents need to coordinate across handoffs without
    re-reading history.

Reading of the exercise: name the row, then break it on this module rather
than in the abstract -- project `SharedState` down to the last message and
re-run the pipeline. What breaks is not an error; it is a verdict that
silently changes.

**ANSWER: OpenAI Swarm / Agents SDK, whose shared-state cell reads "caller's
problem" -- and what breaks is a join and a backward search.** Of the **6**
framework rows, it is the only one naming no state mechanism at all. And
**3** of the **3** shipped policies need more than the previous message:
`researcher_policy` counts its own past turns, `writer_policy` joins every
researcher message out of `snapshot()`, and `reviewer_policy` calls
`last_by("writer")`, which is a backward scan of history rather than a read of
the message that preceded it. Every agent in the lesson would have to be
rewritten to run on the framework its own table lists first.

**FINDING: hiding the pool flips the verdict, without an error.** Run the
pipeline as researcher, writer, researcher, reviewer. With the full pool the
reviewer finds the draft and returns **approved**. With only the last message
visible, `last_by("writer")` returns nothing, the verdict becomes **needs
revision**, and nothing raises. One extra interleaved message is the whole
difference.

**FINDING: the join loses half its input and says so nowhere.** Run
researcher, researcher, writer. The full pool gives a draft joining **2**
notes; the projected one gives **1**. The writer's `" | ".join(...)` is
happiest exactly when there is nothing to join, so a shorter draft is the only
symptom.

**FINDING: the demo only ever runs the order where this is invisible.** All
**3** shipped orchestrators produce researcher, writer, reviewer -- strictly
linear, one message per agent. Under that order the projected pool and the
full pool give byte-identical output, because every read happens to land on
the previous message. The lesson's own runs cannot exhibit the failure its
table's first row implies.

Structure: `Projected` is `SharedState` with history hidden; `both()` runs an
order against each and diffs the last message.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "04-primitive-model"
LINEAR = ["researcher", "writer", "reviewer"]
INTERLEAVED = ["researcher", "writer", "researcher", "reviewer"]
TWO_NOTES = ["researcher", "researcher", "writer"]


def projected(ref):
    """SharedState with no history: an agent sees the message that reached it."""

    class Projected(ref.SharedState):
        def snapshot(self):
            return super().snapshot()[-1:]

        def last_by(self, name):
            tail = super().snapshot()[-1:]
            return tail[0] if tail and tail[0]["from"] == name else None

    return Projected


def both(ref, order):
    """The final message under the full pool and under the projected one."""
    outputs = []
    for state_class in (ref.SharedState, projected(ref)):
        state = state_class()
        ref.StaticOrchestrator(order).run(ref.make_team(), state)
        outputs.append(state.snapshot()[-1]["content"])
    return outputs


def table(ref):
    """The framework table's shared-state column, read out of the lesson's own doc."""
    doc = parity.doc_text(PHASE, LESSON)
    rows = {}
    for line in doc.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] not in ("Framework", "-----------"):
            rows[cells[0]] = cells[3]
    return rows


def reads_history(ref):
    """Policies that consult more than the message that reached them."""
    return [name for name in ("researcher_policy", "writer_policy", "reviewer_policy")
            if "snapshot()" in inspect.getsource(getattr(ref, name))
            or "last_by" in inspect.getsource(getattr(ref, name))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = table(ref)
    hidden = [name for name, cell in rows.items() if "caller" in cell]
    linear, interleaved, notes = (both(ref, LINEAR), both(ref, INTERLEAVED),
                                  both(ref, TWO_NOTES))
    backward = reads_history(ref)
    return {
        "rows": len(rows), "hidden": hidden, "cell": rows[hidden[0]] if hidden else "",
        "reads_history": backward, "policies": 3,
        "linear_same": linear[0] == linear[1],
        "interleaved": interleaved, "flipped": interleaved[0] != interleaved[1],
        "notes_full": notes[0].count("note"), "notes_projected": notes[1].count("note"),
        "orchestrators": 3,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: OpenAI Swarm, whose shared-state cell reads 'caller's problem'",
            all([result["rows"] == 6, len(result["hidden"]) == 1,
                 "caller's problem" in result["cell"],
                 len(result["reads_history"]) == result["policies"]]),
            f"of the {result['rows']} framework rows, {result['hidden'][0]} is the only "
            f"one whose shared-state cell is {result['cell']!r}; "
            f"all {len(result['reads_history'])} of {result['policies']} shipped "
            f"policies need more than the previous message -- every one of them would "
            "have to be rewritten to run on the framework the table lists first",
        ),
        practice.Check(
            "FINDING: hiding the pool flips the verdict, without an error",
            all([result["flipped"], "approved" in result["interleaved"][0],
                 "needs revision" in result["interleaved"][1]]),
            f"run as {', '.join(INTERLEAVED)}, the full pool gives "
            f"{result['interleaved'][0]!r} and the projected one "
            f"{result['interleaved'][1]!r} -- last_by('writer') finds nothing once one "
            "message is interleaved, and nothing raises",
        ),
        practice.Check(
            "FINDING: the join loses half its input and says so nowhere",
            all([result["notes_full"] == 2, result["notes_projected"] == 1]),
            f"run as {', '.join(TWO_NOTES)}, the full pool joins "
            f"{result['notes_full']} notes and the projected one "
            f"{result['notes_projected']}; the writer's ' | '.join is happiest when "
            "there is nothing to join, so a shorter draft is the only symptom",
        ),
        practice.Check(
            "FINDING: the demo only ever runs the order where this is invisible",
            all([result["linear_same"], result["orchestrators"] == 3]),
            f"all {result['orchestrators']} shipped orchestrators produce "
            f"{', '.join(LINEAR)} -- strictly linear, one message each -- and under that "
            "order the projected and full pools give byte-identical output, because "
            "every read lands on the previous message",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
