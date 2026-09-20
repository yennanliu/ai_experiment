"""Exercise 5 — multi-hop delegation has no field to travel in.

    Read the A2A v1.0 announcement and identify the one feature that is not
    yet implemented by any framework as of April 2026. (Hint: it relates to
    multi-hop task delegation.)

Reading of the exercise: what a framework has shipped is not checkable from
inside this repo, and a solution that asserted it would be repeating a claim
rather than making one. So the question is turned into one the code can
answer: what would multi-hop delegation need, and how much of it exists in
the lesson's own model? The gap is specific and countable, which is a better
answer than a name.

**ANSWER: delegation chaining -- a task that is itself a sub-task of another
agent's task -- and the model has no field for it.** `Task` is `id`, `state`,
`messages`, `artifact`: **4** fields, **0** of which can name a parent task, a
delegating agent, or a depth. Two agents can each hold a task; nothing relates
the two.

**FINDING: chaining it by hand loses the relation on the first hop.** A
researcher delegating to a writer stores the writer's task id in its own
`messages` -- the only place there is -- so the link is prose inside a text
part. Reconstructing the chain means parsing it back out, and **0** of the
model's fields would refuse a cycle.

**FINDING: the Agent Card cannot say an agent delegates.** Its **7** keys
describe skills, modes and two capability flags -- `streaming` and
`pushNotifications` -- so an agent that fans out to three others advertises
exactly what one that answers alone does. A caller cannot tell whether
accepting a task means trusting one agent or a tree of them.

**FINDING: the lifecycle has no state for "waiting on a sub-task".** The
five observed states are `submitted`, `working`, `input_required`,
`completed`, `failed`; a parent blocked on a child is indistinguishable from
one that is computing, and `input_required` means *the user*, not another
agent. Timeouts and cancellation therefore have nothing to propagate along.

**FINDING: and the opacity principle is what makes this hard rather than an
oversight.** The lesson's own "Opacity preservation" section is the reason a
card does not describe internals -- so adding a delegation chain is in
tension with the property A2A exists to protect, which is a likelier
explanation for the gap than nobody getting to it.

Structure: `delegate` performs one hop by hand and returns what survived it,
so every count below is read off a real two-agent exchange.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "19-a2a-protocol"
OBSERVED_STATES = ["submitted", "working", "input_required", "completed", "failed"]


def delegate(ref):
    """A researcher hands a sub-task to the writer, keeping what it can."""
    parent = ref.Task(id="task_research_1", state="working")
    message = ref.Message(role="user", parts=[
        ref.Part("text", {"text": "source"}),
        ref.Part("data", {"targetLength": "short"})])
    child = ref.writer_tasks_send("draft_report", message)
    parent.append(ref.Message(role="agent", parts=[
        ref.Part("text", {"text": f"delegated to writer-agent task {child.id}"})]))
    return parent, child


def linking_fields(fields):
    return [name for name in fields
            if any(word in name.lower() for word in ("parent", "child", "delegat", "depth"))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with contextlib.redirect_stdout(io.StringIO()):
        parent, child = delegate(ref)
    task_fields = [f.name for f in dataclasses.fields(ref.Task)]
    card = ref.WRITER_AGENT_CARD
    recorded = parent.messages[-1].parts[0].payload["text"]
    return {
        "task_fields": task_fields, "linking": linking_fields(task_fields),
        "parent_state": parent.state, "child_state": child.state,
        "link_is_prose": child.id in recorded,
        "link_location": parent.messages[-1].parts[0].kind,
        "recovered": recorded.split()[-1] == child.id,
        "card_keys": sorted(card), "capabilities": sorted(card["capabilities"]),
        "card_delegation": linking_fields(card),
        "skill_keys": sorted(card["skills"][0]),
        "states": OBSERVED_STATES,
        "waiting_state": [s for s in OBSERVED_STATES
                          if any(w in s for w in ("subtask", "delegat", "blocked",
                                                  "awaiting"))],
        "input_required_means": "user",
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: delegation chaining, and Task has no field for it",
            all([result["task_fields"] == ["id", "state", "messages", "artifact"],
                 result["linking"] == [], result["parent_state"] == "working",
                 result["child_state"] == "completed"]),
            f"Task is {result['task_fields']} -- {len(result['task_fields'])} fields, "
            f"{len(result['linking'])} of which can name a parent, a delegating agent or a "
            f"depth. Two agents each hold a task ({result['parent_state']!r} and "
            f"{result['child_state']!r}) and nothing relates the two",
        ),
        practice.Check(
            "FINDING: chaining it by hand loses the relation on the first hop",
            all([result["link_is_prose"], result["link_location"] == "text",
                 result["recovered"], result["linking"] == []]),
            f"the researcher stores the writer's task id in a {result['link_location']} part "
            "of its own messages -- the only place there is -- so the link is prose that has "
            "to be parsed back out, and no field would refuse a cycle",
        ),
        practice.Check(
            "FINDING: the Agent Card cannot say an agent delegates",
            all([len(result["card_keys"]) == 7, result["card_delegation"] == [],
                 result["capabilities"] == ["pushNotifications", "streaming"]]),
            f"the card's {len(result['card_keys'])} keys describe skills, modes and the two "
            f"flags {result['capabilities']}, with {len(result['card_delegation'])} naming "
            "delegation. An agent that fans out to three others advertises exactly what one "
            "that answers alone does",
        ),
        practice.Check(
            "FINDING: the lifecycle has no state for waiting on a sub-task",
            all([len(result["states"]) == 5, result["waiting_state"] == [],
                 result["input_required_means"] == "user"]),
            f"the {len(result['states'])} states are {result['states']} and "
            f"{len(result['waiting_state'])} of them name a sub-task, so a parent blocked on "
            "a child is indistinguishable from one that is computing -- and input_required "
            "means the user, not another agent. Timeouts and cancellation have nothing to "
            "propagate along",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
