"""Exercise 1 — the clarification asks for a key the resume path ignores.

    Run `code/main.py`. Trace the full Task lifecycle, including the
    input-required pause where the called agent asks for a clarification.

Reading of the exercise: tracing a happy path confirms the states and little
else, so the pause is answered three ways -- with the key the agent's message
names, with the key its code reads, and with no data part at all. The three
replies do not produce three outcomes, and the one that follows the
clarification literally is the one that goes wrong quietly.

**ANSWER: `submitted` -> `working` -> `input_required` -> `working` ->
`completed`.** A send with no `targetLength` data part pauses with the agent
message *"Please specify target_length as a data part."*, and a reply
carrying the value resumes and produces one artifact.

**FINDING: the message names `target_length` and the code reads
`targetLength`.** Following the clarification literally still *completes* the
task -- because `writer_tasks_reply` takes any data part and falls back to
`payload.get("targetLength", "short")`. Asking for `long` under the name the
agent printed yields a **short** summary, with no error anywhere. The two
spellings differ by one character and by the whole outcome.

**FINDING: the pause requires a key the resume does not.**
`writer_tasks_send` pauses unless `"targetLength" in payload`;
`writer_tasks_reply` resumes on the presence of *any* data part. So the
condition that stopped the task is not the condition that restarts it, and a
default is substituted for the answer that was requested.

**FINDING: a reply with no data part is absorbed silently.** The message is
appended, the state stays `input_required`, and nothing is returned to say
so -- **3** messages on the task and **0** signals. A caller that answers in
prose waits forever.

**FINDING: `state` is a bare string with no transition table.** `Task` has
**4** fields and the module has **0** names describing legal transitions, so
`completed` -> `submitted` is a plain assignment. The lifecycle the exercise
asks you to trace is a convention of two functions, not a structure.

Structure: `pause` opens a task and stops it, and `answer` replies once and
reports the state with whatever length reached the artifact.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "19-a2a-protocol"
STATES = ["submitted", "working", "input_required", "working", "completed"]


def pause(ref):
    """Send with no targetLength, which is what stops the task."""
    message = ref.Message(role="user", parts=[ref.Part("text", {"text": "source"})])
    return ref.writer_tasks_send("draft_report", message)


def answer(ref, task, parts):
    resumed = ref.writer_tasks_reply(task.id, ref.Message(role="user", parts=parts))
    length = None
    if resumed.artifact is not None:
        length = resumed.artifact.parts[0].payload["text"].split()[2]
    return resumed.state, length, len(resumed.messages)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with contextlib.redirect_stdout(io.StringIO()):  # the agents narrate to stdout
        return trace(ref)


def trace(ref):
    fresh = ref.Task(id="t-probe")
    opened = pause(ref)
    clarification = opened.messages[-1].parts[0].payload["text"]

    named = answer(ref, pause(ref), [ref.Part("data", {"target_length": "long"})])
    coded = answer(ref, pause(ref), [ref.Part("data", {"targetLength": "long"})])
    prose = answer(ref, pause(ref), [ref.Part("text", {"text": "make it long"})])

    rewound = pause(ref)
    ref.writer_tasks_reply(rewound.id, ref.Message(
        role="user", parts=[ref.Part("data", {"targetLength": "short"})]))
    completed_state = rewound.state
    rewound.state = "submitted"  # a plain assignment, nothing refuses it
    return {
        "initial": fresh.state, "paused": opened.state,
        "clarification": clarification,
        "asks_for": "target_length" in clarification,
        "named": named, "coded": coded, "prose": prose,
        "task_fields": [f.name for f in dataclasses.fields(ref.Task)],
        "module_names": [n for n in dir(ref) if n.isupper() and not n.startswith("_")],
        "completed_state": completed_state, "rewound": rewound.state,
        "states": STATES,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: submitted, working, input_required, working, completed",
            all([result["initial"] == "submitted", result["paused"] == "input_required",
                 result["coded"][0] == "completed", result["coded"][1] == "long",
                 result["completed_state"] == "completed"]),
            f"a fresh Task is {result['initial']!r}, a send with no targetLength pauses at "
            f"{result['paused']!r} with the message {result['clarification']!r}, and a reply "
            f"carrying the value resumes to {result['coded'][0]!r} with one artifact -- the "
            f"{' -> '.join(result['states'])} the exercise asks for",
        ),
        practice.Check(
            "FINDING: the message names target_length and the code reads targetLength",
            all([result["asks_for"], result["named"][0] == "completed",
                 result["named"][1] == "short", result["coded"][1] == "long"]),
            f"following the clarification literally still completes the task, and asking for "
            f"'long' under the printed name yields a {result['named'][1]!r} summary against "
            f"{result['coded'][1]!r} for the spelling the code reads. One character apart, "
            "and no error anywhere",
        ),
        practice.Check(
            "FINDING: the pause requires a key the resume does not",
            all([result["paused"] == "input_required", result["named"][0] == "completed"]),
            "writer_tasks_send pauses unless 'targetLength' is in the payload, while "
            "writer_tasks_reply resumes on the presence of any data part and defaults the "
            "value. The condition that stopped the task is not the condition that restarts "
            "it, and a default is substituted for the answer that was requested",
        ),
        practice.Check(
            "FINDING: a prose reply is absorbed silently, and state has no transition table",
            all([result["prose"][0] == "input_required", result["prose"][1] is None,
                 result["prose"][2] == 3, len(result["task_fields"]) == 4,
                 result["module_names"] == ["TASK_STORE", "WRITER_AGENT_CARD"],
                 result["rewound"] == "submitted"]),
            f"answering in prose leaves the task at {result['prose'][0]!r} with "
            f"{result['prose'][2]} messages and nothing returned to say so. And Task's "
            f"{len(result['task_fields'])} fields and the module's "
            f"{result['module_names']} include no transition table, so completed -> "
            f"{result['rewound']!r} is a plain assignment",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
