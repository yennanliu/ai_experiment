"""Exercise 1 — the field is refused before the rule can read it.

    Add a `last_human_touch` timestamp. Refuse any agent write within five
    seconds of a human edit.

Reading of the exercise: the rule is three lines and it cannot run, because
`commit` validates before it writes and `validate` rejects any key not in
`properties`. So the schema change comes first, and the interesting question
is where the timestamp lives -- in the state document the agent is about to
overwrite, or beside it.

**ANSWER: a schema property, a 5-second window, and 3 of 8 writes refused.**
On a virtual clock in integer milliseconds, agent writes at **+500ms**,
**+2s** and **+4.9s** after a human edit are refused and those at **+5s**,
**+7s**, **+30s**, **+90s** and **+3600s** go through. Adding the field
without the schema change raises `unexpected fields ['last_human_touch']` on
**1** of **1** commits. The window itself is arbitrary: **1s** refuses
**1** of the probes and **60s** refuses **6**, and none is more correct
without knowing how long a human's editor takes to flush.

**FINDING: storing the timestamp in the document the agent overwrites is a
self-clearing guard.** `commit` writes the whole state, so an agent that
reads at +1s, waits, and commits at +6s writes back the `last_human_touch`
it read -- correct here, and wrong the moment a human edits in between,
because the agent's stale copy silently reinstates an old timestamp. Keeping
it in a sidecar file the agent never writes refuses **1** of **1** such
races that the in-document version allows.

**FINDING: the guard needs a clock and the manager has none.**
`StateManager` takes **2** constructor arguments and `commit` takes **1**,
so the comparison time arrives from the caller -- and a caller that passes
its own `now` can pass any `now`. The refusal is advisory unless the
timestamp is read from the filesystem, which is **1** `stat` call the
manager does not make.

**FINDING: the validator knows 5 of JSON Schema's 7 types.** `_check_type`
handles `object`, `array`, `string`, `integer` and `null`; `number` and
`boolean` have no branch. A float timestamp is therefore inexpressible --
`expected integer, got float` -- so the field has to be integer
milliseconds. Two missing `if`s decide the unit of a field the schema is
supposed to describe.

Structure: `human_touch_schema()` adds the property; `guard()` is the
window, on a clock passed in rather than read.
"""

from __future__ import annotations

import copy
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "34-repo-memory-and-state"
WINDOW = 5000
PROBES = (500, 2000, 4900, 5000, 7000, 30000, 90000, 3600000)
JSON_TYPES = ("object", "array", "string", "integer", "number", "boolean",
              "null")


def base_state(last_touch=None):
    state = {"schema_version": 1, "active_task_id": "T-001",
             "touched_files": [], "assumptions": [], "blockers": [],
             "next_action": "pick next task"}
    if last_touch is not None:
        state["last_human_touch"] = last_touch
    return state


def human_touch_schema(ref):
    schema = copy.deepcopy(ref.STATE_SCHEMA)
    schema["properties"]["last_human_touch"] = {"type": "integer"}
    return schema


def refuses(ref, document, schema):
    try:
        ref.validate(document, schema)
        return None
    except ref.SchemaError as exc:
        return str(exc).split(": ", 1)[-1]


def guard(last_touch, now, window=WINDOW):
    return now - last_touch >= window


def sweep(window=WINDOW, touched_at=100000):
    return [guard(touched_at, touched_at + offset, window) for offset in PROBES]


def sidecar_race(ref, root, schema):
    """A human edit lands between the agent's read and its commit."""
    path, sidecar = root / "agent_state.json", root / ".last_human_touch"
    manager = ref.StateManager(path, schema)
    manager.commit(base_state(last_touch=100000))
    sidecar.write_text("100000")
    stale = manager.load()
    sidecar.write_text("105000")
    stale["next_action"] = "agent step"
    in_document = guard(stale["last_human_touch"], 106000)
    from_sidecar = guard(int(sidecar.read_text()), 106000)
    return {"in_document": in_document, "from_sidecar": from_sidecar}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = pathlib.Path(tempfile.mkdtemp())
    schema = human_touch_schema(ref)
    unpatched = refuses(ref, base_state(last_touch=100000), ref.STATE_SCHEMA)
    patched = refuses(ref, base_state(last_touch=100000), schema)
    floats = refuses(ref, {**base_state(last_touch=0), "last_human_touch": 1.5},
                     schema)
    supported = [t for t in JSON_TYPES
                 if ref._check_type({"object": {}, "array": [], "string": "",
                                     "integer": 1, "number": 1.5,
                                     "boolean": True, "null": None}[t], t)]
    allowed = sweep()
    return {
        "probes": len(PROBES), "allowed": sum(allowed), "refused": sum(not a
                                                                      for a in allowed),
        "verdicts": dict(zip(PROBES, allowed)),
        "unpatched_error": unpatched, "patched_error": patched,
        "window": WINDOW,
        "race": sidecar_race(ref, root, schema),
        "manager_args": ref.StateManager.__init__.__code__.co_argcount - 1,
        "commit_args": ref.StateManager.commit.__code__.co_argcount - 1,
        "reads_stat": "stat" in ref.StateManager.commit.__code__.co_names,
        "narrow": sum(not a for a in sweep(window=1000)),
        "wide": sum(not a for a in sweep(window=60000)),
        "float_error": floats, "supported_types": supported,
        "json_types": len(JSON_TYPES),
    }


def verify(result):
    race = result["race"]
    return [
        practice.Check(
            "ANSWER: a 5-second window refusing 3 of 8 writes",
            all([result["probes"] == 8, result["refused"] == 3,
                 result["allowed"] == 5,
                 result["verdicts"][4900] is False,
                 result["verdicts"][5000] is True,
                 result["unpatched_error"] == "unexpected fields "
                 "['last_human_touch']",
                 result["patched_error"] is None]),
            f"writes at 500ms, 2s and 4.9s after a human edit are refused and the other "
            f"{result['allowed']} go through. Without the schema change the field itself "
            f"is rejected: {result['unpatched_error']!r}, and with it "
            f"{result['patched_error']}",
        ),
        practice.Check(
            "FINDING: storing the timestamp in the document is a self-clearing guard",
            all([race["in_document"] is True, race["from_sidecar"] is False]),
            f"commit writes the whole state, so an agent holding a stale copy reinstates "
            f"the timestamp it read: the in-document guard allows the write "
            f"({race['in_document']}) after a human edit it never saw, where a sidecar "
            f"the agent never writes refuses it ({race['from_sidecar']})",
        ),
        practice.Check(
            "FINDING: the guard needs a clock and the manager has none",
            all([result["manager_args"] == 2, result["commit_args"] == 1,
                 result["reads_stat"] is False]),
            f"StateManager takes {result['manager_args']} constructor arguments and "
            f"commit takes {result['commit_args']}, and commit calls stat "
            f"{result['reads_stat']} times -- so the comparison time arrives from the "
            "caller, and a caller that supplies its own now can supply any now",
        ),
        practice.Check(
            "FINDING: the validator knows 5 of JSON Schema's 7 types",
            all([len(result["supported_types"]) == 5,
                 "number" not in result["supported_types"],
                 "boolean" not in result["supported_types"],
                 result["json_types"] == 7,
                 result["float_error"] == "expected integer, got float"]),
            f"_check_type handles {result['supported_types']} of the "
            f"{result['json_types']} JSON Schema types, so a float timestamp cannot be "
            f"expressed at all -- {result['float_error']!r} -- and the field has to be "
            "integer milliseconds. Two missing branches decide the unit",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
