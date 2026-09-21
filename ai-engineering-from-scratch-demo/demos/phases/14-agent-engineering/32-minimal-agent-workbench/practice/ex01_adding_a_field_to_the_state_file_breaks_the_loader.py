"""Exercise 1 — adding a field to the state file breaks the loader.

    Add a `last_run` timestamp to `agent_state.json`. Refuse to run if the
    file is older than 24 hours unless an operator confirms.

Reading of the exercise: `load_state` does `AgentState(**raw)`, so the state
file's schema is the dataclass's signature exactly -- adding one key makes
every older reader raise. That is the first thing a timestamp costs, and it
is worth paying before the staleness rule, because the rule is four lines and
the migration is the part that breaks a running system.

**ANSWER: a `last_run` field, a 24-hour rule, and an operator confirmation
that is the only way through.** On a virtual clock, a state written at hour
**0** and read at hour **23** runs; read at hour **25** it refuses; read at
hour **25** with `confirm=True` it runs and rewrites `last_run`. Over a
**5**-reading sweep at hours 1, 23, 24, 25 and 200, the unconfirmed rule
allows **3** and refuses **2**.

**FINDING: `AgentState(**raw)` makes the schema a hard contract.** Writing
the new file and reading it with the shipped loader raises `TypeError` on
**1** of **1** attempts -- `unexpected keyword argument 'last_run'`. The
lesson calls state "the system of record", and a system of record that
cannot be read by last week's code is a system of record for one version.
Filtering to known fields on load costs **1** line and keeps both readers
working.

**FINDING: the boundary is 24 hours and nothing says which clock.** The file
stores an instant and the rule compares it against "now", so the refusal
depends on the reader's clock, not on anything in the workbench. Two readers
**2** hours apart disagree about the same file at hour **25** -- **1**
refuses and **1** allows. Storing a monotonic run counter alongside the
timestamp makes the comparison local.

**FINDING: there is no operator channel to confirm through.** `run_one_turn`
takes **2** arguments and `AgentState` has **5** fields, none of them an
approval; the confirmation has to arrive as a third argument the shipped
signature does not have. Recording it costs a sixth field -- otherwise the
next session cannot tell a confirmed stale run from a fresh one.

Structure: `TimedState` carries the new field; `gate()` is the staleness
rule on a virtual clock.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "32-minimal-agent-workbench"
STALE_HOURS = 24
READINGS = (1, 23, 24, 25, 200)


def write_state(path, state, last_run_hour):
    payload = {"active_task_id": state.active_task_id,
               "touched_files": list(state.touched_files),
               "assumptions": list(state.assumptions),
               "blockers": list(state.blockers),
               "next_action": state.next_action,
               "last_run": last_run_hour}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_tolerant(ref, path):
    """One line more than load_state: drop keys the dataclass does not know."""
    raw = json.loads(path.read_text())
    known = set(ref.AgentState.__dataclass_fields__)
    return ref.AgentState(**{k: v for k, v in raw.items() if k in known}), raw


def gate(raw, now_hour, confirmed=False, stale=STALE_HOURS):
    age = now_hour - raw["last_run"]
    if age > stale and not confirmed:
        return {"allowed": False, "age": age,
                "reason": f"state is {age}h old, over {stale}h; operator must confirm"}
    return {"allowed": True, "age": age,
            "reason": "confirmed stale run" if age > stale else "fresh"}


def strict_load(ref, path):
    try:
        ref.load_state(path)
        return None
    except TypeError as exc:
        return str(exc).split("__init__() ")[-1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        path = root / "agent_state.json"
        write_state(path, ref.AgentState(active_task_id="T-001"), last_run_hour=0)
        strict = strict_load(ref, path)
        state, raw = load_tolerant(ref, path)
        sweep = {hour: gate(raw, hour)["allowed"] for hour in READINGS}
        confirmed = gate(raw, 25, confirmed=True)
        skewed = {"reader_a": gate(raw, 25)["allowed"],
                  "reader_b": gate(raw, 23)["allowed"]}
    return {
        "readings": len(READINGS), "sweep": sweep,
        "allowed": sum(sweep.values()), "refused": sum(not v for v in sweep.values()),
        "strict_error": strict,
        "tolerant_task": state.active_task_id,
        "raw_keys": sorted(raw),
        "state_fields": list(ref.AgentState.__dataclass_fields__),
        "confirmed": confirmed,
        "skewed": skewed,
        "turn_args": ref.run_one_turn.__code__.co_argcount,
        "approval_fields": [f for f in ref.AgentState.__dataclass_fields__
                            if "confirm" in f or "approv" in f or "run" in f],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a 24-hour rule refusing 2 of 5 readings, confirmable",
            all([result["readings"] == 5, result["allowed"] == 3,
                 result["refused"] == 2,
                 result["sweep"] == {1: True, 23: True, 24: True, 25: False,
                                     200: False},
                 result["confirmed"]["allowed"] is True,
                 result["confirmed"]["age"] == 25]),
            f"over readings at hours {list(result['sweep'])} the rule allows "
            f"{result['allowed']} and refuses {result['refused']}; at hour "
            f"{result['confirmed']['age']} an operator confirmation lets it through "
            f"({result['confirmed']['reason']!r})",
        ),
        practice.Check(
            "FINDING: AgentState(**raw) makes the schema a hard contract",
            all([result["strict_error"] is not None,
                 "last_run" in result["strict_error"],
                 result["tolerant_task"] == "T-001",
                 len(result["raw_keys"]) == 6,
                 len(result["state_fields"]) == 5]),
            f"the shipped load_state raises {result['strict_error']!r} on a file with "
            f"{len(result['raw_keys'])} keys against {len(result['state_fields'])} "
            "dataclass fields. Filtering to known fields on load costs one line and lets "
            "last week's reader keep working",
        ),
        practice.Check(
            "FINDING: the boundary is 24 hours and nothing says which clock",
            all([result["skewed"] == {"reader_a": False, "reader_b": True}]),
            f"the file stores an instant and the rule compares it against the reader's "
            f"now, so two readers two hours apart disagree about the same file: "
            f"{result['skewed']}. A monotonic run counter stored beside the timestamp "
            "makes the comparison local to the workbench",
        ),
        practice.Check(
            "FINDING: there is no operator channel to confirm through",
            all([result["turn_args"] == 2, result["approval_fields"] == [],
                 len(result["state_fields"]) == 5]),
            f"run_one_turn takes {result['turn_args']} arguments and AgentState has "
            f"{len(result['state_fields'])} fields, "
            f"{len(result['approval_fields'])} of them an approval -- so the "
            "confirmation arrives as an argument the signature lacks, and recording it "
            "needs a sixth field",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
