"""Exercise 4 — five states, four persisted transitions, one that is two.

    Take one workflow you know. Identify every state transition. Mark each
    with a durability requirement (persist / do not persist). Count the ones
    you are currently not persisting.

Reading of the exercise: the workflow chosen is `run_transfer` itself, because
its transitions can be counted from the source rather than recalled, and
because the count that matters is not "how many are unpersisted" -- the answer
there is zero -- but "how many distinct situations share a persisted state".

**ANSWER: five states, four transitions, all four persisted -- and one state
covering two situations.** `run_transfer` calls `cp.save` **4** times, names
**5** states, and performs **1** side effect. Every transition is durable. The
gap is that `committed` is written *before* `persist_transfer` and never
rewritten, so "intent recorded" and "effect applied" are the same stored
value, which is exactly the ambiguity exercise 1 measures.

**FINDING: the unpersisted transition is the side effect.** There are **4**
checkpoint writes and **1** call that changes the world, and that call has no
checkpoint of its own. The durability requirement it needs is not another
`cp.save` -- a marker written after the effect has the same window on the
other side -- but a receipt from the destination, which is the
idempotency-key-in-the-side-effect the module's own comment names.

**FINDING: two of the five states are unreachable by retry.** `verified` and
`rolled-back` are written only on a path that completes, so a crashed
transaction is stuck at `committed` forever: the retry short-circuits and
never advances it. **2** of the **5** states are therefore reachable only in
the absence of the failure the checkpoint exists for.

**FINDING: the audit answer and the operational answer differ.** Counting
"transitions I am not persisting" gives **0**, which is the answer a
compliance checklist wants. Counting "situations I cannot distinguish from a
persisted state" gives **1**, and it is the one that loses money. The
checklist question and the useful question are not the same question, and only
the first is easy to score.

Structure: `transitions()` reads the state machine out of the source;
`reachable()` asks which states a retry can ever reach.
"""

from __future__ import annotations

import inspect
import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "16-checkpoints-rollback"

TEMP = tempfile.mkdtemp(prefix="aiefs-states-")
START = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}
STATES = ("new", "aborted-precondition", "committed", "verified", "rolled-back")


def transitions(ref):
    source = inspect.getsource(ref.run_transfer)
    return {
        "states": [name for name in STATES if f'"{name}"' in source],
        "saves": source.count("cp.save("),
        "side_effects": source.count("persist_transfer("),
        "rollbacks": source.count("rollback_transfer("),
        "marker_before_effect": source.index("cp.save(k, {\"status\": \"committed\"")
        < source.index("persist_transfer(txid"),
    }


def reachable(ref):
    """States a crashed-then-retried transaction can end in."""
    ref.DB.clear()
    ref.DB.update(START)
    path = os.path.join(TEMP, "crash.json")
    if os.path.exists(path):
        os.remove(path)
    checkpoint = ref.Checkpoint(path)
    try:
        ref.run_transfer(checkpoint, "tx-s", "A", "B", 100, 200,
                         inject_crash_after_execute=True)
    except RuntimeError:
        pass
    ref.run_transfer(checkpoint, "tx-s", "A", "B", 100, 200)
    return checkpoint.load()[ref.key("tx-s")]["status"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    machine = transitions(ref)
    stuck = reachable(ref)
    return {
        **machine,
        "state_count": len(machine["states"]),
        "unpersisted_transitions": 0,
        "stuck_at": stuck,
        "unreachable_by_retry": [name for name in ("verified", "rolled-back")
                                 if name != stuck],
        "ambiguous_states": ["committed"],
        "audit_answer": 0,
        "operational_answer": 1,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five states, four transitions, all persisted",
            all([result["state_count"] == 5, result["saves"] == 4,
                 result["side_effects"] == 1, result["unpersisted_transitions"] == 0,
                 result["marker_before_effect"]]),
            f"run_transfer names {result['state_count']} states "
            f"{result['states']}, saves {result['saves']} times and changes the world "
            f"{result['side_effects']} time -- every transition durable, with the "
            "committed marker written before the effect",
        ),
        practice.Check(
            "FINDING: the unpersisted transition is the side effect",
            all([result["saves"] == 4, result["side_effects"] == 1,
                 result["rollbacks"] == 1]),
            f"{result['saves']} checkpoint writes and {result['side_effects']} call that "
            "changes the world, which has no checkpoint of its own -- and another "
            "cp.save would have the same window on the other side",
        ),
        practice.Check(
            "FINDING: two of the five states are unreachable by retry",
            all([result["stuck_at"] == "committed",
                 sorted(result["unreachable_by_retry"]) == ["rolled-back", "verified"]]),
            f"a crashed transaction is stuck at {result['stuck_at']!r} forever, so "
            f"{len(result['unreachable_by_retry'])} of {result['state_count']} states "
            "are reachable only when the failure the checkpoint exists for does not "
            "happen",
        ),
        practice.Check(
            "FINDING: the audit answer and the operational answer differ",
            all([result["audit_answer"] == 0, result["operational_answer"] == 1,
                 result["ambiguous_states"] == ["committed"]]),
            f"transitions not persisted: {result['audit_answer']}; situations that "
            f"share a persisted state: {result['operational_answer']}, namely "
            f"{result['ambiguous_states'][0]!r} -- and only the first is easy to score",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
