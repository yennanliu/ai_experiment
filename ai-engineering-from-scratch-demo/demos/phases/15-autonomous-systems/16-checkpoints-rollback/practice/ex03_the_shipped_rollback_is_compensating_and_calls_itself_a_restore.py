"""Exercise 3 — the shipped rollback is compensating and calls itself a restore.

    Design a rollback plan for a specific production action (e.g., "post to a
    Slack channel"). Classify as in-band, compensating, or out-of-band.
    Justify the choice.

Reading of the exercise: the three classes are distinguished by *what the
observer can see afterwards*, not by how the code is written, so the design
states that test and applies it to the chosen action -- and then to the
module's own `rollback_transfer`, which turns out to be classified wrong.

**ANSWER: out-of-band, because a Slack delete is in-band for the channel and
not for the reader.** `chat.delete` removes the message from the channel
history, so a reader arriving afterwards sees the prior state -- in-band. But
anyone present when it posted has the notification, the email digest and
possibly a screenshot, and none of those is reachable. The plan is therefore:
delete (best-effort in-band), post a correction naming the error
(compensating), and page a human (out-of-band) -- **3** steps, of which only
the third is a guarantee.

**FINDING: the shipped `rollback_transfer` is compensating, not a restore.**
Its own comment calls it "restore balances and the prior transfer id", and it
works by adding back what it subtracted -- a second write, visible in any
audit log as two transfers rather than none. Measured: after a rollback the
balances match the starting state in **3** of **3** fields, and the database
has been written **2** times. Identical final state, different history.

**FINDING: the distinction is invisible to the module because it has no
history.** `DB` holds **3** keys and the checkpoint stores **1** record per
transaction, so nothing here can express "the balance returned to 1500 having
been 1400 in between". The class an action belongs to is a property of the
observer, and this observer has no memory.

**FINDING: the no-op class has no representation at all.** The lesson says a
no-op rollback "must be named in the proposal", and the checkpoint record
carries **6** fields with **0** for a rollback plan of any kind -- so the one
class that requires stronger HITL is the one the artifact cannot record.
Lesson 15's `Proposal` has the field and this module, which is where the
rollback actually fires, does not.

Structure: `PLAN` is the design; `classify()` applies the observer test, and
the module's own rollback is run through it.
"""

from __future__ import annotations

import inspect
import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "16-checkpoints-rollback"

TEMP = tempfile.mkdtemp(prefix="aiefs-rollback-")
START = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}
# (step, class, is it a guarantee)
PLAN = (
    ("chat.delete the message", "in-band", False),
    ("post a correction naming the error", "compensating", False),
    ("page the on-call human", "out-of-band", True),
)


def classify(state_restored, history_restored):
    """The observer test: what someone arriving afterwards can see."""
    if state_restored and history_restored:
        return "in-band"
    if state_restored:
        return "compensating"
    return "out-of-band"


def rollback_run(ref):
    """Run the module's own rollback and compare state against the start."""
    ref.DB.clear()
    ref.DB.update(START)
    path = os.path.join(TEMP, "rb.json")
    if os.path.exists(path):
        os.remove(path)
    checkpoint = ref.Checkpoint(path)
    ref.run_transfer(checkpoint, "tx-rb", "A", "B", 100, 200, inject_verify_fail=True)
    matched = sum(ref.DB[key] == value for key, value in START.items())
    return matched, len(START)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    matched, fields = rollback_run(ref)
    source = inspect.getsource(ref.rollback_transfer)
    record_fields = ("status", "txid", "from_acct", "to_acct", "amount",
                     "prior_last_transfer_id")
    return {
        "steps": len(PLAN),
        "classes": [kind for _step, kind, _g in PLAN],
        "guarantees": [step for step, _k, guaranteed in PLAN if guaranteed],
        "restored_fields": matched,
        "db_fields": fields,
        "writes_on_rollback": source.count("DB["),
        "calls_itself_restore": "restore" in source.lower(),
        "shipped_class": classify(matched == fields, False),
        "db_keys": len(ref.DB),
        "record_fields": len(record_fields),
        "rollback_plan_fields": [name for name in record_fields if "rollback" in name],
        "history_keys": [key for key in ref.DB if "history" in key or "log" in key],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: out-of-band, because delete is in-band for the channel only",
            all([result["steps"] == 3,
                 result["classes"] == ["in-band", "compensating", "out-of-band"],
                 result["guarantees"] == ["page the on-call human"]]),
            f"the plan is {result['steps']} steps across {result['classes']}, of which "
            f"{len(result['guarantees'])} is a guarantee -- deletion clears the channel "
            "and not the notifications, the digest or a screenshot",
        ),
        practice.Check(
            "FINDING: the shipped rollback is compensating, not a restore",
            all([result["restored_fields"] == result["db_fields"] == 3,
                 result["writes_on_rollback"] == 3,
                 result["calls_itself_restore"],
                 result["shipped_class"] == "compensating"]),
            f"after the rollback {result['restored_fields']} of "
            f"{result['db_fields']} fields match the start, reached by "
            f"{result['writes_on_rollback']} further writes -- identical final state, "
            "two transfers in any audit log",
        ),
        practice.Check(
            "FINDING: the distinction is invisible to the module",
            all([result["db_keys"] == 3, result["history_keys"] == []]),
            f"DB holds {result['db_keys']} keys and "
            f"{len(result['history_keys'])} of them is a history, so nothing here can "
            "express that the balance returned to 1500 having been 1400 in between",
        ),
        practice.Check(
            "FINDING: the no-op class has no representation at all",
            all([result["record_fields"] == 6, result["rollback_plan_fields"] == []]),
            f"the checkpoint record carries {result['record_fields']} fields and "
            f"{len(result['rollback_plan_fields'])} is a rollback plan -- so the one "
            "class the lesson says must be named in the proposal is the one this "
            "artifact cannot record",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
