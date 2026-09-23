"""Exercise 5 — the shipped assertion cannot tell rolled back from never ran.

    Rehearsed-rollback test: design an end-to-end test that runs a real
    workflow, crashes it, and confirms the rollback path fires. What does the
    test assert?

Reading of the exercise: "what does the test assert" is the whole question,
and the module already contains an answer to it -- scenario 4 compares the
balances before and after. That assertion is necessary and not sufficient, and
showing why is the design.

**ANSWER: four assertions, of which the shipped one is the weakest.** A
rehearsed-rollback test must assert (1) the side effect *landed* before the
rollback, (2) the state is restored afterwards, (3) the checkpoint records
`rolled-back`, and (4) the return value says so. The shipped scenario asserts
only (2) -- `balances_before == balances_after` -- which is also **True** for
a run in which nothing happened at all.

**FINDING: the weak assertion passes on an empty run.** Comparing the
database to itself across a workflow that never executed gives **True**,
identically to the real rollback. The test therefore cannot distinguish "the
effect was applied and reversed" from "the effect was never applied", which is
the exact pair exercise 1 shows the engine also cannot distinguish.

**FINDING: assertion (1) needs an observation the module does not make.**
Proving the effect landed means reading the intermediate state, and the
workflow exposes no hook between `persist_transfer` and the verify -- so a
real rehearsal has to instrument the side effect, not the workflow. Measured
from outside: the balance is **1400** at that point and **1500** after,
and only the first of those two numbers is evidence.

**FINDING: the shipped rollback does satisfy (2), (3) and (4).** The
verify-fail scenario restores **3** of **3** database fields, records
`rolled-back` in the checkpoint and returns
`verify-fail-rolled-back`. Adding assertion (1) is the whole delta between
the demonstration and a test -- one observation, and the one that makes the
other three mean something.

Structure: `rehearse()` runs the verify-fail path while sampling the
intermediate state; `empty()` is the run that asserts nothing happened.
"""

from __future__ import annotations

import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "16-checkpoints-rollback"

TEMP = tempfile.mkdtemp(prefix="aiefs-rehearse-")
START = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}
AMOUNT = 100


def fresh(ref, name):
    ref.DB.clear()
    ref.DB.update(START)
    path = os.path.join(TEMP, name)
    if os.path.exists(path):
        os.remove(path)
    return ref.Checkpoint(path)


def rehearse(ref):
    """The verify-fail path, with the intermediate balance sampled."""
    checkpoint = fresh(ref, "rehearse.json")
    sampled = {}
    original = ref.persist_transfer

    def watched(txid, from_acct, to_acct, amount):
        original(txid, from_acct, to_acct, amount)
        sampled["mid"] = ref.DB["balance_A"]

    ref.persist_transfer = watched
    try:
        result = ref.run_transfer(checkpoint, "tx-r", "A", "B", AMOUNT, 200,
                                  inject_verify_fail=True)
    finally:
        ref.persist_transfer = original
    record = checkpoint.load()[ref.key("tx-r")]
    return result, sampled.get("mid"), ref.DB["balance_A"], record["status"]


def empty(ref):
    """A workflow that never runs: the balances still match."""
    fresh(ref, "empty.json")
    before = dict(ref.DB)
    return before == dict(ref.DB), ref.DB["balance_A"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    result, mid, after, status = rehearse(ref)
    unchanged, balance = empty(ref)
    restored = sum(ref.DB[key] == value for key, value in START.items())
    return {
        "assertions": 4,
        "shipped_asserts": ["state restored"],
        "landed": mid,
        "after": after,
        "restored_fields": restored,
        "db_fields": len(START),
        "status": status,
        "result": result,
        "effect_observed": mid is not None and mid != after,
        "empty_passes_weak": unchanged,
        "empty_balance": balance,
        "start_balance": START["balance_A"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four assertions, and the shipped one is the weakest",
            all([result["assertions"] == 4, result["shipped_asserts"] == ["state restored"],
                 result["effect_observed"], result["status"] == "rolled-back",
                 result["result"] == "verify-fail-rolled-back"]),
            f"a rehearsal asserts the effect landed ({result['landed']}), the state is "
            f"restored ({result['after']}), the checkpoint says {result['status']!r} and "
            f"the return says {result['result']!r}; the shipped scenario asserts "
            f"{len(result['shipped_asserts'])} of the {result['assertions']}",
        ),
        practice.Check(
            "FINDING: the weak assertion passes on an empty run",
            all([result["empty_passes_weak"],
                 result["empty_balance"] == result["start_balance"]]),
            f"comparing the database to itself across a workflow that never executed "
            f"gives {result['empty_passes_weak']} at balance "
            f"{result['empty_balance']} -- identical to the real rollback, so the test "
            "cannot tell reversed from never applied",
        ),
        practice.Check(
            "FINDING: assertion (1) needs an observation the module does not make",
            all([result["landed"] == 1400, result["after"] == 1500,
                 result["landed"] != result["after"]]),
            f"the balance is {result['landed']} at the moment the effect lands and "
            f"{result['after']} after the rollback; only the first is evidence, and "
            "reading it means instrumenting the side effect rather than the workflow",
        ),
        practice.Check(
            "FINDING: the shipped rollback does satisfy the other three",
            all([result["restored_fields"] == result["db_fields"] == 3,
                 result["status"] == "rolled-back",
                 result["result"] == "verify-fail-rolled-back"]),
            f"the verify-fail path restores {result['restored_fields']} of "
            f"{result['db_fields']} fields, records {result['status']!r} and returns "
            f"{result['result']!r} -- assertion (1) is the whole delta between a "
            "demonstration and a test",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
