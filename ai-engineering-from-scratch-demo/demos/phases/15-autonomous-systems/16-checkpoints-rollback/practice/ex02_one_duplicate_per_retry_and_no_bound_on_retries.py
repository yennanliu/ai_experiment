"""Exercise 2 — one duplicate per retry, and no bound on retries.

    Modify the "mark as done first, then do it" pattern so the status write
    fires after the action. Rerun the crash scenario. Measure how many
    duplicate actions fire.

Reading of the exercise: "how many" invites the answer 1, from one retry. The
number that matters is the rate -- the reordered pattern produces one
duplicate per attempt with nothing bounding attempts -- so the same crash is
retried 1, 2 and 3 times and the balance is read after each.

**ANSWER: one duplicate per retry.** With the status write moved after
`persist_transfer`, a crash between them leaves **no** record, so every retry
re-executes: **1** retry moves **200**, **2** move **300**, **3** move
**400**, against the **100** the shipped order moves however many times it is
retried. Nothing in the workflow caps attempts.

**FINDING: the reorder trades a silent zero for a loud multiple.** The
shipped order's adjacent-crash failure is a transfer that never happens and
reports success; the reordered one's is a transfer that happens repeatedly and
reports success each time. Both are single-instruction windows. The reorder
does not remove the window, it moves which side of the ledger absorbs it --
and the shipped comment says as much, naming the destination-enforced
idempotency key that actually closes it.

**FINDING: the precondition is what bounds the damage, not the checkpoint.**
Re-running the reordered crash until the balance falls below `min_balance`
stops at **13** transfers, because `balance_A` starts at **1500**, the floor
is **200** and each transfer moves **100**. The only thing that ends an
unbounded retry loop here is a business rule that happens to be checked first.

**FINDING: the verify step cannot see a duplicate.** It compares
`DB["last_transfer_id"]` to `txid`, and a second identical transfer sets the
same id -- so after **3** duplicate executions verify still returns true. The
post-action read confirms that *a* transfer landed, never that *one* did.

Structure: `reordered()` is the pattern the exercise asks for; `attempts()`
retries the same crash a chosen number of times.
"""

from __future__ import annotations

import inspect
import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "16-checkpoints-rollback"

TEMP = tempfile.mkdtemp(prefix="aiefs-reorder-")
START = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}
AMOUNT, FLOOR = 100, 200


def fresh(ref, name):
    ref.DB.clear()
    ref.DB.update(START)
    path = os.path.join(TEMP, name)
    if os.path.exists(path):
        os.remove(path)
    return ref.Checkpoint(path)


def reordered(ref, checkpoint, txid, crash=True):
    """`run_transfer` with the status write after the side effect."""
    key = ref.key(txid)
    record = checkpoint.load().get(key, {"status": "new"})
    if record["status"] in ("committed", "verified"):
        return "idempotent-skip"
    if ref.DB[f"balance_A"] - AMOUNT < FLOOR:
        return "aborted-precondition"
    ref.persist_transfer(txid, "A", "B", AMOUNT)
    if crash:
        return "crashed-before-marker"
    checkpoint.save(key, {"status": "committed", "txid": txid})
    return "ok"


def attempts(ref, retries, name):
    checkpoint = fresh(ref, name)
    for _attempt in range(retries + 1):
        reordered(ref, checkpoint, "tx-reorder")
    return START["balance_A"] - ref.DB["balance_A"]


def shipped_attempts(ref, retries, name):
    checkpoint = fresh(ref, name)
    try:
        ref.run_transfer(checkpoint, "tx-shipped", "A", "B", AMOUNT, FLOOR,
                         inject_crash_after_execute=True)
    except RuntimeError:
        pass
    for _attempt in range(retries):
        ref.run_transfer(checkpoint, "tx-shipped", "A", "B", AMOUNT, FLOOR)
    return START["balance_A"] - ref.DB["balance_A"]


def until_precondition(ref, name):
    checkpoint = fresh(ref, name)
    moved = 0
    while reordered(ref, checkpoint, "tx-loop") != "aborted-precondition":
        moved += 1
    return moved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reorder = [attempts(ref, retries, f"r{retries}.json") for retries in (1, 2, 3)]
    shipped = [shipped_attempts(ref, retries, f"s{retries}.json") for retries in (1, 2, 3)]
    bounded = until_precondition(ref, "loop.json")
    fresh(ref, "verify.json")
    for _duplicate in range(3):
        ref.persist_transfer("tx-dup", "A", "B", AMOUNT)
    return {
        "reordered": reorder,
        "shipped": shipped,
        "duplicates_per_retry": reorder[1] - reorder[0],
        "attempt_cap": None,
        "bounded_at": bounded,
        "start": START["balance_A"], "floor": FLOOR, "amount": AMOUNT,
        "headroom": (START["balance_A"] - FLOOR) // AMOUNT,
        "verify_compares": "last_transfer_id" in inspect.getsource(ref.run_transfer),
        "verify_after_duplicates": ref.DB["last_transfer_id"] == "tx-dup",
        "moved_by_duplicates": START["balance_A"] - ref.DB["balance_A"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one duplicate per retry, against a flat 100 for the shipped order",
            all([result["reordered"] == [200, 300, 400],
                 result["shipped"] == [100, 100, 100],
                 result["duplicates_per_retry"] == 100]),
            f"with the marker written after the effect, 1, 2 and 3 retries move "
            f"{result['reordered']} against {result['shipped']} for the shipped order -- "
            f"{result['duplicates_per_retry']} more per attempt, and nothing caps "
            "attempts",
        ),
        practice.Check(
            "FINDING: the reorder trades a silent zero for a loud multiple",
            all([result["attempt_cap"] is None, result["reordered"][0] > result["shipped"][0]]),
            "the shipped order's adjacent-crash failure is a transfer that never "
            "happens and reports success; the reordered one's is a transfer that "
            "happens repeatedly and reports success each time",
        ),
        practice.Check(
            "FINDING: the precondition is what bounds the damage",
            all([result["bounded_at"] == 13, result["headroom"] == 13]),
            f"retrying until the balance hits the floor stops at "
            f"{result['bounded_at']} transfers -- {result['start']} down to "
            f"{result['floor']} at {result['amount']} each -- so a business rule ends "
            "the loop, not the checkpoint",
        ),
        practice.Check(
            "FINDING: the verify step cannot see a duplicate",
            all([result["verify_compares"], result["verify_after_duplicates"],
                 result["moved_by_duplicates"] == 300]),
            f"verify compares last_transfer_id to txid, and duplicates set the same id "
            f"-- after {result['moved_by_duplicates'] // result['amount']} executions it "
            "still matches, so the read confirms that a transfer landed and never that "
            "one did",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
