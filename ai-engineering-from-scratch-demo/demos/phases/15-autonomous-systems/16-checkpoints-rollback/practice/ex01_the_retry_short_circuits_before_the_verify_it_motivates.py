"""Exercise 1 — the retry short-circuits before the verify it motivates.

    Run `code/main.py`. Verify the four scenarios. For the crash-during-commit
    case, confirm the action fires exactly once across retries.

Reading of the exercise: "exactly once" is the claim to test, and it is true
for the crash the module injects and false for the crash one line earlier --
which the module's own comment says. Both are run, because a guarantee that
holds for one of two adjacent crash points is a different guarantee.

**ANSWER: exactly once for the injected crash.** Crashing after
`persist_transfer` leaves the record at `committed` and the balance already
moved; the retry returns **idempotent-skip** and the balance is unchanged --
**1** transfer across two attempts.

**FINDING: the adjacent crash gives exactly zero.** Crash between `cp.save`
and `persist_transfer` and the record still says `committed` while no money
moved. The retry short-circuits on that record and returns
**idempotent-skip** with the database **unchanged**. The marker records an
intention and is read as an outcome, so one instruction's difference in crash
timing turns "exactly once" into "never, silently".

**FINDING: the retry never reaches verify.** `committed` is one of the **4**
terminal states, so the short-circuit returns before the post-action read --
and the record stays `committed` rather than advancing to `verified`. The
scenario that motivates a verify step is the one scenario in which it does not
run, which is why the zero-transfer case is silent rather than caught.

**FINDING: the four scenarios share one database.** `DB` is a module global
that every scenario mutates, so scenario 3's precondition is evaluated against
balances scenarios 1 and 2 moved. The starting balance of **1500** is
**1300** by the time the precondition runs, which does not change that
scenario's verdict and does mean the four demonstrations are not independent
of each other's order.

Structure: `fresh()` restores the module database; `crash_after()` and
`crash_before()` are the two adjacent failure points.
"""

from __future__ import annotations

import inspect
import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "16-checkpoints-rollback"

TEMP = tempfile.mkdtemp(prefix="aiefs-checkpoint-")
START = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}


def fresh(ref, name):
    ref.DB.clear()
    ref.DB.update(START)
    path = os.path.join(TEMP, name)
    if os.path.exists(path):
        os.remove(path)
    return ref.Checkpoint(path)


def crash_after(ref, txid="tx-after"):
    """The crash the module injects: after the side effect."""
    checkpoint = fresh(ref, f"{txid}.json")
    try:
        ref.run_transfer(checkpoint, txid, "A", "B", 100, 200,
                         inject_crash_after_execute=True)
    except RuntimeError:
        pass
    moved = START["balance_A"] - ref.DB["balance_A"]
    result = ref.run_transfer(checkpoint, txid, "A", "B", 100, 200)
    return result, moved, START["balance_A"] - ref.DB["balance_A"], checkpoint


def crash_before(ref, txid="tx-before"):
    """The crash one line earlier: the marker is written, the effect is not."""
    checkpoint = fresh(ref, f"{txid}.json")
    checkpoint.save(ref.key(txid), {"status": "committed", "txid": txid,
                                    "from_acct": "A", "to_acct": "B",
                                    "amount": 100, "prior_last_transfer_id": None})
    result = ref.run_transfer(checkpoint, txid, "A", "B", 100, 200)
    return result, START["balance_A"] - ref.DB["balance_A"], checkpoint


def terminal_states(ref):
    block = inspect.getsource(ref.run_transfer).split("terminal_results = {")[1]
    return [line.split('"')[1] for line in block.split("}")[0].splitlines() if '"' in line]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    after_result, during, after, after_cp = crash_after(ref)
    before_result, before_moved, before_cp = crash_before(ref)
    source = inspect.getsource(ref.run_transfer)
    return {
        "after_result": after_result,
        "moved_at_crash": during,
        "moved_after_retry": after,
        "after_status": after_cp.load()[ref.key("tx-after")]["status"],
        "before_result": before_result,
        "before_moved": before_moved,
        "before_status": before_cp.load()[ref.key("tx-before")]["status"],
        "terminal": terminal_states(ref),
        "verify_after_short_circuit": source.index("terminal_results") < source.index(
            "Post-action verify"),
        "db_is_global": isinstance(ref.DB, dict),
        "start_balance": START["balance_A"],
        "after_two_scenarios": START["balance_A"] - 200,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: exactly once for the injected crash",
            all([result["after_result"] == "idempotent-skip",
                 result["moved_at_crash"] == 100, result["moved_after_retry"] == 100]),
            f"crashing after the side effect leaves {result['moved_at_crash']} moved, "
            f"and the retry returns {result['after_result']!r} with "
            f"{result['moved_after_retry']} moved in total -- one transfer across two "
            "attempts",
        ),
        practice.Check(
            "FINDING: the adjacent crash gives exactly zero",
            all([result["before_result"] == "idempotent-skip",
                 result["before_moved"] == 0,
                 result["before_status"] == "committed"]),
            f"with the marker written and the effect not, the retry returns "
            f"{result['before_result']!r} and {result['before_moved']} has moved -- the "
            "marker records an intention and is read as an outcome",
        ),
        practice.Check(
            "FINDING: the retry never reaches verify",
            all([len(result["terminal"]) == 4, "committed" in result["terminal"],
                 result["verify_after_short_circuit"],
                 result["after_status"] == "committed"]),
            f"committed is one of the {len(result['terminal'])} terminal states "
            f"{result['terminal']}, so the retry returns before the post-action read and "
            f"the record stays {result['after_status']!r} rather than advancing to "
            "verified",
        ),
        practice.Check(
            "FINDING: the four scenarios share one database",
            all([result["db_is_global"], result["start_balance"] == 1500,
                 result["after_two_scenarios"] == 1300]),
            f"DB is a module global every scenario mutates, so the "
            f"{result['start_balance']} starting balance is "
            f"{result['after_two_scenarios']} by the time scenario 3's precondition "
            "runs -- the four demonstrations are not independent of their order",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
