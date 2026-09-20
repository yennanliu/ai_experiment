"""Exercise 3 — the two orders lose different things, and idempotence loses neither.

    Inject a failure after replay claim but before a simulated export. Define
    and test the transaction or idempotency rule that makes recovery safe.

Reading of the exercise: the injection point the exercise names -- after the
claim, before the export -- only exists if the claim commits first, and the
lesson's `claim_and_consume` runs the operation *before* recording the nonce.
So the exercise is describing the other order, and the honest solution builds
both, injects the failure into each, and states the rule as the thing that
makes the choice stop mattering.

**THE RULE: the export carries the nonce as an idempotency key, and the
record of it is written in the same transaction as the export.** A retry then
finds the key already present and returns the first outcome instead of
performing a second export. That is what makes recovery safe under either
ordering, because neither ordering is safe by itself.

**ANSWER: claim-first loses the work; operate-first duplicates it; keyed
export does exactly one.** Each order has its own window -- claim-first can
crash after the claim, operate-first after the export -- and crashing in it
then retrying gives **0** exports, **2** exports and **1** export
respectively. The windows are different, and that is the point: neither order
is safe, they are unsafe about different things.

**FINDING: the lesson's own order is operate-first, so it is at-least-once.**
`claim_and_consume` calls `operation()` and records the nonce on the line
after, so a crash between the two leaves the nonce unclaimed and the work
done. Safe against losing the export, unsafe against repeating it -- which is
the correct trade only if the export is idempotent, and nothing here says it
is.

**FINDING: the idempotency key has to be the nonce and nothing else.** Keyed
on the arguments, two legitimate exports of the same query to the same
destination collapse into one -- **1** export where **2** were confirmed. The
nonce is per-confirmation, which is the granularity a user actually approved.

Structure: `Ledger` is the simulated export with a keyed side table, and
`attempt` runs one claim/export pair under a chosen order, crashing in that
order's own window.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "15-mcp-security-tool-poisoning"
NONCE = "nonce-1"


class Ledger:
    """The simulated export, with an idempotency key beside it."""

    def __init__(self):
        self.exports, self.keys = [], {}

    def export(self, destination, *, key=None):
        if key is not None and key in self.keys:
            return self.keys[key]  # the retry returns the first outcome
        self.exports.append(destination)
        outcome = {"exported": True, "sequence": len(self.exports)}
        if key is not None:
            self.keys[key] = outcome
        return outcome


def attempt(ledger, claimed, nonce, *, order, fail=None, key=None):
    """One claim/export pair. Each order has its own window to crash in."""
    if nonce in claimed:
        return "already used"
    if order == "claim-first":
        claimed.add(nonce)
        if fail == "after-claim":
            raise RuntimeError("crashed after the claim, before the export")
        return ledger.export("s3://reports", key=key)
    outcome = ledger.export("s3://reports", key=key)
    if fail == "after-export":
        raise RuntimeError("crashed after the export, before the record")
    claimed.add(nonce)  # recorded only once the export has returned
    return outcome


def run(order, fail, *, key=None):
    """Crash in that order's own window, then retry, and count real exports."""
    ledger, claimed = Ledger(), set()
    try:
        attempt(ledger, claimed, NONCE, order=order, fail=fail, key=key)
    except RuntimeError:
        pass
    retry = attempt(ledger, claimed, NONCE, order=order, key=key)
    return len(ledger.exports), retry


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = inspect.getsource(ref.ReplayStore.claim_and_consume)
    operation_line = next(i for i, line in enumerate(source.splitlines())
                          if "operation()" in line)
    record_line = next(i for i, line in enumerate(source.splitlines())
                       if "_consumed[nonce]" in line)

    ledger, claimed = Ledger(), set()
    attempt(ledger, claimed, "n-a", order="operate-first", key="n-a")
    attempt(ledger, claimed, "n-b", order="operate-first", key="n-b")
    by_nonce = len(ledger.exports)

    coarse, coarse_claimed = Ledger(), set()
    attempt(coarse, coarse_claimed, "n-a", order="operate-first", key="s3://reports")
    attempt(coarse, coarse_claimed, "n-b", order="operate-first", key="s3://reports")
    return {
        "claim_first": run("claim-first", "after-claim"),
        "operate_first": run("operate-first", "after-export"),
        "idempotent": run("operate-first", "after-export", key=NONCE),
        "operation_before_record": operation_line < record_line,
        "by_nonce": by_nonce, "by_arguments": len(coarse.exports),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: claim-first loses the work, operate-first duplicates it, keyed does one",
            all([result["claim_first"] == (0, "already used"),
                 result["operate_first"][0] == 2,
                 result["idempotent"][0] == 1]),
            f"each order crashed in its own window and retried: claim-first (crash after "
            f"the claim) gives {result['claim_first'][0]} exports and answers "
            f"{result['claim_first'][1]!r}; operate-first (crash after the export) gives "
            f"{result['operate_first'][0]}; and the nonce-keyed store, same crash, gives "
            f"{result['idempotent'][0]}",
        ),
        practice.Check(
            "THE RULE: the nonce is the idempotency key, written with the export",
            all([result["idempotent"][0] == 1,
                 result["idempotent"][1] == {"exported": True, "sequence": 1}]),
            f"the retry finds the key present and returns the first outcome, "
            f"{result['idempotent'][1]}, instead of exporting again. Recovery is then safe "
            "under either ordering, which matters because neither ordering is safe alone",
        ),
        practice.Check(
            "FINDING: the lesson's own order is operate-first, so it is at-least-once",
            result["operation_before_record"],
            "claim_and_consume calls operation() and records the nonce on a later line, so a "
            "crash between the two leaves the nonce unclaimed and the work done. Safe "
            "against losing the export and unsafe against repeating it -- the correct trade "
            "only if the export is idempotent, which nothing here says it is",
        ),
        practice.Check(
            "FINDING: the idempotency key has to be the nonce and nothing else",
            all([result["by_nonce"] == 2, result["by_arguments"] == 1]),
            f"two legitimate confirmations of the same query to the same destination give "
            f"{result['by_nonce']} exports keyed on the nonce and "
            f"{result['by_arguments']} keyed on the arguments. The nonce is "
            "per-confirmation, which is the granularity a user actually approved",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
