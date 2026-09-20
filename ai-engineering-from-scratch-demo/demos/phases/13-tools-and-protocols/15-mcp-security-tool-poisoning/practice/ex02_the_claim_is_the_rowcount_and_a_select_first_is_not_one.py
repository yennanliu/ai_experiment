"""Exercise 2 — the claim is the rowcount, and a SELECT first is not one.

    Replace the in-memory replay store with a persistent conditional insert
    and prove two processes cannot both claim one nonce.

Reading of the exercise: "conditional insert" names the mechanism precisely,
so the solution is built on `INSERT ... ON CONFLICT DO NOTHING` and the
`rowcount` it returns, and the near-miss is built alongside it -- a
`SELECT`-then-`INSERT` that looks equivalent and is not. The two are then run
through the same interleaving, which is the only way to tell them apart.

**ANSWER: one row inserted, one gateway refused.** Two `SecurityGateway`
instances sharing one database and replaying one confirmation produce **1**
export; the loser sees `rowcount == 0` and answers
`requestState was already used`. The table holds **1** row.

**FINDING: the rowcount is the claim, and reading first gives it away.** Run
through the same interleave -- both check, then both write -- the conditional
insert still claims **1** of **2** while check-then-insert claims **2**. The
uniqueness constraint is not what makes it safe on its own; it is that the
database decides and reports, in one statement, rather than answering a
question the caller then acts on.

**FINDING: the in-memory store cannot lose this race because it never enters
it.** Two `ReplayStore` instances hold separate `_consumed` dicts behind
separate `threading.Lock`s, so both claim the same nonce -- **2** exports.
Inside one instance the lock is correct, which is exactly why a
single-process test passes.

**FINDING: the gateway builds its own store unless one is handed to it.**
`SecurityGateway.__init__` defaults `replay_store` to a fresh `ReplayStore`,
so two gateways in two processes are unshared by construction and shared only
by an argument someone has to remember to pass. The default is the unsafe
configuration.

Structure: `SqlStore` offers both claim strategies behind one flag, so the
race is the same code twice with `conditional` toggled.
"""

from __future__ import annotations

import inspect
import pathlib
import sqlite3
import tempfile
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "15-mcp-security-tool-poisoning"
ARGUMENTS = {"query": "q4", "destination": "s3://reports"}
ACCEPT = {"confirm": {"action": "accept", "content": {"confirm": True}}}
USED = "requestState was already used"


class SqlStore:
    """A persistent replay store; `conditional=False` is the near-miss."""

    def __init__(self, ref, path, *, conditional=True):
        self.ref, self.path, self.conditional = ref, path, conditional
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS used (nonce TEXT PRIMARY KEY, ts REAL)")

    def _db(self):
        return sqlite3.connect(self.path, isolation_level="IMMEDIATE", timeout=0.5)

    def claimed(self, nonce, expires_at):
        """True if this caller is the one that took the nonce."""
        with self._db() as db:
            if self.conditional:
                cursor = db.execute(
                    "INSERT INTO used VALUES (?, ?) ON CONFLICT(nonce) DO NOTHING",
                    (nonce, expires_at))
                return cursor.rowcount == 1
            seen = db.execute("SELECT 1 FROM used WHERE nonce = ?", (nonce,)).fetchone()
            return seen is None  # the gap: the row is written by `write` afterwards

    def write(self, nonce, expires_at):
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO used VALUES (?, ?)", (nonce, expires_at))

    def rows(self):
        with self._db() as db:
            return db.execute("SELECT count(*) FROM used").fetchone()[0]

    def claim_and_consume(self, nonce, *, expires_at, operation):
        if expires_at <= time.time():
            raise self.ref.ProtocolError(-32602, "requestState has expired")
        if not self.claimed(nonce, expires_at):
            raise self.ref.ProtocolError(-32602, USED)
        return operation()


def confirm(ref, gateway, token):
    body, headers = ref.make_request("tools/call", 2, {
        "name": "notes.export", "arguments": ARGUMENTS,
        "requestState": token, "inputResponses": ACCEPT})
    headers["Mcp-Name"] = "notes.export"
    response = gateway.handle(body, headers)[1]
    if "error" in response:
        return response["error"]["message"]
    return response["result"]["structuredContent"]["exported"]


def opening(ref, gateway):
    body, headers = ref.make_request("tools/call", 1, {"name": "notes.export",
                                                       "arguments": ARGUMENTS})
    headers["Mcp-Name"] = "notes.export"
    return gateway.handle(body, headers)[1]["result"]["requestState"]


def interleave(store, nonce):
    """Both callers check before either writes -- the classic race."""
    verdicts = [store.claimed(nonce, time.time() + 300) for _ in range(2)]
    for _ in range(2):
        store.write(nonce, time.time() + 300)
    return verdicts.count(True)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        def at(name, **kwargs):
            return SqlStore(ref, str(pathlib.Path(directory) / name), **kwargs)

        shared = at("shared.sqlite3")
        pair = [ref.SecurityGateway(replay_store=shared) for _ in range(2)]
        token = opening(ref, pair[0])
        outcomes = [confirm(ref, gateway, token) for gateway in pair]

        unshared = [ref.SecurityGateway(), ref.SecurityGateway()]
        memory_token = opening(ref, unshared[0])
        memory = [confirm(ref, gateway, memory_token) for gateway in unshared]

        default = inspect.signature(ref.SecurityGateway.__init__).parameters["replay_store"]
        return {
            "outcomes": outcomes, "exports": outcomes.count(True), "rows": shared.rows(),
            "refusal": next(o for o in outcomes if isinstance(o, str)),
            "memory": memory, "memory_exports": memory.count(True),
            "conditional_winners": interleave(at("c.sqlite3"), "n-1"),
            "select_first_winners": interleave(at("s.sqlite3", conditional=False), "n-1"),
            "default": default.default,
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one row inserted, one gateway refused",
            all([result["exports"] == 1, result["rows"] == 1,
                 result["refusal"] == USED]),
            f"two gateways sharing one database and replaying one confirmation produce "
            f"{result['exports']} export -- {result['outcomes']} -- and the table holds "
            f"{result['rows']} row. The loser sees rowcount 0 and answers {USED!r}",
        ),
        practice.Check(
            "FINDING: the rowcount is the claim, and reading first gives it away",
            all([result["conditional_winners"] == 1,
                 result["select_first_winners"] == 2]),
            f"through the same interleave -- both check, then both write -- the conditional "
            f"insert claims {result['conditional_winners']} of 2 and check-then-insert "
            f"claims {result['select_first_winners']}. The constraint alone is not what "
            "makes it safe; it is that the database decides and reports in one statement",
        ),
        practice.Check(
            "FINDING: the in-memory store cannot lose this race because it never enters it",
            all([result["memory_exports"] == 2, result["memory"] == [True, True]]),
            f"two ReplayStore instances hold separate _consumed dicts behind separate locks, "
            f"so both claim the same nonce: {result['memory_exports']} exports of one "
            "confirmation. Inside one instance the lock is correct, which is exactly why a "
            "single-process test passes",
        ),
        practice.Check(
            "FINDING: the gateway builds its own store unless one is handed to it",
            result["default"] is None,
            f"SecurityGateway.__init__ defaults replay_store to {result['default']} and then "
            "constructs a fresh ReplayStore, so two gateways in two processes are unshared "
            "by construction and shared only by an argument someone has to remember. The "
            "default is the unsafe configuration",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
