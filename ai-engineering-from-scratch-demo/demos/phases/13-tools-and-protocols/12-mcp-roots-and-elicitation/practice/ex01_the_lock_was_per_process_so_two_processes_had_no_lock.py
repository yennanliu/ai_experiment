"""Exercise 1 — the lock was per-process, so two processes had no lock.

    Replace the in-memory replay store with SQLite. Use one transaction to
    claim the nonce and delete the note, then prove two processes cannot both
    commit.

Reading of the exercise: "two processes" is simulated as two `NotesServer`
instances, because what distinguishes a process here is not an OS boundary but
an unshared `threading.Lock`. That substitution is checked rather than
assumed: the same replay of the same token is run against two in-memory
servers first, and it succeeds twice. Only then is SQLite worth swapping in.

**ANSWER: one shared database, one transaction, and exactly one commit.**
Two servers sharing a `SqliteReplayStore` and replaying one token delete
**1** note between them; the loser gets `-32602 requestState was already
consumed`. The in-memory store is a drop-in swap -- same
`claim_and_consume(nonce, expires_at=, operation=)` -- so nothing in
`NotesServer` changes.

**FINDING: the in-memory store does not fail here, it does not participate.**
Two `ReplayStore` instances both commit the same nonce, **2** deletions for
**1** token, because `self._consumed` and `self._lock` are per-instance.
Within one instance the same replay is correctly refused, so the defect is
invisible to any single-process test.

**FINDING: claiming and mutating must share the transaction, and the order
inside it does not matter as long as the rollback does.** With the `INSERT`
and the delete in one `with sqlite3.connect(...)` block, an operation that
raises leaves **0** rows behind and the nonce is claimable again -- the retry
succeeds. Committing the claim separately would consume the nonce for an
operation that never happened, which is unretryable by construction.

**FINDING: the capacity limit has to be re-implemented, because it was an
invariant of the dict and not of the design.** `max_entries=1` reproduces
`-32023 replay protection capacity exhausted`, and expiry-based eviction has
to become a `DELETE ... WHERE expires_at <= ?` inside the same transaction.
Neither is free once the store outlives the process.

Structure: `SqliteReplayStore` is the drop-in; `replay` runs one token against
one server and reports the outcome either way.
"""

from __future__ import annotations

import pathlib
import sqlite3
import tempfile
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "12-mcp-roots-and-elicitation"
CONSUMED = "requestState was already consumed"


class SqliteReplayStore:
    """The lesson's ReplayStore, over a database two processes can share."""

    def __init__(self, ref, path, max_entries=1_000, clock=time.time):
        self.ref, self.path, self.max_entries, self._clock = ref, path, max_entries, clock
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS consumed ("
                       "nonce TEXT PRIMARY KEY, expires_at REAL)")

    def _connect(self):
        return sqlite3.connect(self.path, isolation_level="IMMEDIATE", timeout=0.5)

    def claim_and_consume(self, nonce, *, expires_at, operation):
        now = self._clock()
        with self._connect() as db:  # one transaction: claim and mutate, or neither
            db.execute("DELETE FROM consumed WHERE expires_at <= ?", (now,))
            if expires_at <= now:
                raise self.ref.McpError(-32602, "requestState expired")
            if db.execute("SELECT count(*) FROM consumed").fetchone()[0] >= self.max_entries:
                raise self.ref.McpError(-32023, "replay protection capacity exhausted")
            try:
                db.execute("INSERT INTO consumed VALUES (?, ?)", (nonce, expires_at))
            except sqlite3.IntegrityError as exc:
                raise self.ref.McpError(-32602, CONSUMED) from exc
            return operation()

    def rows(self):
        with self._connect() as db:
            return db.execute("SELECT count(*) FROM consumed").fetchone()[0]


def opening(ref, server):
    """Round one: the elicitation's token and its candidate ids."""
    result = server.dispatch(ref.tool_request(1))["result"]
    schema = result["inputRequests"]["delete_choice"]["params"]["requestedSchema"]
    return result["requestState"], schema["properties"]["note_id"]["enum"]


def replay(ref, server, token, note_id):
    """Round two against one server, reporting whichever way it went."""
    request = ref.tool_request(2)
    request["params"].update({"requestState": token, "inputResponses": {"delete_choice": {
        "action": "accept", "content": {"note_id": note_id, "confirm": True}}}})
    response = server.dispatch(request)
    if "error" in response:
        return response["error"]["message"]
    return response["result"]["structuredContent"]


def boom():
    raise RuntimeError("the delete failed after the nonce was claimed")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    memory = [ref.NotesServer(), ref.NotesServer()]
    token, candidates = opening(ref, memory[0])
    unshared = [replay(ref, server, token, candidates[0]) for server in memory]

    with tempfile.TemporaryDirectory() as directory:
        def at(name, **kwargs):
            return SqliteReplayStore(ref, str(pathlib.Path(directory) / name), **kwargs)

        store = at("replay.sqlite3")
        shared = [ref.NotesServer(replay_store=store), ref.NotesServer(replay_store=store)]
        shared_token, shared_candidates = opening(ref, shared[0])
        outcomes = [replay(ref, s, shared_token, shared_candidates[0]) for s in shared]

        failing, later, raised = at("rollback.sqlite3"), time.time() + 60, None
        try:
            failing.claim_and_consume("n-1", expires_at=later, operation=boom)
        except RuntimeError as exc:
            raised = str(exc)
        after_rollback = failing.rows()
        retried = failing.claim_and_consume("n-1", expires_at=later,
                                            operation=lambda: "committed")

        tiny, capacity = at("tiny.sqlite3", max_entries=1), None
        tiny.claim_and_consume("first", expires_at=later, operation=lambda: None)
        try:
            tiny.claim_and_consume("second", expires_at=later, operation=lambda: None)
        except ref.McpError as exc:
            capacity = (exc.code, exc.message)
        return {
            "unshared": unshared, "outcomes": outcomes, "rows": store.rows(),
            "unshared_deleted": sum(isinstance(o, dict) for o in unshared),
            "shared_deleted": sum(isinstance(o, dict) for o in outcomes),
            "refusal": next(o for o in outcomes if isinstance(o, str)),
            "raised": raised, "after_rollback": after_rollback, "retried": retried,
            "capacity": capacity,
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one shared database, one transaction, exactly one commit",
            all([result["shared_deleted"] == 1, result["rows"] == 1,
                 result["refusal"] == CONSUMED]),
            f"two servers sharing a SqliteReplayStore and replaying one token delete "
            f"{result['shared_deleted']} note between them -- {result['outcomes']} -- leaving "
            f"{result['rows']} consumed row. The swap is drop-in, so NotesServer is unchanged",
        ),
        practice.Check(
            "FINDING: the in-memory store does not fail here, it does not participate",
            all([result["unshared_deleted"] == 2,
                 all(isinstance(o, dict) for o in result["unshared"])]),
            f"two ReplayStore instances both commit the same nonce, "
            f"{result['unshared_deleted']} deletions for one token, because _consumed and "
            "_lock are per-instance. Within one instance the same replay is correctly "
            "refused, so the defect is invisible to any single-process test",
        ),
        practice.Check(
            "FINDING: claiming and mutating must share the transaction",
            all([result["raised"] is not None, result["after_rollback"] == 0,
                 result["retried"] == "committed"]),
            f"an operation that raises inside the block leaves {result['after_rollback']} "
            f"rows and the nonce is claimable again -- the retry returns {result['retried']!r}. "
            "Committing the claim separately would consume a nonce for an operation that "
            "never happened, which is unretryable by construction",
        ),
        practice.Check(
            "FINDING: the capacity limit has to be re-implemented",
            result["capacity"] == (-32023, "replay protection capacity exhausted"),
            f"max_entries=1 reproduces {result['capacity']}, and expiry eviction becomes a "
            "DELETE ... WHERE expires_at <= ? inside the same transaction. Both were "
            "invariants of a dict that the design never stated, and neither is free once the "
            "store outlives the process",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
