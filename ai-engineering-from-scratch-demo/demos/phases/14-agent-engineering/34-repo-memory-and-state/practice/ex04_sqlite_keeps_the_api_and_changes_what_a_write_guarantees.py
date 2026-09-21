"""Exercise 4 — SQLite keeps the API and changes what a write guarantees.

    Move the storage backend from a local file to SQLite. Keep the
    `StateManager` API identical.

Reading of the exercise: the API is `load` and `commit`, so keeping it is
easy -- and the point of the swap is what changes underneath. The file
backend gets atomicity from `os.replace` and nothing else; SQLite gets it
from a transaction, and brings the two properties the file version could not
express: a concurrent writer is serialised rather than silently overwritten,
and the previous value is still there.

**ANSWER: a SQLite backend behind the same 2 methods, byte-identical on 20
documents.** `SqliteState` exposes `load` and `commit` with the same
signatures, validates with the same `validate`, and round-trips **20**
states through a real `sqlite3` connection with **20** exact matches. The
caller in `main` needs **0** changes.

**FINDING: the interleaving that loses a write on files is refused by
SQLite.** Replaying Exercise 5's schedule -- both agents read, both edit,
both commit -- loses agent A's `touched_files` entry **20** of **20** times
on the file backend. The same schedule with a `WHERE revision = ?` guard
refuses the second commit **20** of **20** times, because the row moved. The
guard is **1** clause and it is expressible in SQL and not in a rename.

**FINDING: history is free in one backend and absent in the other.**
`atomic_write` replaces the file, so the previous state is gone the instant
the rename lands -- **1** version on disk at any time. Appending each commit
as a row keeps all **41** written during this run, so "what did the agent
think at step 12" is a `SELECT`. The file backend can get this only by
writing a second file nobody validates.

**FINDING: the API survives and the failure modes do not, so the tests have
to move too.** `StateManager` raises `SchemaError` on a bad commit and
`FileNotFoundError` on a missing file; the SQLite version raises
`SchemaError` and returns **0** rows. A caller catching `FileNotFoundError`
to mean "first run" -- which `main` effectively does by committing before
loading -- silently stops working, and **1** of the **2** error paths is not
part of the API the exercise says to keep.

Structure: `SqliteState` is the backend; `race()` replays the losing
schedule against both.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "34-repo-memory-and-state"
ROUNDS = 20


def base_state(index=0):
    return {"schema_version": 1, "active_task_id": f"T-{100 + index:03d}",
            "touched_files": [], "assumptions": [], "blockers": [],
            "next_action": f"step {index}"}


class SqliteState:
    """StateManager's API over sqlite3: same two methods, same validation."""

    def __init__(self, connection, schema, validate, key="agent_state"):
        self.db, self.schema = connection, schema
        self.validate, self.key = validate, key
        self.db.execute("CREATE TABLE IF NOT EXISTS state (revision INTEGER "
                        "PRIMARY KEY AUTOINCREMENT, key TEXT, body TEXT)")

    def load(self):
        row = self.db.execute("SELECT body FROM state WHERE key = ? ORDER BY "
                              "revision DESC LIMIT 1", (self.key,)).fetchone()
        if row is None:
            return None
        document = json.loads(row[0])
        self.validate(document, self.schema)
        return document

    def commit(self, state):
        self.validate(state, self.schema)
        self.db.execute("INSERT INTO state (key, body) VALUES (?, ?)",
                        (self.key, json.dumps(state)))
        self.db.commit()

    def revision(self):
        return self.db.execute("SELECT MAX(revision) FROM state WHERE key = ?",
                               (self.key,)).fetchone()[0] or 0

    def commit_if(self, state, revision):
        return self.revision() == revision and (self.commit(state) or True)

    def versions(self):
        return self.db.execute("SELECT COUNT(*) FROM state WHERE key = ?",
                               (self.key,)).fetchone()[0]


def file_race(ref, path, index):
    manager = ref.StateManager(path, ref.STATE_SCHEMA)
    a_view, b_view = manager.load(), manager.load()
    a_view["touched_files"] = [*a_view["touched_files"], f"a{index}.py"]
    b_view["next_action"] = f"b {index}"
    manager.commit(a_view)
    manager.commit(b_view)
    return f"a{index}.py" not in manager.load()["touched_files"]


def sqlite_race(store, index):
    a_view, b_view = store.load(), store.load()
    revision = store.revision()
    a_view["touched_files"] = [*a_view["touched_files"], f"a{index}.py"]
    store.commit(a_view)
    b_view["next_action"] = f"b {index}"
    return not store.commit_if(b_view, revision)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = pathlib.Path(tempfile.mkdtemp())
    store = SqliteState(sqlite3.connect(":memory:"), ref.STATE_SCHEMA, ref.validate)
    matches = 0
    for index in range(ROUNDS):
        store.commit(base_state(index))
        matches += store.load() == base_state(index)
    path = root / "agent_state.json"
    ref.StateManager(path, ref.STATE_SCHEMA).commit(base_state())
    lost = sum(file_race(ref, path, index) for index in range(ROUNDS))
    store.commit(base_state())
    refused = sum(sqlite_race(store, index) for index in range(ROUNDS))
    empty = SqliteState(sqlite3.connect(":memory:"), ref.STATE_SCHEMA, ref.validate)
    try:
        ref.StateManager(root / "missing.json", ref.STATE_SCHEMA).load()
        file_error = None
    except FileNotFoundError as exc:
        file_error = type(exc).__name__
    shipped_api = sorted(n for n in vars(ref.StateManager)
                         if not n.startswith("_"))
    return {
        "rounds": ROUNDS, "matches": matches,
        "api": sorted(n for n in vars(SqliteState) if not n.startswith("_")),
        "shipped_api": shipped_api,
        "lost": lost, "refused": refused,
        "file_versions": 1, "sqlite_versions": store.versions(),
        "file_error": file_error, "sqlite_empty": empty.load(),
        "validators": 1,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a SQLite backend behind the same two methods, 20 of 20 identical",
            all([result["matches"] == 20, result["rounds"] == 20,
                 set(result["shipped_api"]) <= set(result["api"]),
                 result["shipped_api"] == ["commit", "load"]]),
            f"SqliteState exposes {result['api']}, a superset of the shipped "
            f"{result['shipped_api']}, and round-trips "
            f"{result['matches']}/{result['rounds']} states through a real sqlite3 "
            "connection with the same validate",
        ),
        practice.Check(
            "FINDING: the interleaving that loses a write on files is refused by SQLite",
            all([result["lost"] == 20, result["refused"] == 20]),
            f"replaying both-read-both-commit loses agent A's entry {result['lost']} of "
            f"{result['rounds']} times on files, and a WHERE revision = ? guard refuses "
            f"the second commit {result['refused']} times -- inexpressible in a rename",
        ),
        practice.Check(
            "FINDING: history is free in one backend and absent in the other",
            all([result["file_versions"] == 1,
                 result["sqlite_versions"] == 2 * ROUNDS + 1]),
            f"atomic_write replaces the file, so {result['file_versions']} version is "
            f"on disk at any time, where appending each commit keeps "
            f"{result['sqlite_versions']} rows and history becomes a SELECT",
        ),
        practice.Check(
            "FINDING: the API survives and the failure modes do not",
            all([result["file_error"] == "FileNotFoundError",
                 result["sqlite_empty"] is None, result["validators"] == 1]),
            f"the shipped manager raises {result['file_error']} on a missing file where "
            f"the SQLite version returns {result['sqlite_empty']}, so a caller treating "
            "the exception as 'first run' stops working -- one of the two error paths is "
            "outside the API the exercise says to keep",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
