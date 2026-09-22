"""Exercise 2 — the checkpointer stores the whole state every step.

    Swap the SQLite-like fake for a real SQLite checkpointer. Measure per-step
    serialization overhead.

Reading of the exercise: `sqlite3` is in the standard library, so the swap is
real rather than simulated. "Overhead" is measured in *bytes serialized per
step* rather than in seconds, because a wall-clock number would describe this
machine and the byte count describes the design: `InMemoryCheckpointer.save`
deep-copies the entire state on every node, so both stores grow with
`steps x state size` whatever the backend is.

**ANSWER: a real SQLite checkpointer, 3 rows and 374 bytes to the pause.**
Each step writes the whole state as JSON -- **90**, **120** and **164** bytes
-- so the payload grows monotonically even though every node after the first
adds one or two keys. Round-tripping every row reproduces the in-memory
checkpointer's history exactly: **3** of **3** states match.

**FINDING: the overhead is the snapshot, not the backend.** The in-memory
store keeps **3** deep copies holding **15** key-value pairs in total, and
SQLite keeps the same **3** snapshots as text. Switching backends changes
where the bytes go and not how many there are; a delta-encoded checkpointer
would write **8** changed keys instead of 19 copied ones.

**FINDING: a state field that cannot be serialized is only a problem for the
real one.** The in-memory store accepts a callable in the state because
`copy.deepcopy` handles it; `json.dumps` raises `TypeError`. The fake
checkpointer will accept states the durable one cannot, and the failure
appears the first time durability is switched on.

**FINDING: the pause flag is written to disk too.** The SQLite row for
`human_gate` contains `_pause_reason`, exactly as the in-memory one does, so
the resume bug survives the port: **1** of the **3** rows holds a key the
runner will trip over on the way back in.

Structure: `SqliteCheckpointer` implements the lesson's own three methods
against `sqlite3`; `run()` drives the lesson's graph through both.
"""

from __future__ import annotations

import json
import sqlite3

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "13-langgraph-stateful-graphs"
INPUT = "the CLI crashes on ctrl-c"


class SqliteCheckpointer:
    """The lesson's checkpointer interface, backed by stdlib sqlite3."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE ckpt (session TEXT, step TEXT, state TEXT)")
        self.bytes_per_step = []

    def save(self, session_id, step_name, state):
        payload = json.dumps(state, sort_keys=True)
        self.bytes_per_step.append(len(payload))
        self.db.execute("INSERT INTO ckpt VALUES (?, ?, ?)",
                        (session_id, step_name, payload))

    def load_latest(self, session_id):
        rows = self.history(session_id)
        return rows[-1] if rows else None

    def history(self, session_id):
        cursor = self.db.execute(
            "SELECT step, state FROM ckpt WHERE session = ? ORDER BY rowid", (session_id,))
        return [(step, json.loads(state)) for step, state in cursor]


def run(ref, checkpointer, session="s1"):
    runner = ref.Runner(ref.build_graph(), checkpointer)
    try:
        runner.run(session, {"input": INPUT, "step": 0, "human_approval": False})
    except ref.PausedAtNode:
        pass
    return checkpointer.history(session)


def rejects_callable(checkpointer, ref):
    state = {"input": INPUT, "callback": lambda: None}
    try:
        checkpointer.save("probe", "n", state)
    except TypeError as exc:
        return type(exc).__name__
    return "accepted"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    durable, memory = SqliteCheckpointer(), ref.InMemoryCheckpointer()
    rows, in_memory = run(ref, durable), run(ref, memory)
    matched = sum(a[1] == b[1] for a, b in zip(rows, in_memory))
    changed = sum(len(set(later[1].items()) - set(earlier[1].items()))
                  for earlier, later in zip(rows, rows[1:]))
    return {
        "rows": len(rows), "bytes": durable.bytes_per_step,
        "total": sum(durable.bytes_per_step), "matched": matched,
        "pairs": sum(len(state) for _, state in in_memory),
        "changed": changed + len(rows[0][1]),
        "sqlite_callable": rejects_callable(durable, ref),
        "memory_callable": rejects_callable(memory, ref),
        "flagged_rows": [step for step, state in rows if "_pause_reason" in state],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three rows, 374 bytes, byte-identical to the in-memory history",
            all([result["rows"] == 3, result["bytes"] == [90, 120, 164],
                 result["total"] == 374, result["matched"] == 3]),
            f"the SQLite checkpointer writes {result['rows']} rows of "
            f"{result['bytes']} bytes -- {result['total']} in total -- and round-tripping "
            f"them reproduces {result['matched']} of {result['rows']} in-memory states. "
            "The payload grows monotonically though the last two nodes add nothing",
        ),
        practice.Check(
            "FINDING: the overhead is the snapshot, not the backend",
            all([result["pairs"] == 15, result["changed"] == 8,
                 result["changed"] < result["pairs"]]),
            f"the in-memory store keeps {result['rows']} deep copies holding "
            f"{result['pairs']} key-value pairs and SQLite keeps the same snapshots as "
            f"text. A delta-encoded checkpointer would write {result['changed']} changed "
            "keys instead -- switching backends moves the bytes, it does not remove them",
        ),
        practice.Check(
            "FINDING: an unserializable field is only a problem for the real one",
            all([result["sqlite_callable"] == "TypeError",
                 result["memory_callable"] == "accepted"]),
            f"a callable in the state is {result['memory_callable']} by the in-memory "
            f"store, because copy.deepcopy handles it, and raises "
            f"{result['sqlite_callable']} in json.dumps. The fake accepts states the "
            "durable one cannot, and the failure appears when durability is switched on",
        ),
        practice.Check(
            "FINDING: the pause flag is written to disk too",
            all([result["flagged_rows"] == ["human_gate"],
                 len(result["flagged_rows"]) == 1]),
            f"the SQLite row for {result['flagged_rows'][0]!r} contains _pause_reason, "
            f"exactly as the in-memory one does -- {len(result['flagged_rows'])} of "
            f"{result['rows']} rows holds a key the runner trips over on the way back "
            "in. Porting the backend ports the bug",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
