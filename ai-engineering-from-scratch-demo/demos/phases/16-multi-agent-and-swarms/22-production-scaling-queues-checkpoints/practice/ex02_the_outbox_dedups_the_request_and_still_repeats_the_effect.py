"""Exercise 2 — the outbox dedups the request and still repeats the effect.

    Implement an **outbox** table: every tool call writes to outbox first,
    then a separate goroutine/task executes. Verify idempotency by running the
    tool call twice.

Reading of the exercise: the outbox shares the reference `CheckpointStore`'s
SQLite connection, so a step's outbox row and its checkpoint commit in one
transaction; "running the tool call twice" is done at each of the three
places a real system repeats it -- the agent re-issuing, two workers
resuming, and the executor retrying.

**ANSWER: idempotent at the outbox, keyed on (thread_id, super_step).**
Issuing the same call twice leaves 1 row and 1 execution. Crashing the agent
at step 3 and resuming with **two** workers -- the lease-less race exercise 1
found -- executes 5 effects for 5 steps through the outbox, against 7 when the
tool is called inline, as the reference agent would have to.

**FINDING: the outbox is at-least-once; the receiver makes it exactly-once.**
An executor that crashes after the tool succeeds but before marking the row
done re-sends it on restart: 2 effects when the tool ignores the key, 1 when
the tool dedups on it. The lesson's "both steps idempotent" is load-bearing,
and the second step is not the outbox's to provide.

**FINDING: the shared transaction is what makes a retry safe.** A step whose
output varies between attempts (as an LLM's does) writes payload "attempt-1",
then crashes before its checkpoint commits. On the shared connection the
outbox row rolls back with it and the retry's payload is the one executed. On
a separately committed outbox the key dedups the retry away, so the executor
sends attempt-1's payload while the checkpoint records attempt-2's state:
1 divergent effect that no dedup key can see.

Structure: `Outbox` puts its table on the reference store's connection and
never commits on its own -- `CheckpointStore.write` commits both -- unless
built with `separate=True`; `drain()` is the executor task.
"""

from __future__ import annotations

import collections
import contextlib
import io
import sqlite3

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "22-production-scaling-queues-checkpoints"


class Outbox:
    def __init__(self, store, separate=False):
        self.conn = sqlite3.connect(":memory:") if separate else store.conn
        self.separate, self.issued = separate, 0
        self.conn.execute("CREATE TABLE IF NOT EXISTS outbox (key TEXT PRIMARY KEY, "
                          "payload TEXT, done INTEGER DEFAULT 0)")

    def put(self, key, payload):
        self.issued += 1  # what an inline tool call would have executed
        self.conn.execute("INSERT OR IGNORE INTO outbox (key, payload) VALUES (?, ?)", (key, payload))
        if self.separate:
            self.conn.commit()

    def drain(self, tool, crash_after_effect=False):
        rows = self.conn.execute("SELECT key, payload FROM outbox WHERE done = 0").fetchall()
        for key, payload in rows:
            tool(key, payload)
            if crash_after_effect:
                return
            self.conn.execute("UPDATE outbox SET done = 1 WHERE key = ?", (key,))
            self.conn.commit()


def agent(ref, store, outbox, thread_id, crash_at=None):
    """The reference agent with an outbox row written before each checkpoint."""
    original = store.write
    store.write = lambda tid, step, state: (outbox.put(f"{tid}:{step}", f"charge {state}"),
                                            original(tid, step, state))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(RuntimeError):
        ref.run_agent_with_checkpoint(store, thread_id, crash_at=crash_at)
    store.write = original


def resumed_twice(ref):
    store, effects = ref.CheckpointStore(":memory:"), collections.Counter()
    outbox = Outbox(store)
    agent(ref, store, outbox, "t-1", crash_at=3)
    snapshot = store.latest("t-1")
    for _ in range(2):  # both workers read the crash checkpoint before either writes
        store.latest = lambda _tid: (snapshot[0], dict(snapshot[1]))
        agent(ref, store, outbox, "t-1")
    outbox.drain(lambda key, payload: effects.update([key]))
    return sum(effects.values()), outbox.issued


def retry_after_effect(ref, receiver_dedups):
    store, sent, seen = ref.CheckpointStore(":memory:"), collections.Counter(), set()
    outbox = Outbox(store)
    outbox.put("t-1:0", "charge")
    store.conn.commit()

    def tool(key, payload):
        if not (receiver_dedups and key in seen):
            sent[key] += 1
        seen.add(key)
    outbox.drain(tool, crash_after_effect=True)
    outbox.drain(tool)
    return sent["t-1:0"]


def divergence(ref, separate):
    store, executed = ref.CheckpointStore(":memory:"), []
    outbox = Outbox(store, separate=separate)
    outbox.put("t-1:0", "attempt-1")
    store.conn.rollback()  # the crash, before CheckpointStore.write commits
    outbox.put("t-1:0", "attempt-2")
    store.write("t-1", 0, {"payload": "attempt-2"})
    outbox.drain(lambda key, payload: executed.append(payload))
    return executed, store.latest("t-1")[1]["payload"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store, effects = ref.CheckpointStore(":memory:"), collections.Counter()
    twice = Outbox(store)
    for _ in range(2):
        twice.put("t-9:0", "charge")
        store.conn.commit()
        twice.drain(lambda key, payload: effects.update([key]))
    return {
        "twice_rows": store.conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0],
        "twice_effects": effects["t-9:0"], "race": resumed_twice(ref),
        "retry": {dedup: retry_after_effect(ref, dedup) for dedup in (False, True)},
        "shared": divergence(ref, separate=False), "split": divergence(ref, separate=True),
    }


def verify(result):
    (outboxed, inline), retry = result["race"], result["retry"]
    return [
        practice.Check(
            "ANSWER: idempotent at the outbox, keyed on (thread_id, super_step)",
            all([result["twice_rows"] == 1, result["twice_effects"] == 1,
                 outboxed == 5, inline == 7]),
            f"the same call issued twice leaves {result['twice_rows']} row and "
            f"{result['twice_effects']} execution; two workers resuming a crashed thread "
            f"cause {outboxed} effects for 5 steps through the outbox against {inline} inline",
        ),
        practice.Check(
            "FINDING: the outbox is at-least-once; the receiver makes it exactly-once",
            retry == {False: 2, True: 1},
            f"an executor crashing after the effect and before marking done re-sends: "
            f"{retry[False]} effects when the tool ignores the key, {retry[True]} when it "
            "dedups on it",
        ),
        practice.Check(
            "FINDING: the shared transaction is what makes a retry safe",
            result["shared"] == (["attempt-2"], "attempt-2")
            and result["split"] == (["attempt-1"], "attempt-2"),
            f"shared connection: executed {result['shared'][0]}, checkpoint "
            f"{result['shared'][1]!r}; separate commit: executed {result['split'][0]}, "
            f"checkpoint {result['split'][1]!r} -- a divergence the dedup key hides",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
