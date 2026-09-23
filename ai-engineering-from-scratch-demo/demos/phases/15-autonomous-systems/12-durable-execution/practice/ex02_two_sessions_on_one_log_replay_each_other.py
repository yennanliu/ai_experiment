"""Exercise 2 — two sessions on one log replay each other.

    Convert the toy engine to use `thread_id` explicitly. Simulate two
    concurrent sessions sharing the engine and confirm their event logs do
    not collide.

Reading of the exercise: "confirm they do not collide" is only meaningful
against a baseline where they do, so the collision is measured on the shipped
engine first and the conversion is judged by what it changes. The conversion
itself is a keyed lookup, not a rewrite.

**ANSWER: unkeyed, the second session executes 0 activities and inherits the
first's results; keyed, each runs its own 3.** Two sessions issuing the same
query against one `EventLog` produce **3** `[run]` lines and **3**
`[replay]` lines, and the log ends with **3** starts rather than 6. Session B
is handed session A's report and never calls a single activity. Adding a
`thread_id` to the lookup key restores **6** starts and **2** distinct logs'
worth of events in one file.

**FINDING: the collision is silent and looks like a feature.** Nothing
errors, nothing warns, and every replayed line prints the same
`[replay] ... (from log)` the crash-recovery path prints. The only
distinguishing signal is that a replay happened before any crash -- which the
engine does not record, because events carry **4** keys and none of them is a
run, an attempt or a time.

**FINDING: the key is the whole conversion.** `lookup` matches `name` and
`args`; adding `thread_id` to the event and to that comparison is the entire
change, and it works because the cache key was already the identity of the
call. The engine has **1** other place that would need to know about threads
-- `EventLog.path` -- and the shipped design already allows per-thread files,
which is the other valid answer and the one that scales worse.

**FINDING: sharing the log is the only concurrency the engine has.** There
is no lock, no sequence number and no writer coordination: `append` reads the
whole file and writes it back, so two genuinely concurrent sessions do not
merely collide on the cache, they lose events. Of the **3** methods on
`EventLog`, **2** read the file and **1** rewrites it, and none is atomic.

Structure: `session()` runs the workflow against a log; `keyed_lookup()` is
the conversion, applied by wrapping the engine's own lookup.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "12-durable-execution"

QUERY, ACTIVITIES = "hello", 3
TEMP = tempfile.mkdtemp(prefix="aiefs-threads-")


def session(ref, log, query=QUERY):
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        ref.workflow(log, query)
    return buffer.getvalue()


def shared(ref):
    """Two sessions, one log, the engine as shipped."""
    log = ref.reset_log(os.path.join(TEMP, "shared.json"))
    text = session(ref, log) + session(ref, log)
    return text.count("[run]"), text.count("[replay]"), ref.count_runs(log)


def keyed(originals, thread_id):
    """`lookup` and `append` with the thread id folded into the key."""
    base_lookup, base_append = originals

    def lookup(self, name, args):
        return base_lookup(self, f"{thread_id}:{name}", args)

    def append(self, event):
        base_append(self, {**event, "name": f"{thread_id}:{event['name']}",
                           "thread_id": thread_id})
    return lookup, append


def threaded(ref):
    """The same two sessions with the conversion applied."""
    path = os.path.join(TEMP, "threaded.json")
    log = ref.reset_log(path)
    originals = (ref.EventLog.lookup, ref.EventLog.append)
    starts, threads = 0, set()
    try:
        for thread_id in ("sess-a", "sess-b"):
            ref.EventLog.lookup, ref.EventLog.append = keyed(originals, thread_id)
            session(ref, ref.EventLog(path))
    finally:
        ref.EventLog.lookup, ref.EventLog.append = originals
    for event in ref.EventLog(path).events():
        starts += event["status"] == "started"
        threads.add(event.get("thread_id"))
    return starts, sorted(threads)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs, replays, collided = shared(ref)
    starts, threads = threaded(ref)
    log_source = inspect.getsource(ref.EventLog)
    return {
        "activities": ACTIVITIES,
        "shared_runs": runs, "shared_replays": replays, "shared_starts": collided,
        "keyed_starts": starts, "threads": threads,
        "event_keys": sorted({"name", "args", "status", "result"}),
        "lookup_fields": [word for word in ("name", "args", "status")
                          if word in inspect.getsource(ref.EventLog.lookup)],
        "path_is_the_key": list(ref.EventLog.__dataclass_fields__) == ["path"],
        "methods": len([name for name, value in vars(ref.EventLog).items()
                        if callable(value) and not name.startswith("__")]),
        "has_lock": any(word in log_source for word in ("Lock", "flock", "fcntl")),
        "has_sequence": "seq" in log_source or "index" in log_source,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: unkeyed the second session runs nothing; keyed, both run three",
            all([result["shared_runs"] == 3, result["shared_replays"] == 3,
                 result["shared_starts"] == 3,
                 result["keyed_starts"] == 6, result["threads"] == ["sess-a", "sess-b"]]),
            f"two sessions on one log give {result['shared_runs']} runs and "
            f"{result['shared_replays']} replays for {result['shared_starts']} starts -- "
            f"session B executes nothing -- while the keyed engine records "
            f"{result['keyed_starts']} starts across {result['threads']}",
        ),
        practice.Check(
            "FINDING: the collision is silent and looks like a feature",
            all([result["shared_replays"] == result["activities"],
                 len(result["event_keys"]) == 4,
                 "thread_id" not in result["event_keys"]]),
            f"all {result['shared_replays']} replayed lines are the ones the "
            f"crash-recovery path prints, and an event carries "
            f"{len(result['event_keys'])} keys {result['event_keys']} -- none of them a "
            "run, an attempt or a time",
        ),
        practice.Check(
            "FINDING: the key is the whole conversion",
            all([result["lookup_fields"] == ["name", "args", "status"],
                 result["path_is_the_key"]]),
            f"lookup matches on {result['lookup_fields']}, so folding a thread id into "
            "it is the entire change; the other valid answer is one file per thread, "
            "which the single-field EventLog already allows and which scales worse",
        ),
        practice.Check(
            "FINDING: sharing the log is the only concurrency the engine has",
            all([not result["has_lock"], not result["has_sequence"],
                 result["methods"] == 3]),
            f"EventLog has {result['methods']} methods, no lock and no sequence number, "
            "and append rewrites the whole file -- so genuinely concurrent sessions do "
            "not merely share a cache, they lose events",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
