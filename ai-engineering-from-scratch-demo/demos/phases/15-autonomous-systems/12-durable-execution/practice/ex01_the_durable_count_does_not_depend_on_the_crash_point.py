"""Exercise 1 — the durable count does not depend on the crash point.

    Run `code/main.py`. Observe the difference in activity-execution count
    between naive retry and replay. Change the crash point and show the
    replay count changes accordingly.

Reading of the exercise: the last clause predicts that the replay count moves
with the crash point. It does not, and checking that is the exercise -- so
both retry strategies are run at every crash point rather than at the one
`main` hard-codes.

**ANSWER: the durable total is 3 at every crash point; the naive total is
the crash point plus 3.** Crashing after activity **1** costs **4** naive
starts and **3** durable; after activity **2**, **5** and **3**. The durable
count is invariant because it equals the number of distinct `(name, args)`
pairs the workflow calls -- which is **3** -- and a replay adds none.

**FINDING: replay is keyed by arguments, not by position.**
`EventLog.lookup` matches on name, args and `status == "done"`, so calling
`fetch_docs("x")` twice in one pass yields **1** start and **2** events: the
second call prints `[replay]` on the *first* run, before any crash. That is
memoization. A real engine keys by sequence position, because a workflow is
allowed to call the same activity twice and mean it.

**FINDING: a crash inside an activity leaves an orphan the metric counts.**
`lookup` requires a `done` event, so the activity correctly re-runs -- but the
`started` event from the failed attempt stays in the log forever, and
`count_runs` counts every `started`. After one mid-activity crash and one
successful retry the log holds **3** events, **2** of them starts. The number
the exercise asks you to observe is inflated by exactly the number of crashes.

**FINDING: the log is rewritten whole on every append.** `append` reads the
entire file, appends in memory and dumps it back, with no temp file, no
rename and no fsync -- so writing the **n**th event costs O(n), the workflow
costs O(n^2), and every event is a window in which a crash truncates the log
that exists to survive crashes.

Structure: `durable()` and `naive()` run the shipped workflow under both
strategies at a chosen crash point; `orphan()` crashes inside an activity.
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
TEMP = tempfile.mkdtemp(prefix="aiefs-durable-")


def quietly(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        try:
            fn(*args, **kwargs)
        except RuntimeError:
            pass
    return buffer.getvalue()


def fresh(ref, name):
    return ref.reset_log(os.path.join(TEMP, name))


def durable(ref, crash):
    """Total activity starts across attempts when the log survives."""
    log = fresh(ref, f"durable-{crash}.json")
    for attempt in (1, 2):
        log = ref.EventLog(log.path)
        quietly(ref.workflow, log, QUERY, crash_after=crash if attempt == 1 else -1)
    return ref.count_runs(log)


def naive(ref, crash):
    """Total activity starts across attempts when the log is discarded."""
    total = 0
    for attempt in (1, 2):
        log = fresh(ref, f"naive-{crash}.json")
        quietly(ref.workflow, log, QUERY, crash_after=crash if attempt == 1 else -1)
        total += ref.count_runs(log)
    return total


def repeated(ref):
    """The same activity with the same arguments, twice, in one pass."""
    log = fresh(ref, "repeat.json")
    text = quietly(lambda: [ref.fetch_docs(log, "x"), ref.fetch_docs(log, "x")])
    return ref.count_runs(log), len(log.events()), text.count("[replay]")


def orphan(ref):
    """A crash inside an activity, then a successful retry of the same name."""
    log = fresh(ref, "orphan.json")

    @ref.activity("flaky")
    def boom(_value):
        raise RuntimeError("died mid-activity")

    @ref.activity("flaky")
    def fine(value):
        return value * 2

    quietly(boom, log, 1)
    quietly(fine, log, 1)
    return len(log.events()), ref.count_runs(log)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    append = inspect.getsource(ref.EventLog.append)
    return {
        "durable": [durable(ref, crash) for crash in (1, 2)],
        "naive": [naive(ref, crash) for crash in (1, 2)],
        "activities": ACTIVITIES,
        "repeated": list(repeated(ref)),
        "lookup_keys": [word for word in ("name", "args", "status")
                        if word in inspect.getsource(ref.EventLog.lookup)],
        "lookup_uses_position": "index" in inspect.getsource(ref.EventLog.lookup),
        "orphan": list(orphan(ref)),
        "rewrites_whole_file": "json.dump(evs" in append,
        "atomic": any(word in append for word in ("rename", "NamedTemporary", "fsync")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: durable is 3 at every crash point, naive is crash + 3",
            all([result["durable"] == [3, 3], result["naive"] == [4, 5],
                 result["activities"] == 3]),
            f"crashing after activity 1 and 2 costs {result['naive']} naive starts "
            f"against {result['durable']} durable -- the durable total is the number of "
            f"distinct activities, {result['activities']}, and does not move with the "
            "crash point",
        ),
        practice.Check(
            "FINDING: replay is keyed by arguments, not by position",
            all([result["repeated"] == [1, 2, 1], not result["lookup_uses_position"],
                 result["lookup_keys"] == ["name", "args", "status"]]),
            f"the same activity twice with the same arguments gives "
            f"{result['repeated'][0]} start, {result['repeated'][1]} events and "
            f"{result['repeated'][2]} replay line on the first pass -- lookup matches on "
            f"{result['lookup_keys']} and never on position",
        ),
        practice.Check(
            "FINDING: a crash inside an activity leaves an orphan the metric counts",
            result["orphan"] == [3, 2],
            f"after one mid-activity crash and a successful retry the log holds "
            f"{result['orphan'][0]} events of which {result['orphan'][1]} are starts, so "
            "count_runs is inflated by exactly the number of crashes",
        ),
        practice.Check(
            "FINDING: the log is rewritten whole on every append",
            all([result["rewrites_whole_file"], not result["atomic"]]),
            "append reads the file, appends in memory and dumps it back with no temp "
            "file, rename or fsync -- O(n) per event, O(n^2) per workflow, and a crash "
            "window on every one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
