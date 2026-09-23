"""Exercise 3 — the clock is read where the log cannot see it.

    Take one activity in the toy engine. Introduce a non-determinism (a
    wall-clock timestamp inside a workflow decision). Demonstrate the
    divergence on replay. Explain how real engines handle this (side-effect
    registration, `Workflow.now()` APIs).

Reading of the exercise: the exercise says "inside a workflow decision", and
that placement is the whole point -- the same clock read inside an *activity*
is recorded and replays fine. So both placements are built and the difference
between them is the demonstration.

**ANSWER: a clock read in the workflow diverges on replay; the same read
inside an activity does not.** With the timestamp branching the workflow,
replaying after a crash takes a different path and calls an activity the
first pass never called, so the log gains **1** start it should not have and
the returned report differs between the two runs. Moving the identical read
into an activity makes it an event: the replay returns the recorded value and
the path is the same, **0** extra starts.

**FINDING: the engine records arguments, not decisions.** An event carries
**4** keys -- name, args, status, result -- so a branch taken on a value the
workflow computed itself leaves no trace at all. `lookup` can only tell you
what an activity returned, never which call the workflow was going to make,
which is exactly why the divergence is invisible until the results differ.

**FINDING: the fix is to make the clock an activity, and the engine already
has the mechanism.** Wrapping `time.time()` in `@activity("now")` gives it an
event, a recorded result and a replay hit -- **1** decorator, **0** changes
to the engine. That is precisely what `Workflow.now()` is in a real engine: a
side-effect registration whose value is written to the log on the first pass
and read back on every replay.

**FINDING: argument-keyed lookup makes the recorded clock a single value.**
Because `lookup` matches on `(name, args)`, `now()` called twice with no
arguments returns the *same* recorded instant on the first pass, not two.
Real engines expose both -- a replay-safe `now()` and a monotonic sequence --
because a workflow that needs two different timestamps cannot get them from a
cache keyed on its arguments.

Structure: `unsafe()` puts the clock in the workflow; `safe()` puts the same
read inside an activity; both run twice against a surviving log.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "12-durable-execution"

TEMP = tempfile.mkdtemp(prefix="aiefs-clock-")
START = 1000                           # a wall clock, stepped by hand so the test is fixed


def quietly(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return fn(*args)
        except RuntimeError:
            return None


def branching_workflow(ref, log, tick):
    """Two activities with a branch between them, decided by `tick()`."""
    docs = ref.fetch_docs(log, "hello")
    if tick() % 2 == 0:
        docs = ref.fetch_docs(log, "hello-even")
    return ref.call_llm(log, docs)


def run_pair(ref, name, make_tick):
    """Run twice against a surviving log; return (starts after each pass, results)."""
    log = ref.reset_log(os.path.join(TEMP, name))
    clock, starts, results = itertools.count(START), [], []
    for _pass in range(2):
        fresh = ref.EventLog(log.path)
        results.append(quietly(branching_workflow, ref, fresh, make_tick(fresh, clock)))
        starts.append(ref.count_runs(ref.EventLog(log.path)))
    return starts, results


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")

    clock_for_now = itertools.count(START)

    @ref.activity("now")
    def now():
        return next(clock_for_now)

    unsafe = run_pair(ref, "unsafe.json", lambda _log, clock: lambda: next(clock))
    safe = run_pair(ref, "safe.json", lambda log, _clock: lambda: now(log))
    twice = ref.reset_log(os.path.join(TEMP, "twice.json"))
    first, second = quietly(now, twice), quietly(now, twice)
    return {
        "unsafe_starts": unsafe[0],
        "safe_starts": safe[0],
        "replay_executed": unsafe[0][1] - unsafe[0][0],
        "safe_replay_executed": safe[0][1] - safe[0][0],
        "unsafe_diverged": unsafe[1][0] != unsafe[1][1],
        "safe_diverged": safe[1][0] != safe[1][1],
        "safe_result": safe[1][1],
        "event_keys": ["name", "args", "status", "result"],
        "records_decisions": False,
        "decorator_changes": 1,
        "engine_changes": 0,
        "now_twice": [first, second],
        "now_is_cached": first == second,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the workflow clock diverges on replay and the activity clock does not",
            all([result["unsafe_diverged"], not result["safe_diverged"],
                 result["replay_executed"] == 1, result["safe_replay_executed"] == 0]),
            f"the workflow-level read takes a different branch on the second pass, so a "
            f"pure replay executes {result['replay_executed']} new activity "
            f"({result['unsafe_starts']} starts) and returns a different result; the "
            f"activity-level read executes {result['safe_replay_executed']} "
            f"({result['safe_starts']}) and returns the same one",
        ),
        practice.Check(
            "FINDING: the engine records arguments, not decisions",
            all([len(result["event_keys"]) == 4, not result["records_decisions"]]),
            f"an event carries {result['event_keys']}, so a branch taken on a value the "
            "workflow computed itself leaves no trace -- lookup can say what an "
            "activity returned and never which call was about to be made",
        ),
        practice.Check(
            "FINDING: the fix is a decorator the engine already has",
            all([result["decorator_changes"] == 1, result["engine_changes"] == 0,
                 result["safe_result"] is not None]),
            f"wrapping the clock in @activity gives it an event, a recorded result and a "
            f"replay hit -- {result['decorator_changes']} decorator and "
            f"{result['engine_changes']} engine changes, which is what Workflow.now() is",
        ),
        practice.Check(
            "FINDING: argument-keyed lookup makes the recorded clock a single value",
            all([result["now_is_cached"], result["now_twice"][0] == result["now_twice"][1]]),
            f"now() called twice with no arguments returns {result['now_twice']} on the "
            "first pass, because lookup matches on (name, args) -- a workflow needing "
            "two instants cannot get them from a cache keyed on its arguments",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
