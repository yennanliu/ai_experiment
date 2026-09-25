"""Exercise 1 — the crossover does not exist because the work is a sleep.

    Run `code/main.py`, then modify the lead to spawn 5 workers instead of 3.
    Observe the wall-clock effect. At what worker count does spawn overhead
    exceed parallel savings in this demo?

Reading of the exercise: measure the spawn cost, then solve the inequality
rather than hunting for the crossing by trial -- because the answer is that
there isn't one, and the reason is the one line that stands in for the work.

**ANSWER: never, at any worker count.** A thread costs tens of microseconds to
start and join; `fake_web_fetch` sleeps **0.3 s**, four orders of magnitude
more. Parallel wall clock is `0.3 + N x spawn` and the sequential baseline is
`0.3 x N`, so overhead exceeds savings only if spawn exceeded 0.3 s per
worker. Measured at N = 3, 5, 10 and 50 the wall clock stays flat near 0.3 s
while the baseline grows linearly, and the gap widens with every worker added.

**FINDING: the win is an artefact of `time.sleep`.** `sleep` releases the GIL
and models work that costs no CPU. Swapping it for a fixed count of arithmetic
operations -- computing instead of waiting -- makes the threaded arm **no
faster** than running the same three calls one after another, because CPython
runs one thread at a time. The measurement has to be counted in operations
rather than in wall clock, or each thread simply does less work and the
contention hides itself. The demo's headline is a property of its stand-in,
not of the supervisor pattern, and a real worker that parses what it fetched
sits between the two.

**FINDING: `plan()` never reads the query.** It returns **3** sub-questions
with fixed suffixes -- historical origins, state of the art 2026, open
problems -- for any input; two entirely different queries produce the same
**3** decompositions. So "modify the lead to spawn 5 workers" means editing a
list literal, and worker count is a property of the lead's source rather than
of the question asked.

**FINDING: the demo prints two hardcoded timings beside a measured one.**
`stats["wall_clock_seconds"]` is measured. The two lines under it -- "~0.9s (3
* 0.3s)" and "~0.35s" -- are string literals. Change the worker count as the
exercise instructs and the measured number moves while both claims about it
stay put.

Structure: `timed()` runs the lead's fan-out for a given worker count;
`cpu_bound()` repeats it with the sleep replaced by arithmetic.
"""

from __future__ import annotations

import inspect
import re
import threading
import time

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "05-supervisor-orchestrator-pattern"
COUNTS = (3, 5, 10, 50)
SLEEP = 0.3
PROBE = 200
ROUNDS = 30000


def spawn_cost(count=PROBE):
    """Seconds to start and join one thread that does nothing."""
    start = time.perf_counter()
    threads = [threading.Thread(target=lambda: None) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return (time.perf_counter() - start) / count


def fan_out(work, count):
    """The lead's fan-out: `count` threads, started then joined."""
    start = time.perf_counter()
    threads = [threading.Thread(target=work) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - start


def burn(rounds=ROUNDS):
    """A fixed amount of arithmetic -- counted in operations, not in wall clock,
    so that threads contending for the GIL show up as elapsed time rather than
    as each thread quietly doing less."""
    total = 0
    for _ in range(rounds):
        total += sum(range(200))
    return total


def sleeping_arm():
    """Wall clock for the lead's fan-out at each worker count, against the baseline."""
    measured = {n: fan_out(lambda: time.sleep(SLEEP), n) for n in COUNTS}
    return {
        "parallel": {n: round(seconds, 3) for n, seconds in measured.items()},
        "baseline": {n: round(SLEEP * n, 3) for n in COUNTS},
        "flat": max(measured.values()) < SLEEP * 2,
        "crosses": any(seconds > SLEEP * n for n, seconds in measured.items()),
    }


def cpu_arm():
    """The same fan-out with the sleep replaced by arithmetic that holds the GIL."""
    threaded, serial = fan_out(burn, 3), sum(fan_out(burn, 1) for _ in range(3))
    return {"cpu_threaded": round(threaded, 3), "cpu_serial": round(serial, 3),
            "cpu_no_win": threaded > serial / 2, "true_win": round(serial / 3, 3)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    lead = ref.Lead(ref.Trace())
    spawn = spawn_cost()
    return {
        **sleeping_arm(), **cpu_arm(),
        "spawn": spawn, "sleep": SLEEP, "ratio": SLEEP / spawn, "counts": list(COUNTS),
        "plans": [len(lead.plan(q)) for q in ("A", "a totally different question")],
        "suffixes": len({s.split(" -- ")[-1] for q in ("A", "B") for s in lead.plan(q)}),
        "measured_key": "wall_clock_seconds" in src,
        "literals": [claim for claim in ("~0.9s", "~0.35s") if claim in src],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: never, at any worker count",
            all([not result["crosses"], result["flat"], result["ratio"] > 100,
                 result["parallel"][50] < result["baseline"][50]]),
            f"a thread costs {result['spawn'] * 1e6:.0f}us to start and join while the "
            f"worker sleeps {result['sleep']}s -- {result['ratio']:.0f}x more -- so "
            f"parallel is 0.3 + N x spawn against a baseline of 0.3 x N; at N=50 that is "
            f"{result['parallel'][50]}s against {result['baseline'][50]}s and the gap "
            "widens with every worker",
        ),
        practice.Check(
            "FINDING: the win is an artefact of time.sleep",
            result["cpu_no_win"],
            f"sleep releases the GIL and models work that costs no CPU; the same "
            f"arithmetic takes {result['cpu_threaded']}s across three threads against "
            f"{result['cpu_serial']}s run one after another, where a real three-way win "
            f"would be {result['true_win']}s -- CPython runs one thread at a time",
        ),
        practice.Check(
            "FINDING: plan() never reads the query",
            all([result["plans"] == [3, 3], result["suffixes"] == 3]),
            f"plan returns {result['plans'][0]} sub-questions with "
            f"{result['suffixes']} fixed suffixes for any input, so two entirely "
            "different queries decompose identically -- worker count is a property of "
            "the lead's source, not of the question",
        ),
        practice.Check(
            "FINDING: the demo prints two hardcoded timings beside a measured one",
            all([result["measured_key"], result["literals"] == ["~0.9s", "~0.35s"]]),
            f"stats['wall_clock_seconds'] is measured while "
            f"{' and '.join(result['literals'])} are string literals -- change the "
            "worker count as instructed and the measured number moves while both claims "
            "about it stay put",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
