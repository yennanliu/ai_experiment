"""Exercise 3 — asyncio loses, because the lesson's executor does no I/O.

    Replace the thread pool with `asyncio.gather`. Benchmark both. You should
    see small wins on async because of lower context-switch cost, but only if
    executors do real I/O.

Reading of the exercise: both arms are built and benchmarked, and the exercise's
own escape clause is taken as the hypothesis to test rather than a caveat to
repeat. `executor_weather` is `time.sleep`, which is not I/O -- it is a blocking
call that holds a thread. So the exercise's condition is not met by the lesson
it is set on, and the measurement should show that.

**ANSWER: both arms hit the same `max` floor, and neither wins by enough to
name.** Three calls at 40/60/80 ms finish in one 80 ms window under
`ThreadPoolExecutor` and under `asyncio.gather` alike, because the cost being
divided is a sleep and both mechanisms overlap sleeps perfectly.

**FINDING: `asyncio.gather` over `time.sleep` is not concurrent at all.**
Wrapping the lesson's executor in an `async def` and gathering it runs the three
sleeps **sequentially** -- the coroutine never yields, so the loop cannot
interleave. Measured against the same 80 ms floor, the naive async arm takes the
**sequential** time instead. Getting the win requires `asyncio.sleep`, or
`run_in_executor`, which puts the thread pool back.

**FINDING: so the exercise's condition is the whole result.** "Only if executors
do real I/O" is doing all the work: `time.sleep` releases the GIL, so threads
overlap it, while `await` on a non-awaiting coroutine does not yield, so the
event loop cannot. Against this executor the thread pool is not slightly worse
than async -- it is the only one of the two that works.

**FINDING: and the fix makes the two identical, not async-favourable.** Swapped
to `asyncio.sleep`, the gather arm lands on the same 80 ms floor as the threads,
within a few milliseconds. The context-switch saving the exercise predicts is
real and is far below the resolution of a three-call fan-out; it is a
throughput argument for thousands of concurrent calls, not a latency argument
for three.

Structure: `naive_async` awaits the lesson's own blocking executor, `true_async`
awaits `asyncio.sleep` for the same durations, and `floor` is the max the
arithmetic says both should reach.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "03-parallel-and-streaming-tool-calls"
LATENCY = {"Bengaluru": 40, "Tokyo": 60, "Zurich": 80}
SLACK_MS = 45                   # generous, so the check reads the shape and not the machine


def patched(ref, latencies):
    original = dict(ref.SIMULATED_LATENCY_MS)
    ref.SIMULATED_LATENCY_MS.clear()
    ref.SIMULATED_LATENCY_MS.update(latencies)
    return original


def restore(ref, original):
    ref.SIMULATED_LATENCY_MS.clear()
    ref.SIMULATED_LATENCY_MS.update(original)


async def naive_async(ref, cities):
    """gather over the lesson's own executor, which blocks instead of awaiting."""
    async def call(city):
        return ref.executor_weather(city)
    return await asyncio.gather(*(call(city) for city in cities))


async def true_async(latencies):
    """The same durations with a coroutine that actually yields."""
    async def call(city):
        await asyncio.sleep(latencies[city] / 1000.0)
        return {"city": city}
    return await asyncio.gather(*(call(city) for city in latencies))


def elapsed(thunk):
    start = time.perf_counter()
    value = thunk()
    return (time.perf_counter() - start) * 1000, value


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cities = list(LATENCY)
    floor_ms, ceiling_ms = max(LATENCY.values()), sum(LATENCY.values())
    original = patched(ref, LATENCY)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            threads_ms, threaded = ref.run_parallel(cities)
            naive_ms, _ = elapsed(lambda: asyncio.run(naive_async(ref, cities)))
    finally:
        restore(ref, original)
    true_ms, _ = elapsed(lambda: asyncio.run(true_async(LATENCY)))
    return {
        "floor_ms": floor_ms, "ceiling_ms": ceiling_ms,
        "threads_ms": round(threads_ms), "naive_ms": round(naive_ms),
        "true_ms": round(true_ms),
        "threads_at_floor": threads_ms < floor_ms + SLACK_MS,
        "true_at_floor": true_ms < floor_ms + SLACK_MS,
        "naive_at_ceiling": naive_ms >= ceiling_ms - 5,
        "naive_is_sequential": naive_ms > floor_ms + SLACK_MS,
        "results": len(threaded),
        "gap_ms": round(abs(true_ms - threads_ms)),
        "close": abs(true_ms - threads_ms) < SLACK_MS,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: both working arms hit the same max floor",
            all([result["threads_at_floor"], result["true_at_floor"],
                 result["results"] == 3, result["close"]]),
            f"three calls at {list(LATENCY.values())} ms finish in one "
            f"{result['floor_ms']} ms window under both mechanisms: threads "
            f"{result['threads_ms']} ms, asyncio.gather over asyncio.sleep "
            f"{result['true_ms']} ms -- {result['gap_ms']} ms apart. The cost being divided "
            "is a sleep, and both overlap sleeps perfectly",
        ),
        practice.Check(
            "FINDING: asyncio.gather over time.sleep is not concurrent at all",
            all([result["naive_is_sequential"], result["naive_at_ceiling"]]),
            f"wrapping the lesson's own executor in an async def and gathering it takes "
            f"{result['naive_ms']} ms against a {result['floor_ms']} ms floor and a "
            f"{result['ceiling_ms']} ms sequential ceiling -- it reaches the ceiling. The "
            "coroutine never awaits, so the loop has no point at which to interleave, and "
            "the three sleeps run one after another",
        ),
        practice.Check(
            "FINDING: so the exercise's condition is the whole result",
            all([result["threads_at_floor"], result["naive_is_sequential"]]),
            f"'only if executors do real I/O' is doing all the work here. time.sleep "
            f"releases the GIL, so threads overlap it ({result['threads_ms']} ms); await on "
            f"a non-awaiting coroutine does not yield, so the event loop cannot "
            f"({result['naive_ms']} ms). Against this executor the thread pool is not "
            "slightly worse than async -- it is the only one of the two that works",
        ),
        practice.Check(
            "FINDING: and the fix makes the two identical, not async-favourable",
            all([result["close"], result["gap_ms"] < SLACK_MS, result["true_at_floor"]]),
            f"swapped to asyncio.sleep, the gather arm lands {result['gap_ms']} ms from the "
            f"threads. The context-switch saving the exercise predicts is real and far below "
            "the resolution of a three-call fan-out: it is a throughput argument for "
            "thousands of concurrent calls, not a latency argument for three",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
