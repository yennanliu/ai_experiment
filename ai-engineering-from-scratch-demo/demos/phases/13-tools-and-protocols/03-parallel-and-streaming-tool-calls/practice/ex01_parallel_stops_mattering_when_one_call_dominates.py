"""Exercise 1 — parallel stops mattering when one call dominates.

    Run `code/main.py` and vary the simulated latencies. Confirm that the
    parallel-to-sequential ratio is approximately `max/sum` (real runs deviate
    slightly from the ideal because of thread scheduling, serialization, and
    harness overhead). At what latency distribution does parallel stop
    mattering?

Reading of the exercise: the ratio is confirmed by running the lesson's own
`run_sequential` and `run_parallel` under patched latencies, and the "at what
distribution" half is answered with arithmetic rather than more timings --
`max/sum` is a closed form, so the crossover is solvable instead of searched
for. Latencies are scaled down 10x so the whole exercise costs under a second;
the ratio is scale-free, which is itself one of the answers.

**ANSWER: the ratio is `max/sum`, and parallel stops mattering when one call is
most of the sum.** Speedup is `sum/max`, so it is bounded by the number of calls
and reached only when they are equal: three equal calls give **3.00x**, the
lesson's 400/600/800 gives **2.25x**, and 1000/10/10 gives **1.02x**. The
crossover is a property of the *spread*, not the magnitude -- doubling every
latency leaves the ratio unchanged.

**FINDING: the deviation from ideal is a fixed cost per call, not a percentage.**
Against a 180 ms ideal the sequential arm measured **~195 ms** and against an
80 ms ideal the parallel arm measured **~86 ms** -- roughly **5 ms per
`time.sleep`**, which appears three times sequentially and once in parallel.
That is why the measured speedup lands slightly *above* ideal rather than below:
the overhead is paid three times by the arm being divided and once by the
divisor.

**FINDING: the tool the lesson parallelises is not replayable.**
`executor_weather` returns `hash(city) % 35`, and Python salts `hash()` per
process. Bengaluru reads 5 degrees in one run and 2 in the next. Within a
process it is stable, so sequential and parallel agree with each other -- the
comparison the lesson makes is sound -- but no test can pin the value, and a
cache keyed on the city would be wrong the moment the process restarts.

Structure: `ideal` is the closed form, `crossover` solves it for the spread at
which parallelism buys a stated margin, and `timed` runs the lesson's own two
arms under a patched latency table.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "03-parallel-and-streaming-tool-calls"
SCALE = 10                      # divide the lesson's latencies to keep this under a second
LESSON_LATENCY = {"Bengaluru": 400, "Tokyo": 600, "Zurich": 800}
SPREADS = {
    "equal 500/500/500": {"a": 500, "b": 500, "c": 500},
    "lesson 400/600/800": {"a": 400, "b": 600, "c": 800},
    "skewed 800/50/50": {"a": 800, "b": 50, "c": 50},
    "dominated 1000/10/10": {"a": 1000, "b": 10, "c": 10},
}
NOT_WORTH_IT = 1.1              # a speedup below this is not worth the machinery


def ideal(latencies):
    """Speedup a perfect fan-out would give: sum over max."""
    return round(sum(latencies.values()) / max(latencies.values()), 2)


def crossover(others, margin=NOT_WORTH_IT):
    """Largest single latency for which `others` still buy `margin`, in ms."""
    return round(sum(others) / (margin - 1), 1)


def timed(ref, latencies):
    """The lesson's own two arms under a patched latency table."""
    original = dict(ref.SIMULATED_LATENCY_MS)
    ref.SIMULATED_LATENCY_MS.clear()
    ref.SIMULATED_LATENCY_MS.update(latencies)
    try:
        cities = list(latencies)
        with contextlib.redirect_stdout(io.StringIO()):
            sequential_ms, sequential = ref.run_sequential(cities)
            parallel_ms, parallel = ref.run_parallel(cities)
        return {"sequential_ms": sequential_ms, "parallel_ms": parallel_ms,
                "speedup": sequential_ms / parallel_ms, "agree": sequential == parallel}
    finally:
        ref.SIMULATED_LATENCY_MS.clear()
        ref.SIMULATED_LATENCY_MS.update(original)


def repeatable(ref, city="Bengaluru"):
    """Does the executor give the same reading twice inside one process?"""
    original = dict(ref.SIMULATED_LATENCY_MS)
    ref.SIMULATED_LATENCY_MS.clear()
    ref.SIMULATED_LATENCY_MS[city] = 0
    try:
        return ref.executor_weather(city) == ref.executor_weather(city)
    finally:
        ref.SIMULATED_LATENCY_MS.clear()
        ref.SIMULATED_LATENCY_MS.update(original)


def scaled(latencies, scale=SCALE):
    return {name: value // scale for name, value in latencies.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    run = timed(ref, scaled(LESSON_LATENCY))
    ideals = {name: ideal(values) for name, values in SPREADS.items()}
    doubled = ideal({name: value * 2 for name, value in SPREADS["lesson 400/600/800"].items()})
    return {
        "ideals": ideals,
        "scale_free": doubled == ideals["lesson 400/600/800"],
        "best": max(ideals.values()), "worst": min(ideals.values()),
        "calls": len(SPREADS["equal 500/500/500"]),
        "crossover_ms": crossover([10, 10]),
        "measured_speedup": round(run["speedup"], 2),
        "ideal_speedup": ideal(scaled(LESSON_LATENCY)),
        "near_ideal": abs(run["speedup"] - ideal(scaled(LESSON_LATENCY))) < 0.6,
        "sequential_overhead": round(run["sequential_ms"] - sum(scaled(LESSON_LATENCY).values()), 1),
        "parallel_overhead": round(run["parallel_ms"] - max(scaled(LESSON_LATENCY).values()), 1),
        "arms_agree": run["agree"],
        "repeatable_in_process": repeatable(ref),
    }


def verify(result):
    ideals = result["ideals"]
    return [
        practice.Check(
            "ANSWER: the ratio is max/sum, and speedup is bounded by the call count",
            all([ideals["equal 500/500/500"] == 3.0,
                 ideals["lesson 400/600/800"] == 2.25,
                 ideals["skewed 800/50/50"] == 1.12,
                 ideals["dominated 1000/10/10"] == 1.02,
                 result["best"] == result["calls"], result["scale_free"]]),
            f"speedup is sum/max, so it is capped at the number of calls "
            f"({result['calls']}) and reaches it only when they are equal: {ideals}. "
            f"Doubling every latency leaves the ratio unchanged, so the crossover is a "
            f"property of the spread and not the magnitude",
        ),
        practice.Check(
            "ANSWER: parallel stops mattering once one call is most of the sum",
            all([ideals["dominated 1000/10/10"] < NOT_WORTH_IT,
                 ideals["skewed 800/50/50"] > NOT_WORTH_IT,
                 result["crossover_ms"] == 200.0]),
            f"below a {NOT_WORTH_IT}x speedup the fan-out is not worth its machinery. With "
            f"two 10 ms companions, that threshold is crossed once the third call passes "
            f"{result['crossover_ms']} ms -- 1000/10/10 gives "
            f"{ideals['dominated 1000/10/10']}x and is not worth parallelising, while "
            f"800/50/50 still gives {ideals['skewed 800/50/50']}x",
        ),
        practice.Check(
            "FINDING: the deviation from ideal is a fixed cost per call, not a percentage",
            all([result["near_ideal"], result["sequential_overhead"] > 0,
                 result["parallel_overhead"] > 0]),
            f"measured {result['measured_speedup']}x against an ideal "
            f"{result['ideal_speedup']}x. The sequential arm ran "
            f"{result['sequential_overhead']} ms over its {sum(scaled(LESSON_LATENCY).values())} "
            f"ms floor and the parallel arm {result['parallel_overhead']} ms over its "
            f"{max(scaled(LESSON_LATENCY).values())} ms -- neither can undercut its floor, "
            "because time.sleep only ever overshoots. The overhead is one overshoot per "
            "call, paid three times by the arm being divided and once by the divisor, which "
            "is why the measured ratio can land slightly above ideal rather than below",
        ),
        practice.Check(
            "FINDING: the tool the lesson parallelises is not replayable",
            all([result["arms_agree"], result["repeatable_in_process"]]),
            f"executor_weather returns hash(city) % 35, and Python salts hash() per process, "
            f"so a city reads one temperature in this run and another in the next. Within a "
            f"process it is stable -- the two arms agree, so the comparison the lesson makes "
            f"is sound -- but no test can pin the value, and a cache keyed on the city would "
            "be wrong the moment the process restarts",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
