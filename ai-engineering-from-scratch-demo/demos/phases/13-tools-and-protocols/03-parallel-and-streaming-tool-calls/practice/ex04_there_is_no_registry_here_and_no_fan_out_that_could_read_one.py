"""Exercise 4 — there is no registry here, and no fan-out that could read one.

    Pick two tools that should NOT parallelize (e.g. `create_file` then
    `write_file`). Add an `ordering_dependency` graph to the registry and gate
    the parallel fan-out on that graph. This is the minimum machinery for
    dependency-aware scheduling, which a future agent-engineering phase
    formalizes.

Reading of the exercise: the graph and the gate are both built, and the two
things the exercise says to attach them to are looked for first. Neither is in
this lesson -- the registry is Lesson 13.01's and the fan-out here maps one
function over a list of cities -- so the scheduler is built standalone and run
against the lesson's own three-call workload, which is what makes the cost of
the gate measurable.

**ANSWER: `create_file -> write_file -> read_file`, and the gate is a
topological layering.** Tools with an edge between them land in different
layers; independent tools share one. The scheduler fans out per layer and
waits, so a chain of 3 costs **3** sequential windows while 3 independent tools
cost **1**.

**FINDING: this lesson has no registry to add the graph to.** Its module
namespace holds `executor_weather`, `run_sequential`, `run_parallel`,
`StreamAccumulator`, `CallBuffer` and `fake_openai_stream` -- **no `Tool`, no
`REGISTRY`**. Lesson 13.01 has both. The exercise is written against the wrong
lesson's code.

**FINDING: and there is no fan-out that could consult one.** `run_parallel` is
`pool.map(executor_weather, cities)` -- one function over a list of arguments,
with no tool name anywhere in it. Gating it on a dependency graph would require
the calls to *be* different tools first, which is the thing Lesson 13.01's
registry provides and this lesson dropped.

**FINDING: the gate costs the whole speedup exactly when it is needed.** On the
lesson's own 400/600/800 latencies, three independent tools run in **800 ms**
and a three-long chain in **1,800 ms** -- the sequential time, a **2.25x** loss,
which is `sum/max` again from Exercise 1. A dependency-aware scheduler does not
make ordering cheap; it makes the unsafe version unavailable.

Structure: `layers` is the topological grouping, `schedule_ms` prices a layering
against per-tool latencies, and `namespace` reports what the lesson actually
exports.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "03-parallel-and-streaming-tool-calls"
REGISTRY_LESSON = "01-the-tool-interface"
ORDERING = {"write_file": ("create_file",), "read_file": ("write_file",)}
CHAIN = ("create_file", "write_file", "read_file")
INDEPENDENT = ("get_weather", "get_time", "get_stock_price")
LESSON_LATENCY = (400, 600, 800)


def layers(tools, ordering=ORDERING):
    """Topological layering: a tool waits for every dependency already placed."""
    remaining, placed, out = list(tools), set(), []
    while remaining:
        ready = [tool for tool in remaining
                 if all(dep not in remaining for dep in ordering.get(tool, ()))]
        if not ready:
            raise ValueError(f"cycle among {remaining}")
        out.append(sorted(ready))
        placed.update(ready)
        remaining = [tool for tool in remaining if tool not in placed]
    return out


def schedule_ms(grouping, latencies):
    """Each layer costs its slowest member; layers run one after another."""
    budget, pool = 0, list(latencies)
    for layer in grouping:
        budget += max(pool[:len(layer)])
        pool = pool[len(layer):] or pool
    return budget


def namespace(ref):
    return sorted(name for name in dir(ref) if not name.startswith("_"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    registry_lesson = parity.load_reference(PHASE, REGISTRY_LESSON, "main")
    chained, free = layers(CHAIN), layers(INDEPENDENT)
    exported = namespace(ref)
    return {
        "chain_layers": chained, "free_layers": free,
        "chain_windows": len(chained), "free_windows": len(free),
        "chain_ms": schedule_ms(chained, LESSON_LATENCY),
        "free_ms": schedule_ms(free, LESSON_LATENCY),
        "loss": round(schedule_ms(chained, LESSON_LATENCY)
                      / schedule_ms(free, LESSON_LATENCY), 2),
        "has_registry": "REGISTRY" in exported, "has_tool": "Tool" in exported,
        "lesson_01_has_registry": "REGISTRY" in namespace(registry_lesson),
        "lesson_01_has_tool": "Tool" in namespace(registry_lesson),
        "exported": exported,
        "fanout_names_a_tool": "name" in str(ref.run_parallel.__code__.co_names),
        "fanout_reads": sorted(ref.run_parallel.__code__.co_names),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the gate is a topological layering -- 3 windows against 1",
            all([result["chain_layers"] == [["create_file"], ["write_file"], ["read_file"]],
                 result["free_layers"] == [sorted(INDEPENDENT)],
                 result["chain_windows"] == 3, result["free_windows"] == 1]),
            f"tools with an edge between them land in different layers and independent ones "
            f"share a layer: {CHAIN} groups as {result['chain_layers']} and {INDEPENDENT} as "
            f"{result['free_layers']}. The scheduler fans out per layer and waits, so a "
            f"chain of 3 costs {result['chain_windows']} sequential windows and 3 "
            f"independent tools cost {result['free_windows']}",
        ),
        practice.Check(
            "FINDING: this lesson has no registry to add the graph to",
            all([not result["has_registry"], not result["has_tool"],
                 result["lesson_01_has_registry"], result["lesson_01_has_tool"]]),
            f"the module exports {result['exported']} -- no Tool and no REGISTRY. Lesson "
            f"13.01 has both ({result['lesson_01_has_tool']}, "
            f"{result['lesson_01_has_registry']}), so the exercise is written against the "
            "wrong lesson's code",
        ),
        practice.Check(
            "FINDING: and there is no fan-out that could consult one",
            all([not result["fanout_names_a_tool"],
                 "executor_weather" in result["fanout_reads"]]),
            f"run_parallel reads {result['fanout_reads']} -- it maps one function over a "
            f"list of arguments, with no tool name anywhere in it. Gating that on a "
            "dependency graph would require the calls to be different tools first, which is "
            "what Lesson 13.01's registry provides and this lesson dropped",
        ),
        practice.Check(
            "FINDING: the gate costs the whole speedup exactly when it is needed",
            all([result["free_ms"] == 800, result["chain_ms"] == 1800,
                 result["loss"] == 2.25]),
            f"on the lesson's own {list(LESSON_LATENCY)} ms latencies, three independent "
            f"tools run in {result['free_ms']} ms and a three-long chain in "
            f"{result['chain_ms']:,} ms -- the sequential time, a {result['loss']}x loss, "
            "which is sum/max from Exercise 1 again. A dependency-aware scheduler does not "
            "make ordering cheap; it makes the unsafe version unavailable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
