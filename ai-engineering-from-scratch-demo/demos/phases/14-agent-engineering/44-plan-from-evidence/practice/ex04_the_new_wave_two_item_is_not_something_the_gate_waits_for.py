"""Exercise 4 — the new wave-two item is not something the gate waits for.

    Add a work item that can run in the second wave without touching either
    existing branch.

Reading of the exercise: "second wave without touching either branch" pins
the shape exactly -- the item may depend on `contract` and on nothing else.
What the exercise does not say, and what the graph shows immediately, is what
happens at the other end.

**ANSWER: the telemetry item joins wave 2 and the integration gate does not
wait for it.** Adding `telemetry` with `depends_on=("contract",)` gives waves
`[['contract'], ['docs', 'implementation', 'telemetry'], ['integration']]`:
**3** items in wave 2, **0** issues. `integration` still depends on **2** of
the **3**, so the gate closes the plan without ever proving the third.

**FINDING: the plan now has 2 terminal nodes where its author believes it has
one.** Nothing depends on `telemetry`, so it is a sink exactly like
`integration`. "Run the complete acceptance gate" is the name of one of two
ways this plan can end, and the JSON says so plainly once you look for sinks:
**2** of **5** items are terminal.

**FINDING: wave position is an accident, not a guarantee.** There is no path
between `telemetry` and `integration` in either direction -- **0** edges -- so
the only thing putting telemetry first is that it happened to land in an
earlier wave. A resuming session reading the graph is free to run them in
either order, and both orders satisfy every dependency in the plan.

**FINDING: the fix is one edge, and it costs the parallelism nothing.**
Adding `telemetry` to the gate's dependencies keeps **3** waves and the same
wave-2 width; the only thing that changes is that the plan can no longer
finish without the item it added.

Structure: `extended()` adds the item; `reachable()` walks the graph back from
the sinks.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "44-plan-from-evidence"
TELEMETRY = ("telemetry", "Count duplicate rejections by normalized domain",
             ("app/metrics.py:40",), ("contract",),
             "python3 -m unittest tests.test_metrics")


def extended(ref, gate_deps=("implementation", "docs")):
    items = [row for row in ref.example() if row.id != "integration"]
    items.append(ref.WorkItem(*TELEMETRY))
    items.append(ref.WorkItem("integration", "Run the complete acceptance gate",
                              ("pyproject.toml:31",), tuple(gate_deps), "python3 -m unittest"))
    return items


def sinks(items):
    depended_on = {name for item in items for name in item.depends_on}
    return sorted(item.id for item in items if item.id not in depended_on)


def ancestors(items, target):
    """Everything `target` transitively waits on."""
    by_id = {item.id: item for item in items}
    seen, stack = set(), list(by_id[target].depends_on)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(by_id[current].depends_on)
    return seen


def reachable(items):
    """Every item a sink depends on, transitively -- what the gate actually proves."""
    by_id = {item.id: item for item in items}
    seen, stack = set(), list(sinks(items))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(by_id[current].depends_on)
    return seen


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    loose = extended(ref)
    wired = extended(ref, gate_deps=("implementation", "docs", "telemetry"))
    waves = ref.execution_waves(loose)
    covered = ancestors(loose, "integration")
    return {
        "waves": waves, "wave_two": waves[1], "issues": ref.validate(loose),
        "gate_deps": len(next(row.depends_on for row in loose if row.id == "integration")),
        "wave_two_size": len(waves[1]),
        "sinks": sinks(loose), "items": len(loose),
        "covered": sorted(covered),
        "edges_between": sum(name in ancestors(loose, other)
                             for name, other in (("telemetry", "integration"),
                                                 ("integration", "telemetry"))),
        "wired_waves": ref.execution_waves(wired),
        "wired_unreachable": sorted({row.id for row in wired} - reachable(wired)),
        "wired_width": len(ref.execution_waves(wired)[1]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: telemetry joins wave 2 and the gate does not wait for it",
            all([result["wave_two"] == ["docs", "implementation", "telemetry"],
                 result["issues"] == [], result["wave_two_size"] == 3,
                 result["gate_deps"] == 2]),
            f"the plan schedules {result['waves']} with {len(result['issues'])} issues; "
            f"wave two holds {result['wave_two_size']} items and the gate depends on "
            f"{result['gate_deps']} of them, so it closes the plan without proving the third",
        ),
        practice.Check(
            "FINDING: the plan now has 2 terminal nodes where its author believes it has one",
            all([result["sinks"] == ["integration", "telemetry"], result["items"] == 5]),
            f"nothing depends on telemetry, so the sinks are {result['sinks']} -- "
            f"{len(result['sinks'])} of {result['items']} items terminate the plan, and "
            "'run the complete acceptance gate' is the name of one of two ways it can end",
        ),
        practice.Check(
            "FINDING: wave position is an accident, not a guarantee",
            all([result["edges_between"] == 0,
                 "telemetry" not in result["covered"]]),
            f"there are {result['edges_between']} edges between telemetry and integration "
            f"in either direction, and the gate's ancestors are {result['covered']}; the "
            "only thing putting telemetry first is the wave it happened to land in",
        ),
        practice.Check(
            "FINDING: the fix is one edge and costs the parallelism nothing",
            all([len(result["wired_waves"]) == 3, result["wired_width"] == 3,
                 result["wired_unreachable"] == []]),
            f"adding telemetry to the gate's dependencies keeps {len(result['wired_waves'])} "
            f"waves and a wave-two width of {result['wired_width']}; what changes is that "
            "the plan can no longer finish without the item it added",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
