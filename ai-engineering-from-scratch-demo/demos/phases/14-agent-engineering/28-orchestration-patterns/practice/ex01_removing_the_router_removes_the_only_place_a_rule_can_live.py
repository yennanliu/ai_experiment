"""Exercise 1 — removing the router removes the only place a rule can live.

    Convert a supervisor-worker to a swarm by removing the router. What
    breaks? What improves?

Reading of the exercise: both patterns ship, so the conversion is already
done and the question is what the diff costs. The two functions route with
the same `classify`, so the answer cannot be accuracy -- it has to be found
in op count, in where the first hop starts, and in what each topology has a
place to put.

**ANSWER: the swarm is 17% cheaper on this workload and loses the only seat
a policy can occupy.** Over the lesson's 3 tasks the supervisor spends **6**
ops and the swarm **5**, because the one task that happens to start at the
right agent needs no handoff. Both produce the same **3** answers, so nothing
improves except cost.

**FINDING: the swarm's cost depends on dictionary insertion order.** It
starts at `list(SPECIALISTS)[0]`, currently `"refund"`. On an 8-task workload
skewed to refunds that costs **10** ops; reordering `SPECIALISTS` so
`"sales"` comes first costs **15** -- a **50%** swing with no change to any
agent, any task, or any answer. The swarm's saving is a bet on the traffic
mix matching the dictionary.

**FINDING: the supervisor's cost is flat and the swarm's is not.** The
supervisor is **2** ops per task whatever the task is; the swarm is **1** or
**2** depending on where it started. Flat cost is what makes a supervisor
capacity-plannable, and the swarm's advantage here is entirely the tasks that
land on their own agent -- **1** of **3** in this mix, and it is decided by which key
happens to be first.

**FINDING: the hop counter has nowhere to live in a supervisor and nowhere
to be read in a swarm.** `swarm` carries `hops` as a local that is never
returned, and `supervisor_worker` has no loop to bound. The swarm's saving
comes from deleting the seat where a routing policy, a budget or an audit
record would sit -- which is the lesson's "harder to reason about (no single
point of control)" stated as an arithmetic fact.

Structure: `profile()` runs both shipped functions; `reordered()` rebuilds
the specialist table to expose the ordering dependency.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "28-orchestration-patterns"
TASKS = ("I need a refund for invoice 4711", "the CLI crashes on ctrl-c",
         "do you offer volume pricing?")


def results(trace):
    """The answers a pattern actually produced, ignoring routing chatter."""
    return [line.split(": ", 1)[1] for line in trace if ": " in line
            and "handoff" not in line]


def profile(ref, fn, tasks=TASKS):
    trace, ops = fn(list(tasks))
    return {"ops": ops, "answers": results(trace), "lines": len(trace)}


SKEWED = TASKS[:1] * 6 + TASKS[1:]


def reordered(ref, first, tasks):
    """The same swarm with SPECIALISTS reordered, so the start agent changes."""
    original = dict(ref.SPECIALISTS)
    ref.SPECIALISTS.clear()
    ref.SPECIALISTS.update({first: original[first],
                            **{k: v for k, v in original.items() if k != first}})
    try:
        return ref.swarm(list(tasks))[1]
    finally:
        ref.SPECIALISTS.clear()
        ref.SPECIALISTS.update(original)


def per_task(ref, fn):
    return [fn([task])[1] for task in TASKS]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    supervisor = profile(ref, ref.supervisor_worker)
    swarm = profile(ref, ref.swarm)
    source = inspect.getsource(ref.swarm)
    return {
        "tasks": len(TASKS),
        "supervisor_ops": supervisor["ops"], "swarm_ops": swarm["ops"],
        "saving": round(100 * (supervisor["ops"] - swarm["ops"])
                        / supervisor["ops"]),
        "same_answers": supervisor["answers"] == swarm["answers"],
        "answers": len(swarm["answers"]),
        "start_agent": list(ref.SPECIALISTS)[0],
        "skewed": len(SKEWED),
        "skew_matched": reordered(ref, "refund", SKEWED),
        "skew_mismatched": reordered(ref, "sales", SKEWED),
        "swing": round(100 * (reordered(ref, "sales", SKEWED)
                              - reordered(ref, "refund", SKEWED))
                       / reordered(ref, "refund", SKEWED)),
        "supervisor_per_task": per_task(ref, ref.supervisor_worker),
        "swarm_per_task": per_task(ref, ref.swarm),
        "hops_returned": "hops" in (ref.swarm.__code__.co_names
                                    + ref.swarm.__code__.co_varnames)
        and "hops" not in source.split("return")[-1],
        "returns": ref.swarm.__annotations__.get("return"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the swarm is 17% cheaper and produces the same answers",
            all([result["supervisor_ops"] == 6, result["swarm_ops"] == 5,
                 result["saving"] == 17, result["same_answers"] is True,
                 result["answers"] == 3, result["tasks"] == 3]),
            f"over {result['tasks']} tasks the supervisor spends "
            f"{result['supervisor_ops']} ops and the swarm {result['swarm_ops']} -- "
            f"{result['saving']}% cheaper -- and the two produce the same "
            f"{result['answers']} answers ({result['same_answers']}), because both route "
            "with the same classify",
        ),
        practice.Check(
            "FINDING: the swarm's cost depends on dictionary insertion order",
            all([result["start_agent"] == "refund", result["skewed"] == 8,
                 result["skew_matched"] == 10, result["skew_mismatched"] == 15,
                 result["swing"] == 50]),
            f"the swarm starts at list(SPECIALISTS)[0], currently "
            f"{result['start_agent']!r}. On a {result['skewed']}-task workload skewed to "
            f"refunds that costs {result['skew_matched']} ops; reordering the table so "
            f"'sales' comes first costs {result['skew_mismatched']} -- a "
            f"{result['swing']}% swing with no change to any agent, task or answer",
        ),
        practice.Check(
            "FINDING: the supervisor's cost is flat and the swarm's is not",
            all([result["supervisor_per_task"] == [2, 2, 2],
                 result["swarm_per_task"] == [1, 2, 2]]),
            f"the supervisor spends {result['supervisor_per_task']} ops per task and the "
            f"swarm {result['swarm_per_task']}. Flat cost is what makes a supervisor "
            "capacity-plannable, and the swarm's advantage is entirely the one task that "
            "lands on its own agent",
        ),
        practice.Check(
            "FINDING: the hop counter has nowhere to be read",
            all([result["hops_returned"] is True,
                 result["returns"] == "tuple[list[str], int]"]),
            f"swarm carries hops as a local and returns {result['returns']} -- the trace "
            "and the op count, not the hop count. The saving comes from deleting the seat "
            "where a routing policy, a budget or an audit record would sit",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
