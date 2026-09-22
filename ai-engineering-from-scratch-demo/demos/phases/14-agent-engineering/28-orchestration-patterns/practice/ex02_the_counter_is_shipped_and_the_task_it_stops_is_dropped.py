"""Exercise 2 — the counter is shipped, and the task it stops is dropped.

    Add a hop counter to the swarm: refuse after 3 handoffs. Does it catch
    A->B->A bouncing?

Reading of the exercise: `swarm` already reads `while hops < 3`, so the
counter is there. Whether it catches bouncing is a question about
`classify`, which is a pure function of the task text -- the same agent
asked twice gets the same answer, so the second hop always lands. Bouncing
needs a router whose answer depends on where it is standing, and only then
can the counter be tested.

**ANSWER: no -- with the shipped router bouncing cannot occur, and with one
that bounces the counter stops the loop and drops the task.** Over the
lesson's 3 tasks the maximum hop count is **1** and the counter never fires.
Swapping in a router that alternates between two specialists takes hops to
**3** on **3** of **3** tasks, and the loop exits by condition rather than by
`break`: **0** result lines are appended and the function returns normally.

**FINDING: the counter bounds the loop and records nothing.** `hops` is a
local, `swarm` returns `(trace, ops)`, and the exit path appends no line --
so a caller sees a shorter trace and a lower op count and cannot distinguish
"handled cheaply" from "gave up". The bouncing run produces **6** trace lines
and **0** answers against the healthy run's **3** and **3**.

**FINDING: refusing after 3 handoffs is not the same as detecting a cycle.**
A router that cycles through **4** specialists never repeats within the
budget, so the counter fires on a legitimate long route and misses the loop:
**3** hops spent, **0** repeats seen. Remembering visited agents catches the
2-cycle in **2** hops and lets the 4-step route finish, at the cost of one
set per task.

**FINDING: the op count keeps rising while nothing progresses.** Each
bouncing iteration spends **1** op on `classify`, so a dropped task costs
**3** ops -- **50%** more than the **2** a healthy handoff costs -- and the
aggregate op count goes *up* as routing gets worse. A cost metric that rises
on failure cannot be used as a health signal without the answer count beside
it.

Structure: `run_swarm()` is the shipped loop with a pluggable router;
`bouncing()` and `cycling()` are the routers that make the counter testable.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "28-orchestration-patterns"
TASKS = ("I need a refund for invoice 4711", "the CLI crashes on ctrl-c",
         "do you offer volume pricing?")
FOUR = ("refund", "bug", "sales", "billing")


def run_swarm(ref, tasks, router, limit=3, remember=False):
    """The shipped swarm loop, with the router and the cycle rule pluggable."""
    trace, ops, hops_used = [], 0, []
    for task in tasks:
        current, hops, seen = list(ref.SPECIALISTS)[0], 0, {list(ref.SPECIALISTS)[0]}
        while hops < limit:
            ops += 1
            label = router(task, current)
            if current == label:
                trace.append(f"swarm[{current}]: handled")
                break
            if remember and label in seen:
                trace.append(f"swarm[{current}] cycle to {label}")
                break
            seen.add(label)
            trace.append(f"swarm[{current}] handoff -> {label}")
            current, hops = label, hops + 1
        hops_used.append(hops)
    return {"trace": trace, "ops": ops, "hops": hops_used,
            "answers": sum("handled" in line for line in trace),
            "cycles": sum("cycle to" in line for line in trace)}


def bouncing(task, current):
    """A router whose answer depends on where it is standing: refund <-> bug."""
    return "bug" if current == "refund" else "refund"


def cycling(task, current):
    """A legitimate four-step route: refund -> bug -> sales -> billing, then stop."""
    if current == FOUR[-1]:
        return current
    return FOUR[(FOUR.index(current) + 1) % len(FOUR)] if current in FOUR else FOUR[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    healthy = run_swarm(ref, TASKS, lambda task, _c: ref.classify(task))
    bounced = run_swarm(ref, TASKS, bouncing)
    guarded = run_swarm(ref, TASKS, bouncing, remember=True)
    long_route = run_swarm(ref, TASKS[:1], cycling)
    long_guarded = run_swarm(ref, TASKS[:1], cycling, remember=True, limit=5)
    shipped_trace, shipped_ops = ref.swarm(list(TASKS))
    return {
        "tasks": len(TASKS), "shipped_ops": shipped_ops,
        "parity": healthy["ops"] == shipped_ops,
        "max_hops": max(healthy["hops"]), "healthy_answers": healthy["answers"],
        "bounced_hops": bounced["hops"], "bounced_answers": bounced["answers"],
        "bounced_lines": len(bounced["trace"]), "bounced_ops": bounced["ops"],
        "guarded_hops": guarded["hops"], "guarded_cycles": guarded["cycles"],
        "long_hops": long_route["hops"][0], "long_answers": long_route["answers"],
        "long_guarded_answers": long_guarded["answers"],
        "healthy_lines": len(shipped_trace),
        "per_task_healthy": healthy["ops"] // len(TASKS),
        "per_task_bounced": bounced["ops"] // len(TASKS),
        "returns": ref.swarm.__annotations__.get("return"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: no -- bouncing cannot occur, and when it does the task is dropped",
            all([result["max_hops"] == 1, result["parity"] is True,
                 result["bounced_hops"] == [3, 3, 3],
                 result["bounced_answers"] == 0,
                 result["healthy_answers"] == 3, result["tasks"] == 3]),
            f"with the shipped router the maximum hop count is {result['max_hops']} and "
            f"the counter never fires (op parity with the shipped swarm: "
            f"{result['parity']}). A router that alternates takes hops to "
            f"{result['bounced_hops']}, and the loop exits by condition rather than "
            f"break: {result['bounced_answers']} answers against "
            f"{result['healthy_answers']}",
        ),
        practice.Check(
            "FINDING: the counter bounds the loop and records nothing",
            all([result["bounced_lines"] == 9, result["bounced_answers"] == 0,
                 result["returns"] == "tuple[list[str], int]"]),
            f"hops is a local and swarm returns {result['returns']}, and the exit path "
            f"appends no line -- so the bouncing run gives {result['bounced_lines']} "
            f"trace lines and {result['bounced_answers']} answers, and a caller cannot "
            "tell 'handled cheaply' from 'gave up'",
        ),
        practice.Check(
            "FINDING: refusing after 3 handoffs is not detecting a cycle",
            all([result["long_hops"] == 3, result["long_answers"] == 0,
                 result["long_guarded_answers"] == 1,
                 result["guarded_cycles"] == 3, result["guarded_hops"] == [1, 1, 1]]),
            f"a router cycling through four specialists never repeats inside the budget, "
            f"so the counter fires at {result['long_hops']} hops with "
            f"{result['long_answers']} answers. Remembering visited agents catches the "
            f"2-cycle in 1 hop ({result['guarded_hops']}, {result['guarded_cycles']} "
            f"cycles named) and lets the 4-step route finish "
            f"({result['long_guarded_answers']})",
        ),
        practice.Check(
            "FINDING: the op count rises while nothing progresses",
            all([result["per_task_bounced"] == 3, result["per_task_healthy"] == 1,
                 result["bounced_ops"] > result["shipped_ops"]]),
            f"each bouncing iteration spends one op on the router, so a dropped task "
            f"costs {result['per_task_bounced']} ops against a handled one's "
            f"{result['per_task_healthy']}, and the total rises from "
            f"{result['shipped_ops']} to {result['bounced_ops']}. A cost metric that goes "
            "up on failure is not a health signal without the answer count beside it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
