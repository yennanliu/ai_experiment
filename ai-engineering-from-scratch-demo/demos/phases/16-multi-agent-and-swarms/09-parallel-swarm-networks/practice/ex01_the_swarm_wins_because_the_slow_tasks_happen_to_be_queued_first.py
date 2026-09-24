"""Exercise 1 — the swarm wins because the slow tasks happen to be queued first.

    Run `code/main.py`. How much faster is swarm than sequential on the
    variable-duration workload? How much faster than fixed assignment?

Reading of the exercise: the two ratios are one division each, so the work is
in asking what they are ratios *of* -- the swarm's time depends on the order
the queue was filled in, and the fixed baseline's on how tasks were assigned,
and both of those are choices `make_tasks` makes.

**ANSWER: 4.0x faster than sequential and 3.4x faster than fixed assignment.**
The workload is 4 tasks of 0.4s and 4 of 0.1s, 2.0 work-seconds. Sequential
takes 2.0s; fixed assignment takes 1.7s, because worker 0 carries all four slow
tasks *and* task 7, a fast one (`(7 - 3) % 4 == 0`) -- 5 tasks where the demo's
comment implies 4; the swarm takes 0.5s, which is 2.0 / 4 exactly, the ideal.
The reference run, with every sleep scaled by 1/4, reproduces all three.

**FINDING: the swarm is optimal only because the queue is sorted longest-first.**
`make_tasks` enqueues the four slow tasks before the four fast ones, so the
four workers each take one slow task at t=0 -- which is LPT scheduling, not
anything the queue does. The same 8 tasks have 70 distinct orders; the swarm
hits 0.5s on 16 of them, takes 0.6s on 34, and 0.8s on 4 -- the order
`F S S S F F F S` leaves three workers idle while one runs the last slow task.
Against that worst order the speedup over fixed assignment is 2.1x, not 3.4x.

**FINDING: the fixed baseline is built to lose.** Its docstring says so --
"Pre-assignment is pessimal". Assigning `task_id % 4` gives each worker one
slow and one fast task, and finishes in 0.5s: **equal** to the swarm. The 3.4x
measures the `pre_assigned` formula, not supervisor against swarm.

**FINDING: the counts are even, not uneven.** The lesson says the output
"shows per-worker task counts (swarm distributes unevenly but optimally)"; the
swarm's counts are {0: 2, 1: 2, 2: 2, 3: 2}. The demo's own gloss, "slow
workers finish first, fast pull next job", describes nothing that happens --
all four workers are identical and all finish their slow task at 0.4s.

Structure: `makespan()` is greedy list scheduling -- what a FIFO queue drained
by identical workers computes -- so every order is scored exactly rather than
by wall clock; `reference_run()` confirms the shipped functions agree.
"""

from __future__ import annotations

import collections
import heapq
import itertools
import time

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "09-parallel-swarm-networks"
SCALE = 0.25


def makespan(durations, workers=4):
    """Finish time when idle workers pull the next duration off a FIFO queue."""
    free = [0.0] * workers
    for duration in durations:
        heapq.heappush(free, heapq.heappop(free) + duration)
    return round(max(free), 6)


def reference_run(ref, tasks):
    """The shipped schedulers, with fake_work's sleep scaled down by SCALE."""
    original = ref.fake_work
    ref.fake_work = lambda task: time.sleep(task.duration * SCALE)
    try:
        runs = {"sequential": ref.run_sequential(tasks),
                "fixed": ref.run_fixed_assignment(tasks, 4),
                "swarm": ref.run_swarm(tasks, 4)}
    finally:
        ref.fake_work = original
    return {name: (round(wall / SCALE, 2), counts) for name, (wall, counts) in runs.items()}


def worst_load(tasks, owner):
    """Makespan of a fixed assignment: the busiest worker's total."""
    return round(max(sum(t.duration for t in tasks if owner(t) == w) for w in range(4)), 6)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tasks = ref.make_tasks(4)
    durations = [t.duration for t in tasks]
    orders = sorted(set(itertools.permutations(durations)))
    scores = {order: makespan(order) for order in orders}
    worst = max(orders, key=scores.get)
    exact = {"sequential": round(sum(durations), 6), "fixed": worst_load(tasks, lambda t: t.pre_assigned),
             "swarm": makespan(durations)}
    return {
        "exact": exact, "measured": reference_run(ref, tasks),
        "worker0": sum(t.pre_assigned == 0 for t in tasks),
        "orders": len(orders),
        "spread": dict(sorted(collections.Counter(scores.values()).items())),
        "worst": "".join("S" if d > 0.2 else "F" for d in worst),
        "round_robin": worst_load(tasks, lambda t: t.task_id % 4),
        "doc": parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    exact, measured = result["exact"], result["measured"]
    seq, fixed, swarm = exact["sequential"], exact["fixed"], exact["swarm"]
    worst = max(result["spread"])
    agree = all(abs(measured[k][0] - exact[k]) < 0.2 * exact[k] for k in exact)
    return [
        practice.Check(
            "ANSWER: 4.0x faster than sequential, 3.4x faster than fixed",
            all([round(seq / swarm, 2) == 4.0, round(fixed / swarm, 2) == 3.4,
                 result["worker0"] == 5, agree]),
            f"sequential {seq}s, fixed {fixed}s (worker 0 holds {result['worker0']} tasks), "
            f"swarm {swarm}s = the ideal 2.0/4; the reference run, scaled back up, "
            f"measures {measured['sequential'][0]}s, {measured['fixed'][0]}s and "
            f"{measured['swarm'][0]}s",
        ),
        practice.Check(
            "FINDING: the swarm is optimal only because the queue is sorted longest-first",
            all([result["orders"] == 70, result["spread"].get(0.5) == 16, worst == 0.8]),
            f"over the {result['orders']} distinct orders of the same 8 tasks the swarm's "
            f"makespans are {result['spread']}; the order {result['worst']} takes {worst}s, "
            f"cutting the speedup over fixed to {fixed / worst:.1f}x",
        ),
        practice.Check(
            "FINDING: the fixed baseline is built to lose",
            result["round_robin"] == swarm,
            f"assigning task_id % 4 gives each worker one slow and one fast task and "
            f"finishes in {result['round_robin']}s -- equal to the swarm's {swarm}s; the "
            "3.4x measures the pre_assigned formula, not supervisor against swarm",
        ),
        practice.Check(
            "FINDING: the counts are even, not uneven",
            all([measured["swarm"][1] == {0: 2, 1: 2, 2: 2, 3: 2},
                 "distributes unevenly" in result["doc"]]),
            f"the lesson says swarm 'distributes unevenly but optimally'; its counts are "
            f"{measured['swarm'][1]} -- every worker takes one slow task at t=0 and one "
            "fast task at 0.4s",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
