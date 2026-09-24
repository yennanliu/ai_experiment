"""Exercise 2 — low priority starves exactly when high priority alone fills the workers.

    Add a priority queue variant (use `queue.PriorityQueue`). Assign priority by
    task "importance" field. Observe whether low-priority tasks ever starve
    under continuous load.

Reading of the exercise: "continuous load" means a producer that keeps adding
work while the workers run, so the variant needs one -- the shipped swarm has
none -- and "ever starve" is answered by sweeping the producer's rate across
the workers' capacity rather than by watching one run.

**ANSWER: yes, and exactly when the high-priority arrivals alone use every
worker.** Four workers, 0.1s tasks, a burst of k high-priority tasks every
0.1s and one low-priority task waiting from t=0. At k=3 the low task runs at
once -- one worker is free every slot. At k=4 it never runs in 100 slots:
there is always a more important task. At k=5 the same, with the backlog
growing by one task per slot. Starvation is a step at 100% high-priority
utilisation, not a gradual effect.

**FINDING: the shipped swarm cannot be under continuous load.** `run_swarm`
fills the queue before starting any worker, and each worker `return`s on the
first `queue.Empty` -- there is no producer, so a worker that sees an empty
queue exits for good. And `Task` has 3 fields, none of them "importance".

**FINDING: `PriorityQueue` of `(priority, Task)` raises on the first tie.**
`Task` is a plain `@dataclass`, not `order=True`, so two tasks of equal
importance make heapq compare the Tasks and raise `TypeError: '<' not
supported`. The entry needs a sequence number between the two.

**FINDING: linear aging needs no re-scoring at all.** A heap cannot re-rank
entries once pushed, which looks fatal to aging. But with effective priority
`p - a * (now - arrival)`, the `a * now` term is common to every entry, so
the static key `p + a * arrival` orders them identically forever. With
a = 0.1 per slot, the low task runs at slot 10 under k=4 and at slot 12 under
k=5 -- bounded where it was never.

**FINDING: aging picks who waits; it does not shrink the wait.** At k=5 the
backlog after 100 slots is 101 tasks with aging and 101 without -- 500
arrivals plus the low task, 400 served either way. Under
overload the queue grows either way; the lesson's back-pressure item is the
fix, aging only moves the starvation onto whichever high task is newest.

Structure: `simulate()` runs slotted virtual time over a real
`queue.PriorityQueue` -- each slot enqueues a burst, then each worker pulls one
0.1s task -- so every number is exact rather than a wall-clock sample.
"""

from __future__ import annotations

import dataclasses
import inspect
import queue

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "09-parallel-swarm-networks"
WORKERS, SLOTS, HIGH, LOW = 4, 100, 0, 1


def simulate(burst, aging=0.0):
    """(slot the low task ran in or None, backlog after SLOTS) for k=burst per slot."""
    pq, seq, ran = queue.PriorityQueue(), 0, None
    pq.put((LOW, seq, "low"))
    for slot in range(SLOTS):
        for _ in range(burst):
            seq += 1
            pq.put((HIGH + aging * slot, seq, "high"))
        for _ in range(min(WORKERS, pq.qsize())):
            if pq.get()[2] == "low":
                ran = slot
    return ran, pq.qsize()


def tie_breaks(ref):
    pq = queue.PriorityQueue()
    pq.put((1, ref.Task(0, 0.1, 0)))
    try:
        pq.put((1, ref.Task(1, 0.1, 0)))
    except TypeError as exc:
        return str(exc)
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    swarm = inspect.getsource(ref.run_swarm)
    plain = {k: simulate(k) for k in (3, 4, 5)}
    aged = {k: simulate(k, aging=0.1) for k in (4, 5)}
    return {
        "plain": plain, "aged": aged, "tie": tie_breaks(ref),
        "fields": [f.name for f in dataclasses.fields(ref.Task)],
        "prefilled": swarm.index("q.put(t)") < swarm.index("th.start()"),
        "exits_on_empty": "except queue.Empty:\n                return" in swarm,
    }


def verify(result):
    plain, aged = result["plain"], result["aged"]
    return [
        practice.Check(
            "ANSWER: yes, exactly when the high-priority arrivals alone use every worker",
            all([plain[3][0] == 0, plain[4][0] is None, plain[5][0] is None]),
            f"with {WORKERS} workers the low task runs at slot {plain[3][0]} under k=3 "
            f"and never in {SLOTS} slots under k=4 or k=5 -- a step at 100% "
            "high-priority utilisation, not a gradual effect",
        ),
        practice.Check(
            "FINDING: the shipped swarm cannot be under continuous load",
            all([result["prefilled"], result["exits_on_empty"],
                 "importance" not in result["fields"]]),
            f"run_swarm fills the queue before starting a worker and each worker returns "
            f"on the first queue.Empty, so there is no producer; Task's fields are "
            f"{result['fields']}",
        ),
        practice.Check(
            "FINDING: PriorityQueue of (priority, Task) raises on the first tie",
            result["tie"] is not None and "not supported" in result["tie"],
            f"Task is a plain @dataclass, so two equal priorities compare the Tasks: "
            f"TypeError: {result['tie']} -- the entry needs a sequence number",
        ),
        practice.Check(
            "FINDING: linear aging needs no re-scoring at all",
            aged[4][0] == 10 and aged[5][0] == 12,
            f"p - a*(now - arrival) differs from the static key p + a*arrival by a*now, "
            f"common to all entries; with a=0.1 the low task runs at slot {aged[4][0]} "
            f"under k=4 and {aged[5][0]} under k=5",
        ),
        practice.Check(
            "FINDING: aging picks who waits; it does not shrink the wait",
            plain[5][1] == aged[5][1] == SLOTS + 1,
            f"at k=5 the backlog after {SLOTS} slots is {aged[5][1]} with aging and "
            f"{plain[5][1]} without, since {WORKERS * SLOTS} are served either way -- under "
            "overload only back-pressure helps",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
