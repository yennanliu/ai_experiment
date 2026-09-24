"""Exercise 5 — splitting the swarm by type rebuilds fixed assignment.

    Convert the swarm demo to use a `queue.Queue` of (task_type, payload)
    tuples, with workers subscribing only to specific types. What routing rules
    make sense when tasks are heterogeneous?

Reading of the exercise: the shipped 8 tasks are typed by their own duration
("slow" and "fast"), 4 workers subscribe 2 and 2, and each routing rule is
kept only if it removes a failure the conversion actually produces.

**ANSWER: four rules, one per failure.**
(1) One queue per type, never a shared queue with put-back: on the single
`queue.Queue`, fast workers pop slow tasks and must re-queue them -- 14 wasted
gets for 8 tasks, and FIFO order is gone. (2) Dead-letter any type with no
subscriber at publish time: a run_swarm-shaped worker exits only on
`queue.Empty`, so one orphan task keeps every worker spinning -- 1000 gets of 1
item before the cap. (3) Size subscriptions by work-seconds, not by type
count. (4) Let an idle specialist steal from other queues -- see below.

**FINDING: every strict partition is at least 60% slower than the generalist
swarm.** The slow type is 1.6 work-seconds and the fast type 0.4. Split 2/2
the slow pool finishes at 0.8s; split 3/1 it is still 0.8s, because 4 tasks
over 3 workers is 2 rounds; 1/3 is 1.6s. The best of the three is 0.8s
against the untyped swarm's 0.5s. Subscribing by type is fixed assignment at
the type level, and brings back exactly the imbalance the lesson credits the
swarm with removing.

**FINDING: stealing recovers most of it, not all.** With 2/2 subscriptions
and idle workers falling back to the other queue, the fast pair finishes its
4 tasks at 0.2s, then takes two slow tasks: 0.6s. The last 0.1s is the price
of the fast workers starting on the cheap type while slow work was waiting.

Structure: `pool()` is a greedy pull simulation over one `queue.Queue` per
type, where each worker scans its subscriptions in order; `shared()` is the
single-queue put-back version the exercise literally describes.
"""

from __future__ import annotations

import collections
import heapq
import queue

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "09-parallel-swarm-networks"
CAP = 1000


def typed(ref):
    return [("slow" if t.duration > 0.2 else "fast", t.duration) for t in ref.make_tasks(4)]


def pool(items, subscriptions):
    """Makespan when worker w pulls from the first non-empty queue in subscriptions[w]."""
    queues = collections.defaultdict(queue.Queue)
    for kind, duration in items:
        queues[kind].put(duration)
    free, finish = [(0.0, w) for w in range(len(subscriptions))], 0.0
    while free:
        now, worker = heapq.heappop(free)
        mine = [queues[k] for k in subscriptions[worker] if not queues[k].empty()]
        if mine:  # a worker with nothing left in its subscriptions retires
            done = now + mine[0].get()
            finish = max(finish, done)
            heapq.heappush(free, (done, worker))
    return round(finish, 6)


def take(q, kinds):
    """Pop until a task of one of `kinds` turns up; re-queue the rest. (task, wasted)."""
    for wasted in range(q.qsize()):
        kind, duration = q.get()
        if kind in kinds:
            return duration, wasted
        q.put((kind, duration))
    return None, q.qsize()


def shared(items, subscriptions):
    """Wasted gets on one queue.Queue where a worker re-queues types it does not take.
    Idle workers re-scan whenever any worker finishes."""
    q, wasted, busy, idle, now = queue.Queue(), 0, [], list(range(len(subscriptions))), 0.0
    for item in items:
        q.put(item)
    while not q.empty() and busy + idle:
        for worker in list(idle):
            duration, lost = take(q, subscriptions[worker])
            wasted += lost
            if duration is not None:
                idle.remove(worker)
                heapq.heappush(busy, (now + duration, worker))
        if not busy:
            break
        now, worker = heapq.heappop(busy)
        idle.append(worker)
    return wasted


def spin(orphan):
    """A run_swarm-shaped worker: it only stops on queue.Empty."""
    q, gets = queue.Queue(), 0
    q.put(orphan)
    while gets < CAP:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return gets
        gets += 1
        q.put(item)
    return gets


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    items = typed(ref)
    split = {f"{s}/{4 - s}": pool(items, [("slow",)] * s + [("fast",)] * (4 - s))
             for s in (1, 2, 3)}
    return {
        "split": split, "generalist": pool(items, [("slow", "fast")] * 4),
        "stealing": pool(items, [("slow", "fast")] * 2 + [("fast", "slow")] * 2),
        "wasted": shared(items, [("slow",)] * 2 + [("fast",)] * 2),
        "spin": spin(("audit", 0.1)), "tasks": len(items),
        "work": {k: round(sum(d for kind, d in items if kind == k), 6) for k in ("slow", "fast")},
    }


def verify(result):
    best = min(result["split"].values())
    return [
        practice.Check(
            "ANSWER: per-type queues, dead-letters, work-second sizing, stealing",
            result["wasted"] == 14 and result["spin"] == CAP,
            f"one shared queue costs {result['wasted']} wasted gets for {result['tasks']} "
            f"tasks; an orphan type keeps a worker that stops only on queue.Empty "
            f"spinning for {result['spin']} gets of one item",
        ),
        practice.Check(
            "FINDING: every strict partition is at least 60% slower than the generalist swarm",
            all([result["split"] == {"1/3": 1.6, "2/2": 0.8, "3/1": 0.8},
                 result["generalist"] == 0.5, best / result["generalist"] >= 1.6]),
            f"work-seconds {result['work']}; makespans by slow/fast split "
            f"{result['split']} against {result['generalist']}s untyped -- subscribing "
            "by type is fixed assignment at the type level",
        ),
        practice.Check(
            "FINDING: stealing recovers most of it, not all",
            result["generalist"] < result["stealing"] < best,
            f"2/2 subscriptions with fallback finish at {result['stealing']}s, between "
            f"{result['generalist']}s untyped and {best}s strictly partitioned",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
