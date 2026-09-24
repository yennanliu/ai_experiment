"""Exercise 3 — the worker with the most tasks is the least loaded.

    Implement a hot-spot detector: log when any worker processes 3× more tasks
    than the slowest worker. What does that indicate about task-duration
    distribution?

Reading of the exercise: "the slowest worker" is read as the one that
completed fewest tasks, since workers are identical and only their tasks
differ; the detector is then run on the shipped workload in every order it
could be queued, and on one it was not designed for, to see what firing means.

**ANSWER: it indicates the durations are spread, and that the worker with the
FEWEST tasks is the one holding the long ones.** In a pull swarm a worker only
takes fewer tasks by being busy longer, so a 3x count gap is a 3x-or-more
duration gap seen from the other side. On 1 task of 1.2s and 12 of 0.1s the
swarm's counts are [1, 4, 4, 4]: the detector fires at 4x and flags workers
1-3 -- whose busy time is 0.4s each -- while worker 0, busy 1.2s, *is* the
makespan. Counting tasks names the wrong worker; summing busy time names the
right one, at the same 3x.

**FINDING: on the lesson's workload the detector is a perfect makespan-loss
alarm.** Over the 70 distinct orders of 4 slow and 4 fast tasks it fires on
54 and stays silent on 16 -- and the 16 are exactly the orders reaching the
0.5s ideal, while the 54 take 0.6s, 0.7s or 0.8s. With two durations, equal
counts means every worker drew one slow and one fast, which is the optimum.

**FINDING: the shipped demo fires it once, on the baseline.** Fixed
assignment's counts {0: 5, 1: 1, 2: 1, 3: 1} trip it at 5x; the swarm's
{2, 2, 2, 2} never do. There, the most-counted worker really is the loaded
one, because tasks were pushed to it rather than pulled by it -- the same
signal means opposite things under push and pull.

Structure: `schedule()` is greedy list scheduling that records which worker
took each task; `detect()` is the exercise's detector and `detect_busy()` the
corrected one. The reference swarm confirms the [1, 4, 4, 4] counts.
"""

from __future__ import annotations

import heapq
import itertools
import time

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "09-parallel-swarm-networks"
RATIO, SCALE = 3, 0.25
SKEWED = [1.2] + [0.1] * 12


def schedule(durations, workers=4):
    """Per-worker (counts, busy seconds) and makespan for a FIFO pull swarm."""
    free, counts, busy = [(0.0, w) for w in range(workers)], [0] * workers, [0.0] * workers
    for duration in durations:
        start, worker = heapq.heappop(free)
        counts[worker], busy[worker] = counts[worker] + 1, busy[worker] + duration
        heapq.heappush(free, (start + duration, worker))
    return counts, [round(b, 6) for b in busy], round(max(t for t, _ in free), 6)


def detect(counts, ratio=RATIO):
    """The exercise's detector: workers whose count is >= ratio x the minimum."""
    floor = min(counts)
    return [w for w, c in enumerate(counts) if c >= ratio * max(floor, 1e-9)]


def detect_busy(busy, ratio=RATIO):
    return [w for w, b in enumerate(busy) if b >= round(ratio * min(busy), 6)]


def reference_counts(ref, durations):
    tasks = [ref.Task(i, d, 0) for i, d in enumerate(durations)]
    original = ref.fake_work
    ref.fake_work = lambda task: time.sleep(task.duration * SCALE)
    try:
        return ref.run_swarm(tasks, 4)[1]
    finally:
        ref.fake_work = original


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts, busy, span = schedule(SKEWED)
    fired, silent = [], []
    for order in set(itertools.permutations([t.duration for t in ref.make_tasks(4)])):
        c, _, s = schedule(order)
        (fired if detect(c) else silent).append(s)
    return {
        "counts": counts, "busy": busy, "span": span,
        "by_count": detect(counts), "by_busy": detect_busy(busy),
        "ref_counts": sorted(reference_counts(ref, SKEWED).values()),
        "fired": sorted(set(fired)), "n_fired": len(fired),
        "silent": sorted(set(silent)), "n_silent": len(silent),
        "fixed": detect([5, 1, 1, 1]), "swarm": detect([2, 2, 2, 2]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: durations are spread, and the fewest-task worker holds the long ones",
            all([result["counts"] == [1, 4, 4, 4], result["ref_counts"] == [1, 4, 4, 4],
                 result["by_count"] == [1, 2, 3], result["by_busy"] == [0],
                 result["busy"][0] == result["span"]]),
            f"on 1x1.2s + 12x0.1s the counts are {result['counts']} (reference: "
            f"{result['ref_counts']}) and the detector flags workers {result['by_count']}, "
            f"busy {result['busy'][1]}s each, while worker 0 is busy {result['busy'][0]}s "
            f"= the makespan; busy time flags {result['by_busy']}",
        ),
        practice.Check(
            "FINDING: on the lesson's workload it is a perfect makespan-loss alarm",
            all([result["n_fired"] == 54, result["n_silent"] == 16,
                 result["silent"] == [0.5], min(result["fired"]) > 0.5]),
            f"over 70 orders it fires on {result['n_fired']} (makespans "
            f"{result['fired']}) and is silent on {result['n_silent']} (makespans "
            f"{result['silent']}) -- equal counts means one slow and one fast each",
        ),
        practice.Check(
            "FINDING: the shipped demo fires it once, on the baseline",
            result["fixed"] == [0] and result["swarm"] == [],
            f"fixed assignment's {{5, 1, 1, 1}} flags worker {result['fixed']}, which "
            "really is loaded because tasks were pushed to it; the swarm's {2, 2, 2, 2} "
            "flags none -- the same signal means opposite things under push and pull",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
