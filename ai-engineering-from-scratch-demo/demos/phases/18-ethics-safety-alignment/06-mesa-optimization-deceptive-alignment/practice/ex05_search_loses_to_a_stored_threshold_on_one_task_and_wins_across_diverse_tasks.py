"""Exercise 5 — search loses to a stored threshold on one task (0.929 vs 1.000) and wins across diverse tasks (0.967 vs 0.629).

    The four conditions for mesa-optimization (Hubinger Section 3) apply to
    modern LLMs. Name one that might not apply to a specific deployment (e.g., a
    narrowly-scoped classifier) and one that does apply even to such systems.

Reading of the exercise: the narrowly-scoped classifier is the reference's
own -- one threshold task, x ~ N(0, 1), label x > 0. Condition 2 (diverse
sub-tasks: "a general optimizer beats task-specific heuristics") is tested by
racing two kinds of learner: a *heuristic* that stores one fitted threshold
per training task (and the mean of them for a task it has not seen), and a
tiny *learned optimizer* that stores nothing task-specific and at inference
searches for the threshold from 8 labelled examples of the task in front of
it. A task is a threshold t with inputs t + x, x drawn by the reference's
`gen_example` (seed 13). Each world has 20 training and 20 unseen test tasks
with t ~ U(-s, s) (seed 6), scored on 200 queries per task; s = 0 is the
narrow classifier, where every task is the same one. Condition 4
(generalization over memorization) is tested on the reference's own shipped
training and deployment sets.

**ANSWER: condition 2 does not apply to a narrow classifier.**

| task spread s | heuristic | searcher |
|---:|---:|---:|
| 0 (narrow) | 1.000 | 0.929 |
| 0.5 | 0.893 | 0.934 |
| 1 | 0.850 | 0.942 |
| 3 | 0.629 | 0.967 |

On the single task the stored threshold is perfect and search is strictly
worse (by 0.071), so nothing rewards an internal optimizer. As soon as the
tasks differ the order flips, and the searcher's lead grows with the spread
to 0.338. Diversity is what makes a general optimizer pay, and a narrowly
scoped deployment removes it.

**ANSWER: condition 4 applies even to the narrow classifier.** A lookup table
of the 500 training inputs scores 1.000 on training and finds 0 of the 500
deployment inputs in its table: with continuous inputs nothing is seen twice,
so any system that works at all must generalize, however narrow its task.
What narrowness changes is that the generalizing rule is one stored number,
not a search -- the pressure exists, but a heuristic meets it.

**FINDING: the advantage of search grows monotonically with diversity** --
heuristic minus searcher is +0.071, -0.041, -0.092, -0.338 across the four
spreads. Condition 3 (capacity) is the other condition that can survive
narrowness: a narrow classifier fine-tuned from a large backbone keeps the
backbone's capacity, which this toy does not model.

Structure: `rows()` draws a labelled task from the seeded reference stream;
`fit()` is the threshold search (midpoint of the class boundary); `race()`
scores heuristic and searcher on the unseen tasks; `lookup()` is the
memorizer on the shipped sets.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "06-mesa-optimization-deceptive-alignment"
N_TRAIN, N_QUERY, SHOTS, TASKS = 500, 200, 8, 20
SPREADS = (0.0, 0.5, 1.0, 3.0)


def rows(ref, t, n):
    return [(t + x, int(t + x > t)) for x in (ref.gen_example(True).x for _ in range(n))]


def fit(data):
    """The searcher: the midpoint between the largest negative and the smallest positive."""
    neg, pos = [x for x, y in data if not y], [x for x, y in data if y]
    if not (neg and pos):
        return min(pos) if pos else max(neg)
    return (max(neg) + min(pos)) / 2


def accuracy(threshold, data):
    return sum(int(x > threshold) == y for x, y in data) / len(data)


def race(ref, train_tasks, test_tasks):
    """(heuristic accuracy, searcher accuracy, thresholds stored) averaged over test tasks."""
    stored = {t: fit(rows(ref, t, N_TRAIN)) for t in train_tasks}
    fallback = sum(stored.values()) / len(stored)
    heur, search = [], []
    for t in test_tasks:
        queries = rows(ref, t, N_QUERY)
        heur.append(accuracy(stored.get(t, fallback), queries))
        search.append(accuracy(fit(rows(ref, t, SHOTS)), queries))
    return round(sum(heur) / len(heur), 3), round(sum(search) / len(search), 3), len(stored)


def lookup(ref):
    """Condition 4 on the shipped sets: a memorizer's train score and deployment coverage."""
    train = [ref.gen_example(True) for _ in range(500)]
    dep = [ref.gen_example(False, 0.3) for _ in range(500)]
    table = {e.x: e.y_base for e in train}
    return sum(table[e.x] == e.y_base for e in train) / 500, sum(e.x in table for e in dep)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, ref.random = ref.random, random.Random(13)
    try:
        memorizer, tasks, sweep = lookup(ref), random.Random(6), {}
        for spread in SPREADS:
            world = [round(tasks.uniform(-spread, spread), 9) for _ in range(2 * TASKS)]
            sweep[spread] = race(ref, world[:TASKS], world[TASKS:])
        return {"memorizer": memorizer, "sweep": sweep}
    finally:
        ref.random = saved


def verify(result):
    sweep = {s: v[:2] for s, v in result["sweep"].items()}
    gaps = [h - g for h, g in sweep.values()]
    return [
        practice.Check(
            "ANSWER: condition 2 does not apply to a narrow classifier",
            sweep == {0.0: (1.0, 0.929), 0.5: (0.893, 0.934), 1.0: (0.85, 0.942), 3.0: (0.629, 0.967)},
            f"(heuristic, searcher) on unseen tasks by task spread: {sweep}",
        ),
        practice.Check(
            "ANSWER: condition 4 applies even to the narrow classifier",
            result["memorizer"] == (1.0, 0),
            f"lookup table: training accuracy {result['memorizer'][0]}, deployment inputs "
            f"found {result['memorizer'][1]}/500",
        ),
        practice.Check(
            "FINDING: the advantage of search grows with diversity",
            [round(g, 3) for g in gaps] == [0.071, -0.041, -0.092, -0.338],
            "heuristic minus searcher by spread: " + ", ".join(f"{g:+.3f}" for g in gaps),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
