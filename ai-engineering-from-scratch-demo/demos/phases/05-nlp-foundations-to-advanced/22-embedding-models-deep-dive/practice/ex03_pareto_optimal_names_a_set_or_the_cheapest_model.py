"""Exercise 3 — Pareto-optimal names a set, or the cheapest model.

    **Hard.** Run MTEB on three candidate models across your top-2 domain tasks.
    Report MTEB score, p99 latency on a 100-query batch, and $/1M queries. Pick
    the Pareto-optimal one.

Reading of the exercise: `mteb` is not installed and neither is any encoder, so
the three candidates are the lesson's own `hash_embed` at 256, 128 and 64
dimensions, and the two domain tasks are MRR and recall@10 over exercise 1's
corpus. Both are measured; latency and cost are measured on this machine.

"Pick the Pareto-optimal one" has no answer under one task and a surprising
answer under the other. Scored by recall@10 -- 0.8000, 0.7750, 0.7000 -- quality
falls monotonically with width while latency and cost fall with it too, so **no
candidate dominates another and the Pareto set is all three**. Scored by MRR, 64
dimensions ties 256 at **0.9500** while costing a third of the latency, so it
dominates both others and **the set collapses to the cheapest model**. Same three
models, same measurements, opposite conclusions.

The two tasks disagree because they measure different parts of the ranking. MRR
reads only the first relevant hit; recall@10 reads all four. Their orderings are
`256 = 64 > 128` and `256 > 128 > 64`. An MTEB-style mean over "your top-2 tasks"
is decided by which two you pick, and the exercise leaves that to the reader
while asking for a single number.

Two of the three axes are one axis. Dollars per million queries is seconds per
query times a price -- a positive scalar multiple -- so it reproduces the latency
ordering exactly and can never change the Pareto set. Reporting both makes the
frontier look two-dimensional when it is a line.

And p99 on a 100-query batch is one observation: the 99th of 100 sorted samples,
the second-largest in the batch. Repeating the batch moves it by roughly 9-11%,
which is well inside the 200%+ gap between the widest and narrowest candidate --
so the axis separates them here because this workload has no tail, not because
one observation is a percentile.

Structure: `quality` scores one width on both tasks; `latency` times a
100-query batch and returns its p99; `dominated` is the Pareto test; `frontier`
returns the non-dominated set under a named quality metric.
"""

from __future__ import annotations

import importlib.util
import pathlib
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "22-embedding-models-deep-dive"

CORPUS = practice.load_module(
    pathlib.Path(__file__).resolve().parent / "ex01_truncating_a_hash_is_not_matryoshka.py")
DOCS, GOLD, QUERIES, PER_TOPIC = CORPUS.DOCS, CORPUS.GOLD, CORPUS.QUERIES, CORPUS.PER_TOPIC

UNAVAILABLE = ("mteb", "sentence_transformers", "transformers", "torch")
CANDIDATES, TASKS = (256, 128, 64), ("mrr", "recall")
BATCH, REPEATS, PRICE = 100, 9, 1.0


def quality(ref, width):
    """MRR and recall@10 for one candidate -- the two domain tasks."""
    corpus = [ref.hash_embed(doc, width) for doc in DOCS]
    reciprocal, hits = 0.0, 0.0
    for query, topic in QUERIES:
        order = ref.rank(corpus, ref.hash_embed(query, width))
        reciprocal += 1 / next(p for p, (_, i) in enumerate(order, 1) if GOLD[i] == topic)
        hits += sum(1 for _, i in order[:10] if GOLD[i] == topic) / PER_TOPIC
    return {"mrr": round(reciprocal / len(QUERIES), 4), "recall": round(hits / len(QUERIES), 4)}


def latency(ref, width):
    """Microseconds per query and the p99 of one 100-query batch."""
    corpus = [ref.hash_embed(doc, width) for doc in DOCS]
    samples = []
    for step in range(BATCH):
        query = QUERIES[step % len(QUERIES)][0]
        start = time.perf_counter()
        ref.rank(corpus, ref.hash_embed(query, width))
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return {"p99": samples[int(0.99 * BATCH) - 1], "median": statistics.median(samples)}


def dominated(rows, width, metric):
    """Is some other candidate at least as good on all three axes and better on one."""
    axes = lambda row: (row[metric], -row["p99"], -row["cost"])  # noqa: E731
    mine = axes(rows[width])
    return any(all(a >= b for a, b in zip(axes(row), mine)) and axes(row) != mine
               for other, row in rows.items() if other != width)


def measure(ref, width):
    """One candidate's row: both tasks, p99 with its run-to-run spread, median, cost."""
    runs = [latency(ref, width) for _ in range(REPEATS)]
    p99s = [run["p99"] for run in runs]
    median = statistics.median(run["median"] for run in runs)
    return dict(
        quality(ref, width),
        p99=round(statistics.median(p99s), 1),
        spread=round((max(p99s) - min(p99s)) / statistics.median(p99s) * 100, 1),
        median=round(median, 1),
        cost=round(median * 1e-6 * 1e6 * PRICE / 3600, 6),
    )


def column(rows, key):
    """One axis across the candidates, in declaration order."""
    return [rows[width][key] for width in CANDIDATES]


def compare(rows):
    """Non-dominated sets and orderings under each quality metric."""
    return {
        "frontier": {m: [w for w in CANDIDATES if not dominated(rows, w, m)] for m in TASKS},
        "order": {m: sorted(CANDIDATES, key=lambda w: -rows[w][m]) for m in TASKS},
        "cost_ratio": sorted(set(round(c / t, 6) for c, t in
                                 zip(column(rows, "cost"), column(rows, "median")))),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {width: measure(ref, width) for width in CANDIDATES}
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "batch": BATCH, "rows": rows, **compare(rows),
        "gap": round((rows[256]["p99"] - rows[64]["p99"]) / rows[64]["p99"] * 100, 1),
    }


def verify(result):
    rows, frontier = result["rows"], result["frontier"]
    return [
        practice.Check(
            "ANSWER: under recall@10 every candidate is Pareto-optimal",
            len(frontier["recall"]) == len(CANDIDATES),
            f"{result['absent']} are all absent, so the candidates are `hash_embed` at "
            f"{list(CANDIDATES)} dimensions. Recall@10 runs "
            f"{column(rows, 'recall')} while p99 latency runs "
            f"{column(rows, 'p99')}us -- quality and cost fall together, so no "
            f"candidate dominates another and the frontier is {frontier['recall']}",
        ),
        practice.Check(
            "FINDING: under MRR the frontier collapses to the cheapest model",
            frontier["mrr"] == [min(CANDIDATES)],
            f"MRR runs {column(rows, 'mrr')}: 64 dimensions ties 256 at "
            f"{rows[64]['mrr']} at a third of the latency, so it dominates both others and the "
            f"frontier is {frontier['mrr']}. Same models, same run, opposite conclusion",
        ),
        practice.Check(
            "MECHANISM: the two tasks order the candidates differently",
            result["order"]["mrr"] != result["order"]["recall"],
            f"MRR reads only the first relevant hit, recall@10 reads all {PER_TOPIC}: the "
            f"orderings are {result['order']['mrr']} and {result['order']['recall']}. A mean over "
            "'your top-2 tasks' is decided by which two, and the exercise leaves that open",
        ),
        practice.Check(
            "MECHANISM: two of the three axes are the same axis",
            len(result["cost_ratio"]) == 1,
            f"dollars per million queries is seconds per query times a price, so cost/median is "
            f"the single constant {result['cost_ratio'][0]} across all three candidates: "
            f"{column(rows, 'cost')} against "
            f"{column(rows, 'median')}us. A positive scalar multiple cannot "
            "reorder anything or change the Pareto set",
        ),
        practice.Check(
            "FINDING: p99 on a 100-query batch is a single order statistic",
            all(rows[w]["spread"] > 0 for w in CANDIDATES),
            f"it is the 99th of {result['batch']} sorted samples -- the second-largest in the "
            f"batch. Across {REPEATS} repeats it moves by "
            f"{column(rows, 'spread')}%, against a {result['gap']}% gap between "
            "the widest and narrowest candidate. The axis separates them because this workload "
            "has no tail, not because one observation is a percentile",
        ),
        practice.Check(
            "CONTROL: the quality axis is real -- recall does fall with width",
            rows[256]["recall"] > rows[128]["recall"] > rows[64]["recall"],
            f"recall@10 {column(rows, 'recall')} is strictly decreasing, so the "
            "frontier under recall is a genuine trade rather than a measurement artefact. It is "
            "MRR, the metric that saturates, that makes the cheapest model look free",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
