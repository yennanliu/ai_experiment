"""Exercise 5 — the 8% trace fails two diagnostics, and the budget is the bigger one.

    A customer reports only 8% cache hit rate. Name three likely causes and
    the diagnostic you would run for each.

Reading of the exercise: each cause is built as a workload over the lesson's
own cache, and each diagnostic is run on every workload. That tests whether a
diagnostic points to its own cause and to no other. The causes:
(1) inconsistent prompt ordering, which is the lesson's scrambled trace at
8.4%; (2) dynamic content at the front of the prefix, here a per-request
timestamp segment; (3) a KV budget too small for the working set, here 156
blocks. The diagnostics, all run on logged prompts:
(1) orderings per segment set: how many different orders the same components
arrive in;
(2) first-segment cardinality: distinct first segments over requests;
(3) replay: re-run the trace with no budget, and count the missed tokens whose
exact path had been cached before and was evicted.

**ANSWER: all three causes produce a near-zero hit rate, and each diagnostic
separates its own cause.** Scrambled ordering gives 8.4%: 6 orderings of one
segment set. The timestamp prefix gives 0.0%: 80 first segments in 80
requests, and a replay ceiling of 0.0% as well, so no budget would help. The 156-block budget gives 0.0%: a replay ceiling of 96.0% and 96.0%
of tokens lost to eviction. On the healthy trace the three read 1 ordering, 1
first segment and 26.8 points of eviction loss. Each fix recovers the rate:
fixed order 69.1%, the timestamp moved to just before the question 66.7%,
157 blocks 69.1%.

**FINDING: the lesson's own 8% trace fails two diagnostics, and the budget is
the bigger failure.** For the scrambled trace the replay ceiling is 79.5%, and
71.0% of its tokens are misses on a path that had been cached and then evicted.
Ordering explains only the gap from 96.0% to 79.5%. The fixed-order trace scores
69.1% at the same budget; removing the budget with the order left scrambled
reaches 79.5%. Run all three diagnostics before accepting the first cause
that fits.

**FINDING: in this toy the budget cause is a cliff, not a slope.** At 156
blocks the RAG hit rate is 0.0%; at 157 it is 69.1%. 157 is SYSTEM plus one
document, 125 + 32 blocks, which is what lets SYSTEM survive the eviction
cascade.

Structure: `diagnose()` returns the three readings for one trace; `run()`
wraps the reference cache and records misses.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "06-sglang-radixattention"
TIGHT, UNBOUNDED = 156, 10**6


def run(ref, reqs, budget=160):
    """(hit rate, share of prompt tokens missed on a path that had been cached)."""
    cache, inserted, saved, evicted, total = ref.RadixCache(budget), set(), 0, 0, 0
    for r in reqs:
        reused = cache.walk(r.segments)
        acc = 0
        for i, seg in enumerate(r.segments):
            acc += ref.token_count(seg)
            path = tuple(r.segments[: i + 1])
            evicted += ref.token_count(seg) if acc > reused and path in inserted else 0
            inserted.add(path)
        cache.insert(r.segments)
        saved, total = saved + reused, total + acc
    return round(saved / total, 4), round(evicted / total, 4)


def diagnose(ref, reqs, budget=160):
    orders = {}
    for r in reqs:
        orders.setdefault(frozenset(r.segments[:-1]), set()).add(tuple(r.segments[:-1]))
    hit, evicted = run(ref, reqs, budget)
    return {
        "hit": hit, "orderings": max(len(v) for v in orders.values()),
        "first_segments": len({r.segments[0] for r in reqs}),
        "ceiling": run(ref, reqs, UNBOUNDED)[0], "evicted": evicted,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rag = ref.workload_rag()
    stamped = [ref.Request(r.rid, [f"TS_{r.rid}"] + r.segments) for r in rag]
    moved = [ref.Request(r.rid, r.segments[:-1] + [f"TS_{r.rid}", r.segments[-1]]) for r in rag]
    return {
        "healthy": diagnose(ref, rag), "scrambled": diagnose(ref, ref.workload_scrambled()),
        "dynamic": diagnose(ref, stamped), "tight": diagnose(ref, rag, TIGHT),
        "fixes": (run(ref, rag)[0], run(ref, moved)[0], run(ref, rag, TIGHT + 1)[0]),
    }


def verify(result):
    h, s, d, t = (result[k] for k in ("healthy", "scrambled", "dynamic", "tight"))
    return [
        practice.Check(
            "ANSWER: all three causes produce a near-zero hit rate, and each diagnostic "
            "separates its own cause",
            all([(s["hit"], d["hit"], t["hit"]) == (0.0844, 0.0, 0.0),
                 (s["orderings"], d["orderings"], t["orderings"], h["orderings"]) == (6, 1, 1, 1),
                 (d["first_segments"], t["first_segments"], h["first_segments"]) == (80, 1, 1),
                 (t["ceiling"], t["evicted"]) == (0.9602, 0.9602), h["evicted"] == 0.2697,
                 result["fixes"] == (0.6906, 0.6672, 0.6906)]),
            f"scrambled {s}; dynamic {d}; tight {t}; healthy {h}; "
            f"fixes (order, move timestamp, +1 block) {result['fixes']}",
        ),
        practice.Check(
            "FINDING: the lesson's own 8% trace fails two diagnostics, and the budget is "
            "the bigger failure",
            s["ceiling"] == 0.7946 and s["evicted"] == 0.7102 and s["orderings"] > 1,
            f"scrambled: replay ceiling {s['ceiling']:.1%}, {s['evicted']:.1%} of tokens "
            f"missed after eviction, {s['orderings']} orderings",
        ),
        practice.Check(
            "FINDING: in this toy the budget cause is a cliff, not a slope",
            t["hit"] == 0.0 and result["fixes"][2] == h["hit"] == 0.6906,
            f"{TIGHT} blocks: {t['hit']:.1%}; {TIGHT + 1} blocks: {result['fixes'][2]:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
