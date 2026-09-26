"""Exercise 1 — cache-aware serves the RAG queue in FCFS order, so there is no delta.

    Run `code/main.py`. Compare FCFS and cache-aware on the same workload.
    Where does the delta come from — prefill savings, decode savings, or queue
    delay?

Reading of the exercise: "the delta" is taken at face value -- first measured
on the workload the lesson compares, then traced to its source. The toy counts
only prompt tokens served from cache, so decode and queue delay are read off a
serial-server model built on the recorded serving order: each request costs
its uncached prompt tokens, and its completion time is the running sum.

**ANSWER: on the RAG workload there is no delta -- both schedulers score
69.1%, because cache-aware serves the requests in exactly FCFS order.** Its
score is max over prefixes of (requests sharing it) x (prefix tokens), and the
SYSTEM+TOOLS prefix every request shares wins for all 80: 80 x 2300 = 184000,
a tie, and a stable sort of ties is the arrival order. The printed "hit rate
clears 80% on RAG" is not what the run prints.

**FINDING: where a delta exists it is all prefill.** On the scrambled
workload cache-aware lifts hit rate 8.4% -> 26.5%, 41400 more prompt tokens
served from cache. Decode savings are zero by construction: a `Request` has no
output tokens, and a reused prefix changes no decode step. Queue delay is a
consequence, not a source: on the serial model mean completion time falls
104525 -> 70181 token-units because there is less prefill to wait behind,
while the reorder pushes one request back 71 places.

**FINDING: the budget, not the scheduler, caps the RAG hit rate.** One RAG
path is 125 + 19 + 32 + 4 = 180 blocks against a 160-block budget, so TOOLS
and a document never fit beside SYSTEM and every request reuses SYSTEM alone
(79 x 2000 = 158000 tokens). Serving grouped by document -- a real
depth-first order -- gives 96.0% from 180 blocks, where FCFS and cache-aware
both give 82.5%; they stay equal at every budget.

Structure: `trace()` swaps in a recording RadixCache (restored after) so the
served order and per-request reuse come from the reference's own `simulate`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "06-sglang-radixattention"
BUDGETS = (160, 180, 250, 400)


def trace(ref, reqs, scheduler, budget=160):
    """[(question, reused, prompt tokens)] in served order, from ref.simulate."""
    original, served = ref.RadixCache, []

    class Recording(original):
        def walk(self, segments):
            reused = super().walk(segments)
            total = sum(ref.token_count(s) for s in segments)
            served.append((segments[-1], reused, total))
            return reused

    ref.RadixCache = lambda: Recording(budget)
    try:
        return ref.simulate(reqs, scheduler), served
    finally:
        ref.RadixCache = original


def mean_completion(served):
    clock, done = 0, []
    for _, reused, total in served:
        clock += total - reused
        done.append(clock)
    return round(sum(done) / len(done))


def scores(ref, reqs):
    """The reference's cache-aware score for every request, recomputed."""
    count = {}
    for r in reqs:
        for i in range(1, len(r.segments) + 1):
            count[tuple(r.segments[:i])] = count.get(tuple(r.segments[:i]), 0) + 1
    return {max(count[tuple(r.segments[:i])] * sum(map(ref.token_count, r.segments[:i]))
                for i in range(1, len(r.segments) + 1)) for r in reqs}


def max_pushback(fcfs, aware):
    arrival = {q: i for i, (q, _, _) in enumerate(fcfs)}
    return max(i - arrival[q] for i, (q, _, _) in enumerate(aware))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rag, scr = ref.workload_rag(), ref.workload_scrambled()
    by_doc = sorted(rag, key=lambda r: r.segments[2])
    runs = {k: trace(ref, w, s) for k, w, s in [
        ("rag_f", rag, "FCFS"), ("rag_c", rag, "CACHE_AWARE"),
        ("scr_f", scr, "FCFS"), ("scr_c", scr, "CACHE_AWARE")]}
    hit = {k: round(r[0]["hit_rate"], 4) for k, r in runs.items()}
    sweep = {b: tuple(round(trace(ref, w, s, b)[0]["hit_rate"], 4) for w, s in
                      [(rag, "FCFS"), (rag, "CACHE_AWARE"), (by_doc, "FCFS")])
             for b in BUDGETS}
    reused = {r for _, r, _ in runs["rag_f"][1]}
    path = [-(-ref.token_count(s) // ref.BLOCK_TOKENS) for s in rag[0].segments]
    return {
        "hit": hit, "scores": scores(ref, rag), "same_order": runs["rag_f"][1] == runs["rag_c"][1],
        "gain": runs["scr_c"][0]["saved"] - runs["scr_f"][0]["saved"],
        "completion": (mean_completion(runs["scr_f"][1]), mean_completion(runs["scr_c"][1])),
        "pushback": max_pushback(runs["scr_f"][1], runs["scr_c"][1]),
        "decode_fields": sorted(vars(rag[0])), "sweep": sweep,
        "reused": reused, "path": path, "budget": ref.KV_BUDGET_BLOCKS,
    }


def verify(result):
    hit, sweep = result["hit"], result["sweep"]
    return [
        practice.Check(
            "ANSWER: on the RAG workload there is no delta; cache-aware serves in FCFS order",
            hit["rag_f"] == hit["rag_c"] == 0.6906 and result["same_order"]
            and result["scores"] == {80 * 2300},
            f"both {hit['rag_f']:.1%}; every request scores 80 x 2300 = 184000 and the "
            "stable sort keeps arrival order -- not the 80% the script prints",
        ),
        practice.Check(
            "FINDING: where a delta exists it is all prefill",
            all([(hit["scr_f"], hit["scr_c"]) == (0.0844, 0.2653), result["gain"] == 41400,
                 result["decode_fields"] == ["rid", "segments"],
                 result["completion"] == (104525, 70181), result["pushback"] == 71]),
            f"scrambled {hit['scr_f']:.1%} -> {hit['scr_c']:.1%}, {result['gain']} prompt "
            f"tokens; Request has only {result['decode_fields']}, no output tokens; mean "
            f"completion {result['completion'][0]} -> {result['completion'][1]} "
            f"token-units while one request is pushed back {result['pushback']} places",
        ),
        practice.Check(
            "FINDING: the budget, not the scheduler, caps the RAG hit rate",
            all([sum(result["path"]) == 180 > result["budget"], result["reused"] == {0, 2000},
                 all(f == c for f, c, _ in sweep.values()), sweep[180] == (0.8247, 0.8247, 0.9602)]),
            f"one path is {result['path']} = {sum(result['path'])} blocks over a "
            f"{result['budget']}-block budget, so only SYSTEM is reused; (FCFS, cache-aware, "
            f"grouped by doc) by budget {sweep}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
