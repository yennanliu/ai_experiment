"""Exercise 5 — let the planner call write the cache, and the fan-out reads it 5.5x cheaper at zero added latency.

    You batch 10 parallel sub-queries per user question. Rewrite for
    cache-friendliness without adding end-to-end latency.

Reading of the exercise: one user question is a planner call that produces
the 10 sub-queries, then the fan-out. Every call shares a cacheable prefix:
a 4000-token system prompt plus 1000 tokens of question context; the planner
adds 200 tokens of its own instructions and each sub-query 100. A cache entry
becomes visible 300 ms after the request that writes it starts (the lesson's
figure). Planner and sub-query latencies are fixed virtual numbers, 1.5 s and
1.0 s, so latency is computed, never timed. Prices are the reference's
constants.

**ANSWER: put the cache breakpoint at the end of the shared prefix in the
planner call, and fan out after it returns.** The fan-out already has to wait
for the planner, so the planner is the "sequential first" request for free:
it writes the prefix, and all 10 sub-queries read it.

| strategy | input bill | writes / reads | end-to-end |
|---|---:|---:|---:|
| no caching | $0.1686 | 0 / 0 | 2.5 s |
| naive parallel | $0.2061 | 10 / 0 | 2.5 s |
| sequential-first (lesson) | $0.0509 | 1 / 9 | 2.8 s |
| planner-primed | $0.0374 | 1 / 10 | 2.5 s |

Planner-primed is 5.5x cheaper than naive and 27% cheaper than the lesson's
fix, with no added latency; the lesson's fix costs 300 ms.

**FINDING: naive parallel caching costs 22% more than no caching.** Ten
simultaneous misses each pay the 1.25x write: $0.2061 against $0.1686.

**FINDING: with no planner, keep only the system prompt warm.** If the
sub-queries are templated client-side there is no free predecessor. Move the
question context after the breakpoint and keep the 4000-token system prompt
warm with traffic or a keepalive under 5 minutes: 10 reads at $0.0450,
4.2x cheaper than naive's $0.1905, with no added latency.

**FINDING: the reference's "serialize first" costs no latency.**
`parallel_penalty=False` just treats wave requests like single ones; the
simulator has no clock for the 300 ms. Anthropic's docs (fetched 2026-09-26):
"a cache entry only becomes available after the first response begins" --
the wait is time to first token, which the planner's reply already exceeds.

Structure: `run()` replays one question's calls against a cache whose entries
become visible 300 ms after the writing request starts.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "14-prompt-semantic-caching"
SYSTEM, CONTEXT, PLAN_TAIL, SUB_TAIL, N = 4000, 1000, 200, 100, 10
VISIBLE, PLANNER_S, SUB_S = 0.3, 1.5, 1.0


def run(ref, calls):
    """calls: (start_s, cached_key or None, prefix_tokens, tail_tokens, latency_s)."""
    visible, cost, writes, reads, end = {}, 0.0, 0, 0, 0.0
    for start, key, prefix, tail, latency in sorted(calls, key=lambda c: c[0]):
        if key is None:
            rate = ref.BASE_INPUT
        elif visible.get(key, float("inf")) <= start:
            rate, reads = ref.CACHED_INPUT, reads + 1
        else:
            rate, writes = ref.CACHE_WRITE_5MIN, writes + 1
            visible[key] = min(visible.get(key, float("inf")), start + VISIBLE)
        cost += (prefix * rate + tail * ref.BASE_INPUT) / 1e6
        end = max(end, start + latency)
    return {"cost": round(cost, 4), "writes": writes, "reads": reads, "e2e": round(end, 2)}


def question(planner_key, sub_key, stagger=0.0):
    shared = SYSTEM + CONTEXT
    calls = [(0.0, planner_key, shared, PLAN_TAIL, PLANNER_S)]
    for i in range(N):
        start = PLANNER_S + (stagger if i else 0.0)
        calls.append((start, sub_key, shared, SUB_TAIL, SUB_S))
    return calls


def no_planner(key, warm):
    calls = [(-60.0, key, SYSTEM, 0, 0.0)] if warm else []
    tail = SUB_TAIL + (CONTEXT if warm else 0)
    prefix = SYSTEM + (0 if warm else CONTEXT)
    return calls + [(0.0, key, prefix, tail, SUB_S) for _ in range(N)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    warm = run(ref, no_planner("S", warm=True))
    warm["cost"] = round(warm["cost"] - SYSTEM * ref.CACHE_WRITE_5MIN / 1e6, 4)
    return {
        "none": run(ref, question(None, None)),
        "naive": run(ref, question(None, "SQ")),
        "seq_first": run(ref, question(None, "SQ", stagger=VISIBLE)),
        "primed": run(ref, question("SQ", "SQ")),
        "cold_no_planner": run(ref, no_planner("SQ", warm=False)), "warm_no_planner": warm,
        "sim_has_clock": "arrived_at" in inspect.getsource(ref.simulate),
    }


def verify(r):
    none, naive, seq, primed = r["none"], r["naive"], r["seq_first"], r["primed"]
    cold, warm = r["cold_no_planner"], r["warm_no_planner"]
    return [
        practice.Check(
            "ANSWER: put the breakpoint in the planner call and fan out after it returns",
            all([(primed["writes"], primed["reads"]) == (1, 10), primed["e2e"] == naive["e2e"],
                 seq["e2e"] == naive["e2e"] + VISIBLE, primed["cost"] < seq["cost"],
                 round(naive["cost"] / primed["cost"], 1) == 5.5]),
            f"naive {naive}, sequential-first {seq}, planner-primed {primed}: "
            f"{naive['cost'] / primed['cost']:.1f}x cheaper than naive, "
            f"{1 - primed['cost'] / seq['cost']:.0%} under sequential-first, same latency",
        ),
        practice.Check(
            "FINDING: naive parallel caching costs 22% more than no caching",
            naive["writes"] == N and round(naive["cost"] / none["cost"], 2) == 1.22,
            f"{naive['writes']} writes, ${naive['cost']} against ${none['cost']} uncached",
        ),
        practice.Check(
            "FINDING: with no planner, keep only the system prompt warm",
            warm["reads"] == N and round(cold["cost"] / warm["cost"], 1) == 4.2
            and warm["e2e"] == cold["e2e"],
            f"cold fan-out {cold}; system prompt kept warm {warm}, the keepalive write "
            "excluded as a shared standing cost",
        ),
        practice.Check(
            "FINDING: the reference's 'serialize first' costs no latency",
            not r["sim_has_clock"],
            "simulate() never reads arrived_at, so parallel_penalty=False fixes the "
            "penalty at zero latency; the docs make the wait time-to-first-token",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
