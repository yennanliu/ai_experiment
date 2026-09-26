"""Exercise 2 — the tenant loses time to LMCache at the lesson's rates, and saves only below 0.4 ms per block.

    A tenant shares a 6K-token system prompt across 200 queries/hour. Compute
    expected LMCache savings per tenant.

Reading of the exercise: "savings" is prefill time saved per hour, priced
with the lesson's own constants (40 prefill tokens/ms, 16-token blocks,
3.0 ms per LMCache block load). Savings are counted against engine-local
prefix caching, which is what a cluster has without LMCache, and against no
cache at all as the ceiling. The hour is simulated on exercise 1's `run()`: the
tenant's 200 queries land on 4 engines of 4 prefix slots each, with m of the
lesson's own template requests between consecutive tenant queries. m is how
much other traffic evicts the tenant's prefix.

**ANSWER: negative -- LMCache costs the tenant 2.9 to 141 seconds of prefill
an hour.** One 6K prefill is 150 ms; loading the same 375 blocks from LMCache
is 1125 ms, so each LMCache hit costs 975 ms more than recomputing. With no
other traffic (m = 0) the tenant makes 3 LMCache hits an hour, -2.9 s. At
m = 16 it makes 145, -141.4 s. At 0.4 ms/block every m saves exactly 0. At
0.2 ms/block LMCache saves 0.2 s (m = 0) to 10.9 s (m = 16) an hour.

**FINDING: engine-local prefix caching already takes almost all of the
ceiling.** Against no cache the most any cache can save is
199 x 150 ms = 29.85 s an hour. With m = 0 native caching misses only once
per engine (4 prefills, 600 ms) and saves 29.4 s of that. What is left for
LMCache is the eviction traffic: 3.9 s at m = 1, 21.9 s at m = 16.

**FINDING: the lesson's "<1K tokens: transfer time > re-prefill" has no
threshold in this model.** Both costs are linear in tokens, so a load costs
7.5x the prefill at 6K and 8K tokens. Block rounding puts it at 7.56x at 1K
and 7.68x at 500. No prompt length makes a transfer cheaper, because there is
no fixed per-transfer cost to amortize.

Structure: `hour()` builds the interleaved workload and prices the tenant's
own requests from `run()`'s per-request costs.
"""

from __future__ import annotations

import pathlib
import random

from harness import parity, practice

HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
PROMPT, QUERIES, SLOTS = 6000, 200, 4
MIXES, LOADS = (0, 1, 2, 4, 8, 16), (3.0, 0.4, 0.2)


def workload(ref, m, seed=3):
    rng, other = random.Random(seed), ref.make_workload(QUERIES * m) if m else []
    reqs, mine = [], []
    for q in range(QUERIES):
        mine.append(len(reqs))
        reqs.append(ref.Request(PROMPT, rng.randint(150, 400), "tenant"))
        reqs += other[q * m:(q + 1) * m]
    return reqs, mine


def hour(ref, m, load_ms):
    """(tenant prefill ms native, tenant prefill ms with LMCache, tenant LMCache hits)."""
    reqs, mine = workload(ref, m)
    native = EX01.run(ref, reqs, "NATIVE_ONLY", SLOTS)["costs"]
    lm = EX01.run(ref, reqs, "LMCACHE", SLOTS, load_ms)
    hits = len(set(lm["hits"]) & set(mine))
    return sum(native[i] for i in mine), sum(lm["costs"][i] for i in mine), hits


def ratio(ref, tokens):
    blocks = -(-tokens // ref.KV_BLOCK_TOKENS)
    return round(blocks * ref.LMCACHE_TIME_MS_PER_BLOCK / (tokens / ref.PREFILL_TOK_PER_MS), 2)


def solve():
    ref = parity.load_reference(EX01.PHASE, EX01.LESSON, "main")
    grid = {(m, c): hour(ref, m, c) for m in MIXES for c in LOADS}
    return {
        "prefill": PROMPT / ref.PREFILL_TOK_PER_MS,
        "load": -(-PROMPT // ref.KV_BLOCK_TOKENS) * ref.LMCACHE_TIME_MS_PER_BLOCK,
        "saved": {k: round((n - lm) / 1000, 1) for k, (n, lm, _) in grid.items()},
        "hits": {m: grid[m, 3.0][2] for m in MIXES},
        "native": {m: grid[m, 3.0][0] for m in MIXES},
        "ratios": {t: ratio(ref, t) for t in (500, 1000, 6000, 8000)},
    }


def verify(result):
    saved, hits, native = result["saved"], result["hits"], result["native"]
    ceiling = (QUERIES - 1) * result["prefill"]
    return [
        practice.Check(
            "ANSWER: negative -- LMCache costs the tenant 2.9 to 141 seconds of prefill an hour",
            all([result["prefill"] == 150, result["load"] == 1125,
                 saved[0, 3.0] == -2.9, saved[16, 3.0] == -141.4,
                 all(saved[m, 0.4] == 0 for m in MIXES),
                 (saved[0, 0.2], saved[16, 0.2]) == (0.2, 10.9)]),
            f"a hit loads for {result['load']:.0f} ms against a {result['prefill']:.0f} ms "
            f"prefill; tenant LMCache hits by interleave m {hits}; seconds saved per hour at "
            f"3.0 ms/block {[saved[m, 3.0] for m in MIXES]}, at 0.2 "
            f"{[saved[m, 0.2] for m in MIXES]}",
        ),
        practice.Check(
            "FINDING: engine-local prefix caching already takes almost all of the ceiling",
            ceiling == 29850 and native[0] == 600 and (native[1], native[16]) == (3900, 21900),
            f"ceiling {ceiling / 1000} s/hour; native prefill left for LMCache to save, by m: "
            f"{ {m: v / 1000 for m, v in native.items()} } s",
        ),
        practice.Check(
            "FINDING: '<1K tokens: transfer > re-prefill' has no threshold in this model",
            result["ratios"] == {500: 7.68, 1000: 7.56, 6000: 7.5, 8000: 7.5},
            f"load / prefill cost by prompt tokens {result['ratios']}: both linear, no fixed "
            "per-transfer cost",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
