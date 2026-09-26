"""Exercise 2 — evictions rise under every cause, so only prefix spread and hash churn tell them apart.

    Your cache hit rate drops from 70% to 12%. Diagnose three possible causes
    and the observables that would confirm each.

Reading of the exercise: each cause is reproduced on the reference
simulator, not just named. It then gets four observables a real fleet can
log: replicas per prefix, distinct prefix hashes per request, the share of
hashes seen only once, and evictions per request. The healthy baseline is the
reference GLOBAL router with 10 cache slots per replica, which hits 70.7%,
the exercise's 70%. Caches are exercise 1's FIFO, so runs repeat.

**ANSWER: the router went blind, the KV cache shrank, or the prefix stopped
being stable. Each has its own signature.**

- *Router lost cache state.* The KV-event feed stalls or the router restarts
  empty, and traffic falls back to round-robin. Hit rate goes to 23.8%.
  Replicas serving each prefix go from 2.8 to 10.4, and the hash mix stays
  the same.
- *KV capacity cut.* A longer `--max-model-len`, a lower
  `--gpu-memory-utilization` or a bigger model means fewer blocks. At 10 -> 2
  slots the hit rate is 14.5%. Evictions per request go 0.26 -> 0.85, and
  prefix spread and hash mix do not move.
- *Prefix instability.* A timestamp, user ID or request ID lands at the front
  of the prompt. When 55% of requests carry one, the hit rate is 12.4%.
  Distinct hashes per request go 0.04 -> 0.60, and 93% of hashes are seen
  once.

Evictions rise under all three, so they confirm that something is wrong but
not which cause it is.

**FINDING: routing alone cannot take this fleet to 12%.** Blind routing
floors at cache slots over working set, 10/40, and measures 23.8%. A drop
to 12% needs a cache-side cause as well: round-robin with the cache halved to
5 slots reaches 13.1%.

**FINDING: the reference tie-breaker is dead.** `queue_depth` goes up by 1
with probability 0.4 and down by 1 every request, clamped at 0, so it is 0
for all 1000 requests. `min(local, key=queue_depth)` therefore always returns
replica 0, and only 3 of the 12 replicas ever cache a prefix.

**FINDING: with a working local tie-breaker, cross-region buys 0.09 ms at
2K tokens.** Route each prefix to one fixed local replica instead, here by
prefix index mod 4, at the shipped 12 slots. REGIONAL then hits 88.0% at a mean of 166.4 ms
with no cross-region traffic, which beats the shipped GLOBAL's 233.6 ms.
GLOBAL on the same affinity gets 166.3 ms by making 648 cross-region calls.

**FINDING: the shipped hit rates are not reproducible.** Eviction is
`set.pop()`, whose order follows string hashes. Across `PYTHONHASHSEED` 0-5,
`main.py` prints GLOBAL at 86.2-89.2%. The check reproduces that spread in
process by relabelling the 40 prefixes as integers under 6 seeded
permutations, which gives 85.3-88.8%.

Structure: the workload variants are small functions, and `observe()` turns
one `fleet()` run into the four observables. `affinity()` runs k independent
one-replica fleets, one per prefix class.
"""

from __future__ import annotations

import collections
import pathlib
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "11-multi-region-kv-locality"
EX01 = practice.load_module(next(pathlib.Path(__file__).resolve().parent.glob("ex01_*.py")))
SLOTS, SHRUNK, UNSTABLE = 10, 2, 0.55


def unstable(base, frac, seed=3):
    """A fraction of requests carry a per-request token ahead of the shared prefix."""
    rng = random.Random(seed)
    return [type(r)(r.origin_region, f"{r.prefix_hash}#{i}" if rng.random() < frac
                    else r.prefix_hash) for i, r in enumerate(base)]


def observe(ref, strategy, reqs, cap=SLOTS):
    stats, replicas, served = EX01.fleet(ref, strategy, reqs, cap)
    seen = collections.Counter(q.prefix_hash for q in served)
    placed = {(q.prefix_hash, q.served_by.region, q.served_by.idx) for q in served}
    return {
        "hit": round(stats["hit_rate"], 3),
        "spread": round(len(placed) / len(seen), 1),
        "distinct": round(len(seen) / len(served), 2),
        "once": round(sum(v == 1 for v in seen.values()) / len(seen), 2),
        "evict": round(sum(r.prefix_cache.evictions for r in replicas) / len(served), 2),
        "caching": sum(bool(r.prefix_cache) for r in replicas),
        "depths": {q.served_by.queue_depth for q in served},
    }


def relabelled(ref, seed):
    """Shipped set.pop() eviction, prefixes relabelled as ints (deterministic hashes)."""
    perm = list(range(40))
    random.Random(seed).shuffle(perm)
    reqs = [ref.Request(r.origin_region, perm[int(r.prefix_hash.split("_")[1])])
            for r in ref.make_workload()]
    return round(ref.simulate("GLOBAL", reqs)["hit_rate"], 3)


def affinity(ref, strategy, k=4):
    """Prefix i always goes to local replica i mod k: k independent one-replica fleets."""
    base, ttft, hits, cross = ref.make_workload(), 0.0, 0.0, 0
    for part in range(k):
        sub = [r for r in base if int(r.prefix_hash.split("_")[1]) % k == part]
        stats = EX01.fleet(ref, strategy, sub, per_region=1)[0]
        ttft, hits = ttft + stats["mean_ttft"] * len(sub), hits + stats["hit_rate"] * len(sub)
        cross += stats["crossregion"]
    return round(hits / len(base), 3), round(ttft / len(base), 2), cross


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = ref.make_workload()
    return {
        "base": observe(ref, "GLOBAL", base),
        "blind": observe(ref, "ROUND_ROBIN", base),
        "shrunk": observe(ref, "GLOBAL", base, SHRUNK),
        "unstable": observe(ref, "GLOBAL", unstable(base, UNSTABLE)),
        "blind_half": observe(ref, "ROUND_ROBIN", base, SLOTS // 2)["hit"],
        "regional": observe(ref, "REGIONAL", base, 12),
        "relabelled": [relabelled(ref, seed) for seed in range(6)],
        "shipped_global": EX01.fleet(ref, "GLOBAL", base)[0]["mean_ttft"],
        "aff": (affinity(ref, "REGIONAL"), affinity(ref, "GLOBAL")),
    }


def verify(result):
    base, blind, shrunk, unst = (result[k] for k in ("base", "blind", "shrunk", "unstable"))
    causes = (blind, shrunk, unst)
    reg_run, labels, (reg, glob) = result["regional"], result["relabelled"], result["aff"]
    return [
        practice.Check(
            "ANSWER: router blind, KV capacity cut, or prefix unstable -- each has a signature",
            all([base["hit"] == 0.707, blind["hit"] == 0.238, shrunk["hit"] == 0.145,
                 unst["hit"] == 0.124, blind["spread"] > 3 * base["spread"],
                 blind["distinct"] == shrunk["distinct"] == base["distinct"],
                 abs(shrunk["spread"] - base["spread"]) < 0.5, unst["once"] > 0.9,
                 all(c["evict"] > 2 * base["evict"] for c in causes)]),
            f"baseline {base}; blind {blind}; capacity {SLOTS}->{SHRUNK} {shrunk}; "
            f"{UNSTABLE:.0%} unstable prefixes {unst}",
        ),
        practice.Check(
            "FINDING: routing alone cannot take this fleet to 12%",
            blind["hit"] > 0.2 and result["blind_half"] == 0.131,
            f"round-robin floors near {SLOTS}/40 slots per working set at {blind['hit']:.1%}; "
            f"round-robin at {SLOTS // 2} slots reaches {result['blind_half']:.1%}",
        ),
        practice.Check(
            "FINDING: the reference tie-breaker is dead",
            reg_run["depths"] == {0} and reg_run["caching"] == 3 and base["caching"] == 3,
            f"queue_depth values seen {reg_run['depths']}; {reg_run['caching']} of 12 replicas ever "
            f"cache a prefix under REGIONAL and {base['caching']} under GLOBAL",
        ),
        practice.Check(
            "FINDING: with a working local tie-breaker, cross-region buys 0.09 ms at 2K tokens",
            all([reg[2] == 0, reg[1] < result["shipped_global"], glob[2] > 600,
                 round(reg[1] - glob[1], 2) == 0.09]),
            f"prefix-affinity REGIONAL {reg[0]:.1%} at {reg[1]} ms, 0 cross-region; shipped "
            f"GLOBAL {result['shipped_global']:.1f} ms; affinity GLOBAL {glob[1]} ms at "
            f"{glob[2]} cross-region calls",
        ),
        practice.Check(
            "FINDING: the shipped hit rates are not reproducible",
            len(set(labels)) == 6 and max(labels) - min(labels) > 0.03,
            f"GLOBAL hit rate under 6 prefix relabellings with set.pop() eviction: {labels}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
