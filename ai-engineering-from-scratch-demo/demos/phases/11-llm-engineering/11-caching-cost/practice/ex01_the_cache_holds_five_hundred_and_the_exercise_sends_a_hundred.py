"""Exercise 1 — the cache holds 500, the exercise sends 100, and neither policy evicts.

    **Implement LRU eviction for the semantic cache.** Replace the
    oldest-first eviction with least-recently-used. Track the last access time
    for each entry and evict the entry with the oldest access time when the
    cache is full. Compare hit rates between the two strategies over 100
    queries.

Reading of the exercise: both policies are run over the same 100-query stream,
built from a 30-query catalogue with a Zipf-like repeat pattern so that recency
is worth something, and the LRU variant is the lesson's `SemanticCache` with two
lines changed.

**ANSWER: over 100 queries neither policy evicts anything.** `SemanticCache`
defaults to `max_size=500`, the stream puts 22 distinct queries in it,
and the two strategies return identical hit rates because the branch that
distinguishes them never runs.

**FINDING: `get` does not touch the timestamp, so an LRU built on the existing
field would still be FIFO.** The entry dict carries `timestamp`, set once in
`put`, and `access_count`, incremented on every hit and read by nothing. The
exercise's "track the last access time" is a field that has to be added, not a
field that exists.

**ANSWER: shrink the cache until it binds and LRU wins by 5 points.** At
`max_size=10` over the same stream, FIFO hits 0.580 with 32 evictions and LRU
hits 0.630 with 27. That is the comparison the exercise wants, and it needs a
cache size the exercise does not set.

**FINDING: the similarity threshold is worth exactly as much as the policy.**
At 0.85, a four-word query differing by one word scores 0.75 and misses.
Lowering the threshold to 0.7 takes FIFO from 0.580 to 0.630 -- a gain of
0.050, the same as switching to LRU. Two levers of equal size, and the exercise
varies one of them.

**FINDING: `access_count` starts at 1 in `put`.** A never-read entry and an
entry read once are indistinguishable, so any policy built on the counter the
lesson already tracks cannot tell a cold entry from a warm one.

Structure: `lru_put` and `lru_get` are the two changed lines, `stream` builds
the query sequence, and `run` measures one policy at one size.
"""

from __future__ import annotations

import itertools
import random
import time

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "11-caching-cost"
TOPICS = ("refund window", "api rate limit", "uptime guarantee", "data retention",
          "encryption at rest", "billing cycle", "team seats", "audit logs",
          "sso setup", "webhook retries", "export format", "trial length",
          "support hours", "region choice", "backup schedule")
CATALOGUE = [f"{verb} {topic}" for topic in TOPICS for verb in ("explain", "describe")]
STREAM_LENGTH, SEED = 100, 7


def stream(n=STREAM_LENGTH, seed=SEED):
    """A Zipf-like repeat pattern: early queries recur, later ones are one-offs."""
    generator = random.Random(seed)
    weights = [1 / (i + 1) for i in range(len(CATALOGUE))]
    return generator.choices(CATALOGUE, weights=weights, k=n)


def lru_get(cache, query):
    """The lesson's `get`, plus the one line that records the access."""
    hit = cache.get(query)
    if hit:
        for entry in cache.entries:
            if entry["query"] == hit["original_query"]:
                entry["last_access"] = next(CLOCK)
    return hit


def lru_put(cache, query, response):
    """The lesson's `put`, evicting by last access rather than by insertion."""
    if len(cache.entries) >= cache.max_size:
        cache.entries.sort(key=lambda e: e.get("last_access", 0))
        cache.entries.pop(0)
    cache.entries.append({"query": query, "embedding": EMBED(query), "response": response,
                          "timestamp": time.time(), "access_count": 1,
                          "last_access": next(CLOCK)})


CLOCK = itertools.count(1)
EMBED = None


def run(ref, policy, size, threshold=0.85):
    cache = ref.SemanticCache(similarity_threshold=threshold, max_size=size)
    evictions = 0
    for query in stream():
        hit = lru_get(cache, query) if policy == "lru" else cache.get(query)
        if hit:
            continue
        before = len(cache.entries)
        if policy == "lru":
            lru_put(cache, query, "response")
        else:
            cache.put(query, "response")
        evictions += before == size
    stats = cache.stats()
    return {"hit_rate": stats["hit_rate"], "evictions": evictions,
            "size": len(cache.entries), "max_size": size}


def solve():
    global EMBED
    ref = parity.load_reference(PHASE, LESSON, "caching_cost")
    EMBED = ref.simple_embed
    default = ref.SemanticCache().max_size
    big = {policy: run(ref, policy, default) for policy in ("fifo", "lru")}
    small = {policy: run(ref, policy, 10) for policy in ("fifo", "lru")}
    loose = run(ref, "fifo", 10, threshold=0.7)
    probe = ref.SemanticCache(max_size=2)
    probe.put("q", "r")
    return {
        "default_size": default, "distinct": len(set(stream())),
        "stream": STREAM_LENGTH, "big": big, "small": small,
        "loose_hit_rate": loose["hit_rate"],
        "threshold_gain": round(loose["hit_rate"] - small["fifo"]["hit_rate"], 3),
        "policy_gain": round(small["lru"]["hit_rate"] - small["fifo"]["hit_rate"], 3),
        "entry_fields": sorted(probe.entries[0]),
        "initial_access_count": probe.entries[0]["access_count"],
        "near_miss": round(ref.cosine_similarity(ref.simple_embed("what is the price"),
                                                 ref.simple_embed("what is the address")), 3),
    }


def verify(result):
    big, small = result["big"], result["small"]
    return [
        practice.Check(
            "ANSWER: over 100 queries neither policy evicts anything",
            all([big["fifo"]["evictions"] == 0, big["lru"]["evictions"] == 0,
                 big["fifo"]["hit_rate"] == big["lru"]["hit_rate"],
                 result["distinct"] < result["default_size"]]),
            f"`SemanticCache` defaults to max_size={result['default_size']} and the "
            f"{result['stream']}-query stream puts {result['distinct']} distinct queries in "
            f"it, so both policies evict {big['fifo']['evictions']} entries and return the "
            f"same hit rate, {big['fifo']['hit_rate']}. The branch that distinguishes them "
            "never runs",
        ),
        practice.Check(
            "FINDING: `get` never touches the timestamp, so the existing field is FIFO",
            all([result["entry_fields"] == ["access_count", "embedding", "query",
                                            "response", "timestamp"],
                 result["initial_access_count"] == 1]),
            f"an entry carries {result['entry_fields']}: `timestamp` is set once in `put` "
            f"and `access_count` is incremented on every hit and read by nothing. 'Track "
            "the last access time' is a field that has to be added, not one that exists",
        ),
        practice.Check(
            "ANSWER: shrink the cache until it binds and LRU wins by 5 points",
            all([small["lru"]["hit_rate"] > small["fifo"]["hit_rate"],
                 small["fifo"]["evictions"] > 0, result["policy_gain"] >= 0.05]),
            f"at max_size=10 over the same stream, FIFO hits "
            f"{small['fifo']['hit_rate']} with {small['fifo']['evictions']} evictions and "
            f"LRU hits {small['lru']['hit_rate']} with {small['lru']['evictions']}, a gain "
            f"of {result['policy_gain']}. That is the comparison the exercise wants, at a "
            "cache size the exercise does not set",
        ),
        practice.Check(
            "FINDING: the threshold is worth exactly as much as the policy",
            result["threshold_gain"] == result["policy_gain"],
            f"at the default threshold of 0.85 a four-word query differing by one word "
            f"scores {result['near_miss']} and misses, so only close repeats hit. Lowering "
            f"it to 0.7 takes FIFO from {small['fifo']['hit_rate']} to "
            f"{result['loose_hit_rate']}, a gain of {result['threshold_gain']} -- the same "
            f"as switching to LRU. Two levers of equal size, and the exercise varies one",
        ),
        practice.Check(
            "FINDING: access_count starts at 1, so cold and warm entries look alike",
            result["initial_access_count"] == 1,
            f"`put` sets access_count to {result['initial_access_count']} rather than 0, so "
            "a never-read entry and an entry read once are indistinguishable. Any policy "
            "built on the counter the lesson already tracks cannot tell them apart",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
