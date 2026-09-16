"""Exercise 4 — LRU scores below doing nothing; LFU on the lesson's unused `hit_count` clears the bar.

    Build a prefix cache that uses LRU eviction. Set max_entries to 500 and
    generate 1,000 requests where 60% share one of 5 common prefixes. Measure hit
    rate and compare to unlimited cache. With good eviction, hit rate should stay
    above 55%.

Reading of the exercise: the eviction has to be written, because the lesson's
`PrefixCache` has none -- when `total_entries` reaches `max_entries` its `insert`
returns early and the trie freezes on whatever arrived first. Two policies are
built on the lesson's own trie: LRU, which the exercise names, and LFU on
`TrieNode.hit_count`, which the lesson already maintains on every lookup and
never reads. Both are compared against the unlimited cache, as asked.

**ANSWER: LRU scores 0.451, below the 0.484 of evicting nothing at all; LFU
scores 0.601 and clears the 55% bar.**

    lesson's cache, 500 entries     0.484     500 nodes, none ever evicted
    LRU, 500 entries                0.451     500 nodes
    LFU on hit_count, 500 entries   0.601     500 nodes
    lesson's cache, unlimited       0.623     35,774 nodes

LFU recovers **96%** of the unlimited cache's hit rate from **1.4%** of its
nodes. The policy the exercise names is the one that does not clear the bar the
exercise sets -- and it is beaten by the frozen cache it was meant to fix, which
at least keeps the prefixes it happened to see first.

**FINDING: the reference has no eviction, and fails silently.** `insert`
returns the index it stopped at when the budget is full, and nothing in the
lesson checks that return value. `total_entries` sits at exactly 500 forever and
a second 1,000-token prompt inserted into a full cache adds **0** nodes. That
this still scores 0.484 is the point: freezing the first few prompts is a
policy, and it beats recency here.

**MECHANISM: recency is the wrong key for this workload.** Every request appends
20 unique tokens, so 400 unique prompts churn 24,000 leaf insertions through a
500-node budget. Recency cannot tell a shared prefix from the tail of the request
that just used it; frequency can, and the five shared prefixes are 40 tokens
each -- **200 of the 500 nodes** -- which is exactly what LFU chooses to keep,
and worth **+0.150** over LRU.

**FINDING: "hit rate" counts a single shared token as a hit.** `lookup` records
a hit whenever `depth > 0`. At this vocabulary the two measures nearly agree
(0.601 against a 0.600 share of requests matching a full 40-token prefix), but
the metric is a lower bound on nothing: with a smaller vocabulary every first
token collides and the reported hit rate goes to 1.0 without any prefix being
reused.

Structure: `make_cache` subclasses the lesson's own `PrefixCache` and adds the
eviction it lacks, with `pick` choosing the victim; `workload` builds the 1,000
requests the exercise specifies.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "12-inference-optimization"
SEED, REQUESTS, ENTRIES = 11, 1000, 500
PREFIXES, PREFIX_LEN, TAIL, VOCAB, SHARE = 5, 40, 20, 5000, 0.6


def make_cache(base, node_type, pick, max_entries=ENTRIES):
    """The lesson's trie plus the eviction it has none of, choosing a victim with `pick`."""

    class Cache(base):
        def __init__(self):
            super().__init__(max_entries)
            self.clock, self.used, self.parent = 0, {}, {}

        def touch(self, node):
            self.clock += 1
            self.used[node] = self.clock

        def lookup(self, token_ids):
            depth, kv = super().lookup(token_ids)
            node = self.root
            for tid in token_ids[:depth]:
                node = node.children[tid]
                self.touch(node)
            return depth, kv

        def evict(self):
            leaves = [n for n in self.used if not n.children]
            victim = min(leaves, key=lambda n: pick(n, self.used[n]))
            parent, tid = self.parent.pop(victim)
            del parent.children[tid]
            del self.used[victim]
            self.total_entries -= 1

        def insert(self, token_ids, kv_per_token):
            node = self.root
            for tid in token_ids:
                if tid not in node.children:
                    while self.total_entries >= self.max_entries:
                        self.evict()
                    node.children[tid] = node_type()
                    self.parent[node.children[tid]] = (node, tid)
                    self.total_entries += 1
                node = node.children[tid]
                self.touch(node)
            return len(token_ids)

    return Cache()


def workload():
    """1,000 requests, 60% of them sharing one of 5 common 40-token prefixes."""
    rng = np.random.default_rng(SEED)
    common = [list(rng.integers(0, VOCAB, PREFIX_LEN)) for _ in range(PREFIXES)]
    out = []
    for _ in range(REQUESTS):
        tail = list(rng.integers(0, VOCAB, TAIL))
        head = (list(common[int(rng.integers(0, PREFIXES))]) if rng.random() < SHARE
                else list(rng.integers(0, VOCAB, PREFIX_LEN)))
        out.append(head + tail)
    return out


def run(cache, requests):
    """One pass of the workload, recording the lesson's hit rate and full-prefix reuse."""
    deep = 0
    for request in requests:
        deep += cache.lookup(request)[0] >= PREFIX_LEN
        cache.insert(request, [None] * len(request))
    return {"hit_rate": cache.hit_rate(), "full_prefix": deep / len(requests),
            "nodes": cache.total_entries}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    requests = workload()
    lru = make_cache(ref.PrefixCache, ref.TrieNode, lambda node, tick: (tick,))
    lfu = make_cache(ref.PrefixCache, ref.TrieNode, lambda node, tick: (node.hit_count, tick))
    full = ref.PrefixCache(max_entries=ENTRIES)
    frozen = (full.insert(list(range(REQUESTS)), [None] * REQUESTS),
              full.insert(list(range(REQUESTS, 2 * REQUESTS)), [None] * REQUESTS))
    return {
        "arms": {"lesson": run(ref.PrefixCache(max_entries=ENTRIES), requests),
                 "lru": run(lru, requests),
                 "lfu": run(lfu, requests),
                 "unlimited": run(ref.PrefixCache(max_entries=10 ** 9), requests)},
        "frozen": frozen + (full.total_entries,),
        "prefix_nodes": PREFIXES * PREFIX_LEN,
    }


def table(arms, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in arms.items())


def verify(result):
    arms = result["arms"]
    lru, lfu, unlimited, lesson = arms["lru"], arms["lfu"], arms["unlimited"], arms["lesson"]
    stopped, again, entries = result["frozen"]
    return [
        practice.Check(
            "ANSWER: LRU scores below evicting nothing; LFU on hit_count clears the 55% bar",
            lru["hit_rate"] < lesson["hit_rate"] < 0.55 < lfu["hit_rate"] < unlimited["hit_rate"],
            "hit rates are " + table(arms, "hit_rate", ".3f")
            + " at node counts of " + table(arms, "nodes", ",d")
            + f". LFU recovers {100 * lfu['hit_rate'] / unlimited['hit_rate']:.0f}% of the "
            f"unlimited cache's hit rate from "
            f"{100 * ENTRIES / unlimited['nodes']:.1f}% of its nodes, and the policy the exercise "
            "names is the one that does not clear the bar the exercise sets",
        ),
        practice.Check(
            "FINDING: the reference has no eviction, and fails silently when it fills",
            lesson["nodes"] == ENTRIES and again == 0,
            f"PrefixCache.insert returns the index it stopped at when the budget is full and "
            f"nothing reads that value: a {REQUESTS}-token prompt into an empty {ENTRIES}-entry "
            f"cache stops at {stopped}, a second adds {again} nodes, and total_entries sits at "
            f"{entries} forever. The cache freezes on whatever arrived first -- and still scores "
            f"{lesson['hit_rate']:.3f}, because that is a policy too",
        ),
        practice.Check(
            "MECHANISM: recency is the wrong key for a workload that churns its leaves",
            lfu["hit_rate"] > lru["hit_rate"] and result["prefix_nodes"] < ENTRIES,
            f"every request appends {TAIL} unique tokens, so the unique prompts churn tens of "
            f"thousands of leaf insertions through a {ENTRIES}-node budget, and recency cannot "
            f"tell a shared prefix from the tail of the request that just used it. Frequency can: "
            f"the {PREFIXES} shared prefixes are {result['prefix_nodes']} of the {ENTRIES} nodes, "
            f"and keeping exactly those is worth {lfu['hit_rate'] - lru['hit_rate']:+.3f}",
        ),
        practice.Check(
            "FINDING: the reported hit rate counts a single shared token as a hit",
            all(row["hit_rate"] >= row["full_prefix"] for row in arms.values()),
            "lookup records a hit whenever depth > 0. At this vocabulary the two nearly agree -- "
            + table(arms, "hit_rate", ".3f") + f" against a full-{PREFIX_LEN}-token-prefix share "
            "of " + table(arms, "full_prefix", ".3f")
            + f" -- but that is a property of a {VOCAB:,}-token vocabulary, not of the metric: "
            "with a small one every first token collides and the rate goes to 1.0 unreused",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
