"""Exercise 4 — block LRU strands a branch by evicting its first block, and leaf-first order is the fix.

    Read the SGLang RadixAttention paper. Explain in three sentences why
    tree-shaped LRU eviction beats block-shaped LRU under prefix-heavy load.

Reading of the exercise: the three sentences are in the README; this file
measures the claim they rest on. arXiv:2312.07104 (v2) specifies "an LRU
eviction policy that evicts the least recently used leaf first", so that
common ancestors stay reusable "until those ancestors become leaves". Both
policies run over the lesson's RAG workload at 16-token-block granularity, and
a hit is a block reused from the root down. Block LRU evicts the oldest block
anywhere; ties go to the earliest-inserted block, which is the front of a
branch. Tree LRU evicts the oldest block that has no cached child.

**ANSWER: block LRU evicts a cold branch's first block, and that strands the
rest of the branch.** Once its first block is gone, the branch's remaining
blocks are unreachable but still resident. At 200 blocks tree LRU hits 85.3%
and block LRU 82.1%, with up to 20 dead blocks. At 250 blocks the figures are
91.9% and 87.4%, with 34 dead. Below one request's 180 blocks, at the shipped
160, block LRU evicts SYSTEM's first block and hits 0.0%, with all 160 blocks
stranded at peak; tree LRU keeps 76.8%.

**FINDING: what matters is the eviction order, not the data structure.** Break
block LRU's ties suffix-first instead -- evict the end of the longest path
first -- and it matches tree LRU at 200, 250 and 300 blocks (85.3%, 91.9%,
95.1%). At 160, below one request, it does better still: 80.5% to 76.8%. A tree makes leaf-first order structural; a flat block pool can get
the same result by ordering its evictions correctly.

**FINDING: the lesson's own cache is not strictly tree-shaped.**
`RadixCache.insert` can evict a node's parent and then insert the child, so it
holds nodes that `walk` can never reach: up to 4 stranded blocks on the RAG
run. It also evicts whole segments, which is why it gets 69.1% at the budget
where block-level tree LRU gets 76.8%.

Structure: `run()` is one cache loop; the eviction rule is the only
parameter.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "06-sglang-radixattention"
BUDGETS = (160, 200, 250, 300)


def block_path(ref, segments):
    """Block i's id is the tuple of blocks 0..i, so a block names its whole prefix."""
    flat = [(s, i) for s in segments
            for i in range(-(-ref.token_count(s) // ref.BLOCK_TOKENS))]
    return [tuple(flat[: k + 1]) for k in range(len(flat))]


def victim(cache, rule):
    if rule == "tree":
        parents = {b[:-1] for b in cache}
        return min((b for b in cache if b not in parents), key=lambda b: (cache[b], len(b)))
    tie = (lambda b: len(b)) if rule == "block" else (lambda b: -len(b))
    return min(cache, key=lambda b: (cache[b], tie(b)))


def stranded(cache):
    """Cached blocks with some ancestor missing: resident but unreachable."""
    reachable = set()
    for b in sorted(cache, key=len):
        if len(b) == 1 or b[:-1] in reachable:
            reachable.add(b)
    return len(cache) - len(reachable)


def run(ref, reqs, budget, rule):
    """(block hit rate, peak stranded blocks)."""
    cache, hits, total, dead = {}, 0, 0, 0
    for now, r in enumerate(reqs, 1):
        path, k = block_path(ref, r.segments), 0
        while k < len(path) and path[k] in cache:
            cache[path[k]], k = now, k + 1
        hits, total = hits + k, total + len(path)
        for b in path[k:]:
            while len(cache) >= budget:
                del cache[victim(cache, rule)]
            cache[b] = now
        dead = max(dead, stranded(cache))
    return round(hits / total, 4), dead


def reference_stranded(ref, reqs):
    cache, peak = ref.RadixCache(), 0
    for r in reqs:
        cache.walk(r.segments)
        cache.insert(r.segments)
        peak = max(peak, sum(v[0] for k, v in cache.nodes.items()
                             if len(k) > 1 and k[:-1] not in cache.nodes))
    return peak


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rag = ref.workload_rag()
    runs = {rule: {b: run(ref, rag, b, rule) for b in BUDGETS}
            for rule in ("tree", "block", "suffix")}
    return {**runs, "ref_stranded": reference_stranded(ref, rag),
            "ref_hit": round(ref.simulate(rag, "FCFS")["hit_rate"], 4)}


def verify(result):
    tree, block, suffix = result["tree"], result["block"], result["suffix"]
    return [
        practice.Check(
            "ANSWER: block LRU evicts a cold branch's first block and strands the rest",
            all([tree[200] == (0.8531, 0), block[200] == (0.8211, 20),
                 tree[250] == (0.9187, 0), block[250] == (0.8744, 34),
                 block[160] == (0.0, 160), tree[160][0] == 0.7681]),
            f"(hit rate, peak stranded blocks) tree {tree} vs block {block}",
        ),
        practice.Check(
            "FINDING: what matters is the eviction order, not the data structure",
            all(suffix[b][0] == tree[b][0] for b in (200, 250, 300))
            and suffix[160][0] == 0.8046,
            f"block LRU with suffix-first ties {suffix} against tree {tree}",
        ),
        practice.Check(
            "FINDING: the lesson's own cache is not strictly tree-shaped",
            result["ref_stranded"] == 4 and result["ref_hit"] == 0.6906 < tree[160][0],
            f"RadixCache holds up to {result['ref_stranded']} unreachable blocks and hits "
            f"{result['ref_hit']:.1%} at 160, where block-level tree LRU hits {tree[160][0]:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
