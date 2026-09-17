"""Exercise 3 — the medium-confidence band is empty for any query under twenty words.

    **Implement tiered semantic caching.** Use two similarity thresholds: 0.98
    for high-confidence hits (return immediately) and 0.90 for medium-
    confidence hits (return with a disclaimer: "Based on a similar previous
    question..."). Track which tier each hit came from and measure user
    satisfaction differences.

Reading of the exercise: the tiers are built on the lesson's own
`simple_embed` and `cosine_similarity`, and the question asked first is which
similarities those two can produce -- because a threshold is only a threshold if
values land on both sides of it.

**ANSWER: for a query of n words with k of them shared, the cosine is exactly
k/n.** `simple_embed` is a normalised bag of words, so the similarity grid has
n+1 points. For an 8-word query the reachable values are 0, 0.125, 0.25, ...,
1.0, and *none* of them falls in [0.90, 0.98). The medium tier is unreachable.

**FINDING: 10 words is the shortest query that can reach the band at all.**
Below n = 10 the gap between 1.0 and the next value down, (n-1)/n, is wider
than the band; at n = 10 the value 0.9 lands on its edge, and 20 words are
needed before anything lands strictly inside. The queries here are 5 and 6
words long.

**FINDING: the only hits are exact repeats.** Across 190 distinct pairs from a
catalogue of the lesson's own shape, 0 reach 0.98 and 0 land in [0.90, 0.98).
The 10 hits the tiered lookup records are the queries literally in the cache,
scoring 1.0. The disclaimer branch is written, reachable in principle, and
never taken.

**FINDING: the default threshold is below both tiers.** `SemanticCache` ships
with `similarity_threshold=0.85`, so the cache the lesson runs already accepts
matches the exercise's *medium* tier would reject -- 0.875 for a 7-of-8 word
overlap. Adding the tiers makes the cache stricter, not more nuanced.

**FINDING: "measure user satisfaction differences" has nothing to measure.**
There is no user, no rating and no feedback channel anywhere in the lesson:
`SemanticCache.stats` reports hits, misses, hit rate and size. The tier a hit
came from can be tracked; whether anyone was happier cannot.

Structure: `grid` enumerates the reachable similarities, `tiered_get` is the
two-threshold lookup, and `pairs` scores the catalogue against itself.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "11-caching-cost"
HIGH, MEDIUM = 0.98, 0.90
DISCLAIMER = "Based on a similar previous question..."
TOPICS = ("refund window", "api rate limit", "uptime guarantee", "data retention",
          "encryption at rest", "billing cycle", "team seats", "audit logs",
          "sso setup", "webhook retries")
CATALOGUE = [f"{verb} the {topic} policy" for topic in TOPICS
             for verb in ("explain", "describe")]


def grid(ref, n_words):
    """Every similarity two n-word queries can produce: k shared words gives k/n."""
    base = [f"w{i}" for i in range(n_words)]
    values = set()
    for shared in range(n_words + 1):
        other = base[:shared] + [f"x{i}" for i in range(n_words - shared)]
        values.add(round(ref.cosine_similarity(ref.simple_embed(" ".join(base)),
                                               ref.simple_embed(" ".join(other))), 6))
    return sorted(values)


def tiered_get(ref, cache, query):
    """The two-threshold lookup, with the tier recorded on every hit."""
    embedding = ref.simple_embed(query)
    best, score = None, 0.0
    for entry in cache.entries:
        similarity = ref.cosine_similarity(embedding, entry["embedding"])
        if similarity > score:
            best, score = entry, similarity
    if best is None or score < MEDIUM:
        return None
    tier = "high" if score >= HIGH else "medium"
    return {"tier": tier, "similarity": round(score, 4),
            "response": (best["response"] if tier == "high"
                         else f"{DISCLAIMER} {best['response']}")}


def pairs(ref, catalogue):
    return [round(ref.cosine_similarity(ref.simple_embed(a), ref.simple_embed(b)), 6)
            for a, b in itertools.combinations(catalogue, 2)]


def shortest_medium(ref, limit=40):
    for n_words in range(2, limit):
        if any(MEDIUM <= v < HIGH for v in grid(ref, n_words)):
            return n_words
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "caching_cost")
    scores = pairs(ref, CATALOGUE)
    cache = ref.SemanticCache()
    for query in CATALOGUE[:10]:
        cache.put(query, f"answer to {query}")
    tiers = [tiered_get(ref, cache, query) for query in CATALOGUE]
    return {
        "grid_8": grid(ref, 8),
        "grid_20": [v for v in grid(ref, 20) if MEDIUM <= v < HIGH],
        "shortest_medium": shortest_medium(ref), "default_threshold": cache.threshold,
        "seven_of_eight": round(7 / 8, 3), "stats_fields": sorted(cache.stats()),
        "demo_lengths": sorted({len(q.split()) for q in CATALOGUE}),
        **outcomes(scores, tiers),
    }


def outcomes(scores, tiers):
    return {"pairs": len(scores),
            "high_pairs": sum(1 for s in scores if s >= HIGH),
            "medium_pairs": sum(1 for s in scores if MEDIUM <= s < HIGH),
            "tiers": {tier: sum(1 for t in tiers if t and t["tier"] == tier)
                      for tier in ("high", "medium")},
            "misses": sum(1 for t in tiers if t is None)}


def verify(result):
    grid_8 = result["grid_8"]
    return [
        practice.Check(
            "ANSWER: the similarity grid has n+1 points and none is in the band",
            all([len(grid_8) == 9, not [v for v in grid_8 if MEDIUM <= v < HIGH],
                 0.875 in grid_8]),
            f"`simple_embed` is a normalised bag of words, so two n-word queries sharing k "
            f"words score exactly k/n. For an 8-word query the reachable values are "
            f"{grid_8} -- {len(grid_8)} points, and none of them in [{MEDIUM}, {HIGH})",
        ),
        practice.Check(
            "FINDING: 10 words is the shortest query that can reach the band at all",
            all([result["shortest_medium"] == 10, result["grid_20"] == [0.9, 0.95],
                 max(result["demo_lengths"]) < 10]),
            f"below n = {result['shortest_medium']} the gap between 1.0 and (n-1)/n is "
            f"wider than the band; at n = 10 the single value 0.9 lands on its edge and at "
            f"n = 20 the values {result['grid_20']} land inside it. The queries in play "
            f"here are {result['demo_lengths']} words long",
        ),
        practice.Check(
            "FINDING: the only hits are exact repeats",
            all([result["medium_pairs"] == 0, result["high_pairs"] == 0,
                 result["tiers"]["medium"] == 0, result["tiers"]["high"] == 10]),
            f"across {result['pairs']} distinct catalogue pairs, {result['high_pairs']} "
            f"reach {HIGH} and {result['medium_pairs']} land in the medium band -- none "
            f"reaches either tier. The {result['tiers']['high']} hits the tiered lookup "
            f"records are the queries literally in the cache, scoring 1.0, and the other "
            f"{result['misses']} miss. The disclaimer branch is never taken",
        ),
        practice.Check(
            "FINDING: the default threshold is below both tiers",
            all([result["default_threshold"] < MEDIUM,
                 result["seven_of_eight"] > result["default_threshold"]]),
            f"`SemanticCache` ships with similarity_threshold="
            f"{result['default_threshold']}, so the cache the lesson runs already accepts a "
            f"7-of-8 word overlap at {result['seven_of_eight']} -- a match the exercise's "
            "medium tier would reject. Adding the tiers makes the cache stricter",
        ),
        practice.Check(
            "FINDING: 'measure user satisfaction differences' has nothing to measure",
            result["stats_fields"] == ["cache_size", "hit_rate", "hits", "misses"],
            f"`SemanticCache.stats` reports {result['stats_fields']}. There is no user, no "
            "rating and no feedback channel anywhere in the lesson. The tier a hit came "
            "from can be tracked; whether anyone was happier with it cannot",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
