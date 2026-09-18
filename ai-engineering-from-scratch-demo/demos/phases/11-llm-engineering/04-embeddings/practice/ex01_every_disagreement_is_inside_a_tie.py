"""Exercise 1 — every disagreement is inside a tie, and the count depends on the machine.

    **Metric comparison**: run the same 5 queries against the sample documents
    using cosine similarity, dot product, and euclidean distance. Record the
    top-3 results for each. For which queries do the metrics disagree? Why?

Reading of the exercise: the engine is the lesson's default
`SemanticSearchEngine()` over `SAMPLE_DOCUMENTS`, and "disagree" is read as the
top-3 index lists differing -- not the scores, which are on three different
scales by construction. The answer is then checked for durability, because a
disagreement produced by the last bit of a floating-point norm is not a
property of the metrics.

**ANSWER: the metrics disagree only on queries that contain a tie, and which of
those queries disagrees depends on the processor.** Two of the five queries have
tied scores -- 5 tied chunks and 4 tied chunks. On arm64 both of them disagree;
on the x86_64 runner only the first does, because the tied block's ordering
comes from norms that differ by one unit in the last place and the two
platforms round the dot product differently. Breaking ties by index makes all
five queries agree on both.

**MECHANISM: on unit vectors cosine is the dot product and euclidean is its
image.** `SimpleEmbedder.embed` divides by the L2 norm, so `cosine_similarity`
and `dot_product` agree to within 3e-16 over every query-chunk pair, and
`euclidean_distance` matches `sqrt(2 - 2 * dot)` to 1e-9. The three metrics are
one ranking on three scales, so the only thing left for them to disagree about
is what happens when two scores are equal.

**MECHANISM: euclidean ranks a tie by the stored norm; cosine keeps insertion
order.** For the query that embeds to the zero vector, euclidean scores
`-||0 - v|| = -||v||`, so its top-3 is exactly the three smallest-norm chunks
while cosine returns `[0, 1, 2]`. The stored norms are not exactly 1 -- they
take more than one value, spanning about 1e-16, and both the spread and the
number of distinct values are platform-dependent.

**FINDING: no disagreement can be about similarity.** Across the three queries
with no tie, the smallest gap between two distinct cosine scores is 0.00249 --
thirteen orders of magnitude above the noise that separates the metrics. So the
set of disagreeing queries is always a subset of the set of tied ones, whatever
the machine, and "why do they disagree" has one answer: they do not, except
where there is nothing to disagree about.

**FINDING: one of the five queries embeds to the zero vector.** "how do I reset
my password" has 0 of 270 dimensions non-zero, because not one of its six words
is in the vocabulary the embedder fitted on the chunks. It retrieves nothing and
reports three results with a score of 0.0, which is where 5 of the 9 tied
scores come from.

Structure: `score` is one metric on one pair and `rank` re-implements the
lesson's scoring with an explicit tie-break, so `unanimous` can be asked of
both. `residuals` and `identities` check the algebra between the three metrics,
`rankings` collects the shipped top-3 lists and the stored norms, and `ties`
counts exact-zero scores and measures the smallest gap between distinct ones.
"""

from __future__ import annotations

import math

import numpy as np

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "04-embeddings"
QUERIES = ["refund policy for enterprise customers", "how do I reset my password",
           "what is the SLA uptime guarantee", "data retention and deletion",
           "api rate limits"]
METRICS = ("cosine", "dot", "euclidean")


def top3(engine, query, metric):
    return [hit["index"] for hit in engine.search(query, top_k=3, metric=metric)]


def score(ref, query_vector, vector, metric):
    if metric == "cosine":
        return ref.cosine_similarity(query_vector, vector)
    if metric == "dot":
        return ref.dot_product(query_vector, vector)
    return -ref.euclidean_distance(query_vector, vector)


def rank(ref, query_vector, vectors, metric, k=3):
    scored = [(round(score(ref, query_vector, v, metric), 12), -i)
              for i, v in enumerate(vectors)]
    return [-i for _, i in sorted(scored, reverse=True)[:k]]


def residuals(ref, query_vector, vectors):
    gaps, implied = [], []
    for vector in vectors:
        cosine = ref.cosine_similarity(query_vector, vector)
        gaps.append(abs(cosine - ref.dot_product(query_vector, vector)))
        want = math.sqrt(max(0.0, 2 - 2 * cosine))
        implied.append(abs(ref.euclidean_distance(query_vector, vector) - want))
    return max(gaps), max(implied)


def identities(ref, queries, vectors):
    gaps = [residuals(ref, v, vectors) for v in queries if v.any()]
    return {"cos_gap": max(g for g, _ in gaps), "euclid_gap": max(g for _, g in gaps),
            "dead_gap": residuals(ref, queries[1], vectors)[1], "unit_queries": len(gaps)}


def unanimous(rows):
    return [len({tuple(row[m]) for m in METRICS}) == 1 for row in rows]


def ties(ref, queries, vectors):
    zeros, spreads = [], []
    for query_vector in queries:
        scores = sorted({round(ref.cosine_similarity(query_vector, v), 15) for v in vectors})
        zeros.append(sum(1 for v in vectors if ref.dot_product(query_vector, v) == 0.0))
        spreads += [b - a for a, b in zip(scores, scores[1:])]
    return {"zeros": zeros, "tied": [i for i, n in enumerate(zeros) if n],
            "min_gap": round(min(spreads), 5)}


def rankings(ref, engine, queries, vectors):
    shipped = unanimous([{m: top3(engine, q, m) for m in METRICS} for q in QUERIES])
    tied = unanimous([{m: rank(ref, v, vectors, m) for m in METRICS} for v in queries])
    norms = [float(np.linalg.norm(v)) for v in vectors]
    return {"disagree": [i for i, ok in enumerate(shipped) if not ok],
            "agree_tied": sum(tied), "norm_values": len(set(norms)),
            "norm_spread": max(norms) - min(norms),
            "smallest_norms": sorted(range(len(norms)), key=lambda i: norms[i])[:3],
            "zero_cosine": top3(engine, QUERIES[1], "cosine"),
            "zero_euclid": top3(engine, QUERIES[1], "euclidean")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "embeddings")
    engine = ref.SemanticSearchEngine()
    chunks = engine.index_documents(ref.SAMPLE_DOCUMENTS)
    vectors = engine.index.vectors
    queries = [engine.embedder.embed(q) for q in QUERIES]
    return {
        "chunks": chunks, "dims": len(engine.embedder.vocab),
        **identities(ref, queries, vectors), **ties(ref, queries, vectors),
        **rankings(ref, engine, queries, vectors),
        "zero_query": [q for q, v in zip(QUERIES, queries) if not v.any()],
        "nonzero_dims": [int((v != 0).sum()) for v in queries],
    }


def verify(result):
    disagree, tied = result["disagree"], result["tied"]
    return [
        practice.Check(
            "ANSWER: the metrics disagree only on queries that contain a tie",
            all([set(disagree) <= set(tied), len(disagree) >= 1,
                 result["agree_tied"] == len(QUERIES)]),
            f"{len(tied)} of the {len(QUERIES)} queries have tied scores, at indices {tied}, "
            f"and the {len(disagree)} whose top-3 lists differ are {disagree} -- a subset, "
            f"here and on any other machine. Break ties by index and all "
            f"{result['agree_tied']} agree: nothing they disagree about is a similarity",
        ),
        practice.Check(
            "MECHANISM: on unit vectors cosine is the dot product and euclidean is its image",
            all([result["cos_gap"] < 3e-16, result["euclid_gap"] < 1e-9]),
            f"`SimpleEmbedder.embed` divides by the L2 norm, so cosine_similarity and "
            f"dot_product agree to {result['cos_gap']:.1e} over every query-chunk pair, and "
            f"euclidean matches sqrt(2 - 2 cos) to {result['euclid_gap']:.1e} over the "
            f"{result['unit_queries']} queries that normalise -- one ranking on three "
            f"scales, except for the query that does not normalise, where the same identity "
            f"is off by {result['dead_gap']:.2f}",
        ),
        practice.Check(
            "MECHANISM: euclidean ranks a tie by the stored norm, cosine by insertion order",
            all([result["norm_values"] > 1, result["norm_spread"] < 1e-15,
                 result["zero_euclid"] == result["smallest_norms"],
                 result["zero_cosine"] == [0, 1, 2]]),
            f"for the all-tied query euclidean scores -||v||, so its top-3 is exactly the "
            f"three smallest-norm chunks, {result['zero_euclid']}, while cosine keeps "
            f"insertion order, {result['zero_cosine']}. The stored norms take "
            f"{result['norm_values']} distinct values spanning {result['norm_spread']:.1e}, "
            "and both of those numbers are platform-dependent",
        ),
        practice.Check(
            "FINDING: no disagreement can be about similarity",
            all([result["min_gap"] > 0.002, result["min_gap"] > 1e12 * 3e-16,
                 set(disagree) <= set(tied)]),
            f"the smallest gap between two distinct cosine scores is {result['min_gap']}, "
            "thirteen orders of magnitude above the noise that separates the metrics, so no "
            "reordering of distinct scores is possible. Exact-zero counts per query: "
            f"{result['zeros']} out of {result['chunks']} chunks",
        ),
        practice.Check(
            "FINDING: one of the five queries embeds to the zero vector",
            all([len(result["zero_query"]) == 1, result["nonzero_dims"][1] == 0,
                 result["zeros"][1] == result["chunks"]]),
            f"{result['zero_query'][0]!r} has {result['nonzero_dims'][1]} of "
            f"{result['dims']} dimensions non-zero: not one of its words is in the "
            f"vocabulary the embedder fitted on the chunks. It reports three results with a "
            f"score of 0.0, which is where {result['zeros'][1]} of the "
            f"{sum(result['zeros'])} tied scores come from",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
