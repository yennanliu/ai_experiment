"""Exercise 1 — the metrics agree on every distinct score and disagree only inside ties.

    **Metric comparison**: run the same 5 queries against the sample documents
    using cosine similarity, dot product, and euclidean distance. Record the
    top-3 results for each. For which queries do the metrics disagree? Why?

Reading of the exercise: the engine is the lesson's default
`SemanticSearchEngine()` over `SAMPLE_DOCUMENTS`, and "disagree" is read as the
top-3 index lists differing -- not the scores, which are on three different
scales by construction.

**ANSWER: two of the five queries disagree, and neither disagreement is about
similarity.** `SimpleEmbedder.embed` divides by the L2 norm, so every stored
vector is a unit vector. On unit vectors cosine *is* the dot product -- bit for
bit, not approximately -- and the euclidean distance is
`sqrt(2 - 2 * dot)`, a strictly decreasing function of it. The three metrics are
one ranking, and the only thing left for them to disagree about is what happens
when two scores are equal.

**MECHANISM: cosine and dot break ties by insertion order; euclidean breaks them
by floating-point noise.** `sort` is stable, so equal cosine scores keep index
order. Euclidean scores `-||q - v||`, and the stored norms are not exactly 1 --
they span 1.1e-16 across two distinct values -- so it orders tied chunks by
the last bit of their normalisation.

**FINDING: one of the five queries embeds to the zero vector.** "how do I reset
my password" has 0 of 270 dimensions non-zero, because not one of its six words
is in the vocabulary the embedder fitted on the chunks. Cosine and dot return
0.0 for every chunk and the ranking is `[0, 1, 2]`; euclidean returns `-||v||`
for every chunk and the ranking is float noise. That query retrieves nothing and
reports three results.

**FINDING: the two queries that disagree are exactly the two with a tie.**
Chunks scoring exactly 0.0 per query: `[0, 5, 0, 0, 4]`. The indices with a
non-zero count are `[1, 4]`, and the indices whose top-3 lists differ are
`[1, 4]`. No tie, no disagreement -- the correspondence is exact.

**CONTROL: break ties by index in all three metrics and all five queries
agree.** The disagreement the exercise asks about is entirely tie-breaking, so
the honest answer to "why" is that there is only ever one ranking signal here.

Structure: `rank` re-implements the lesson's search with an explicit tie-break,
`QUERIES` are the five, and `relations` checks the identities between metrics.
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


def rank(ref, query_vector, vectors, metric, k=3):
    """The control: the lesson's scoring, with ties broken by index rather than by luck."""
    scored = []
    for index, vector in enumerate(vectors):
        if metric == "cosine":
            score = ref.cosine_similarity(query_vector, vector)
        elif metric == "dot":
            score = ref.dot_product(query_vector, vector)
        else:
            score = -ref.euclidean_distance(query_vector, vector)
        scored.append((round(score, 12), -index))
    return [-i for _, i in sorted(scored, reverse=True)[:k]]


def residuals(ref, query_vector, vectors):
    """How far cosine is from the dot product, and euclidean from sqrt(2 - 2 cos)."""
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
    """Per query, whether the three metrics produced one top-3 list."""
    return [len({tuple(row[m]) for m in METRICS}) == 1 for row in rows]


def rankings(ref, engine, queries, vectors):
    shipped = unanimous([{m: top3(engine, q, m) for m in METRICS} for q in QUERIES])
    tied = unanimous([{m: rank(ref, v, vectors, m) for m in METRICS} for v in queries])
    norms = sorted({float(np.linalg.norm(v)) for v in vectors})
    return {"disagree": [i for i, ok in enumerate(shipped) if not ok],
            "agree_tied": sum(tied), "norm_values": len(norms),
            "norm_spread": norms[-1] - norms[0],
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
        **identities(ref, queries, vectors), **rankings(ref, engine, queries, vectors),
        "zero_query": [q for q, v in zip(QUERIES, queries) if not v.any()],
        "nonzero_dims": [int((v != 0).sum()) for v in queries],
        "zeros": [sum(1 for x in vectors if ref.dot_product(v, x) == 0.0) for v in queries],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two of the five queries disagree, and only inside ties",
            all([len(result["disagree"]) == 2, result["agree_tied"] == len(QUERIES)]),
            f"the top-3 lists differ on {len(result['disagree'])} of {len(QUERIES)} "
            f"queries: {[QUERIES[i][:26] for i in result['disagree']]}. Re-run the scoring with "
            f"ties broken by index and all {result['agree_tied']} agree, so nothing the "
            "metrics disagree about is a difference in similarity",
        ),
        practice.Check(
            "MECHANISM: on unit vectors cosine is the dot product and euclidean is its image",
            all([result["cos_gap"] < 3e-16, result["euclid_gap"] < 1e-9]),
            f"`SimpleEmbedder.embed` divides by the L2 norm, so cosine_similarity and "
            f"dot_product agree to {result['cos_gap']:.1e} over every query-chunk pair, and "
            f"euclidean matches sqrt(2 - 2 cos) to {result['euclid_gap']:.1e} over the "
            f"{result['unit_queries']} queries that normalise. The three are one ranking on "
            f"three scales -- except for the query that does not normalise, where the same "
            f"identity is off by {result['dead_gap']:.2f}",
        ),
        practice.Check(
            "MECHANISM: euclidean breaks ties on the last bit of the stored norms",
            all([result["norm_values"] > 1, result["norm_spread"] < 1e-15,
                 result["zero_cosine"] != result["zero_euclid"]]),
            f"the stored vectors take {result['norm_values']} distinct norms spanning "
            f"{result['norm_spread']:.1e}, so -||q - v|| separates chunks that cosine ties. "
            f"Cosine keeps insertion order ({result['zero_cosine']}); euclidean returns "
            f"{result['zero_euclid']} for the same query",
        ),
        practice.Check(
            "FINDING: one of the five queries embeds to the zero vector",
            all([len(result["zero_query"]) == 1, result["nonzero_dims"][1] == 0]),
            f"{result['zero_query'][0]!r} has {result['nonzero_dims'][1]} of "
            f"{result['dims']} dimensions non-zero: not one of its words is in the "
            "vocabulary the embedder fitted on the chunks. It retrieves nothing and "
            "reports three results with a score of 0.0",
        ),
        practice.Check(
            "FINDING: the two queries that disagree are exactly the two that have a tie",
            all([[i for i, n in enumerate(result["zeros"]) if n] == result["disagree"],
                 max(result["zeros"]) == result["chunks"]]),
            f"chunks scoring exactly 0.0 against each query: {result['zeros']} out of "
            f"{result['chunks']}. The indices with a non-zero count are "
            f"{[i for i, n in enumerate(result['zeros']) if n]} and the indices that "
            f"disagree are {result['disagree']}. The correspondence is exact: no tie, no "
            "disagreement",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
