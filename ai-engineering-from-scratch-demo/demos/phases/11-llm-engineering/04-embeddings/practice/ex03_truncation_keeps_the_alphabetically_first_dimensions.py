"""Exercise 3 — truncation keeps the alphabetically first words, so the curve is not one.

    **Matryoshka simulation**: build a SimpleEmbedder that produces 500-d
    vectors. Truncate to 50, 100, 200, and 500 dimensions. Measure how
    retrieval recall degrades at each truncation. This simulates Matryoshka
    behavior without needing the real training trick.

Reading of the exercise: the lesson's `SimpleEmbedder` is fitted on the sample
documents at chunk size 100, which gives 270 dimensions rather than 500 -- the
dimensionality is the vocabulary, not a choice -- so the truncations measured
are 50, 100, 200 and the full 270. Recall is the overlap of top-3 against the
untruncated ranking, over the four queries that embed to something.

**ANSWER: recall does not degrade; it wanders.** 0.667 at 50 dimensions, 0.833
at 100, 0.667 at 200 and 1.000 at 270. Adding dimensions 101 to 200 makes
retrieval worse. Whatever that curve is, it is not a degradation curve.

**MECHANISM: the vocabulary is `sorted(vocab_set)`.** So "the first d
dimensions" means "the d alphabetically first words", and the first fifty are
`$29, $500, $99, 0.1%, 1.3., 10%, 100, ...`. Matryoshka's premise is that a
prefix is *sufficient*; an alphabetically sorted vocabulary gives a prefix that
is arbitrary, and the measured recall is a fact about the alphabet.

**FINDING: at 50 dimensions some queries truncate to the zero vector**, because
none of their words sorts that early. Those queries score 0.0 against every
chunk and the ranking falls back to insertion order, so part of the recall at
the short end is scored against an ordering nothing produced.

**CONTROL: order the dimensions by document frequency first and the curve
becomes one.** 0.583, 0.917, 1.000, 1.000 -- monotone non-decreasing, reaching
full recall at 200 of 270 dimensions. That is what a Matryoshka prefix behaves
like.

**FINDING: the obvious importance proxy makes it worse.** Ordering by IDF --
rarest first -- gives 0.500, 0.500, 0.750, 1.000, below the alphabet at two of
the four points, because the rarest words are the ones fewest chunks share.
Which prefix is sufficient is exactly what the training trick decides, so the
exercise's claim to simulate Matryoshka without it is the thing that fails.

Structure: `recall_curve` scores a dimension ordering, and `ORDERINGS` holds the
alphabetical one the lesson ships plus the two controls.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "04-embeddings"
QUERIES = ["refund policy for enterprise customers", "how do I reset my password",
           "what is the SLA uptime guarantee", "data retention and deletion",
           "api rate limits"]
DIMS = (50, 100, 200)
TOP_K = 3


def orderings(matrix, idf):
    """The lesson's dimension order, and the two candidate re-orderings."""
    return {"alphabetical": np.arange(matrix.shape[1]),
            "docfreq": np.argsort(-(matrix > 0).sum(axis=0)),
            "idf": np.argsort(-idf)}


def ranked(ref, query, vectors, k=TOP_K):
    scores = [ref.cosine_similarity(query, v) for v in vectors]
    return [i for i, _ in sorted(enumerate(scores), key=lambda p: -p[1])[:k]]


def recall_at(ref, dim, order, queries, matrix):
    """Top-3 overlap against the untruncated ranking, over the live queries."""
    hits, total = 0, 0
    reduced = [ref.truncate_embedding(row[order], dim) for row in matrix]
    for query in queries:
        base = ranked(ref, query, list(matrix))
        got = ranked(ref, ref.truncate_embedding(query[order], dim), reduced)
        hits, total = hits + len(set(base) & set(got)), total + TOP_K
    return round(hits / total, 3)


def recall_curve(ref, order, queries, matrix):
    dims = list(DIMS) + [matrix.shape[1]]
    return [recall_at(ref, d, order, queries, matrix) for d in dims]


def dead_queries(ref, queries, dim):
    return sum(1 for q in queries if not ref.truncate_embedding(q, dim).any())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "embeddings")
    engine = ref.SemanticSearchEngine(chunk_size=100, overlap=50)
    engine.index_documents(ref.SAMPLE_DOCUMENTS)
    matrix = np.array(engine.index.vectors)
    queries = [q for q in (engine.embedder.embed(t) for t in QUERIES) if q.any()]
    orders = orderings(matrix, engine.embedder.idf)
    return {
        "dims": matrix.shape[1], "chunks": matrix.shape[0], "live": len(queries),
        "vocab_head": engine.embedder.vocab[:7],
        "curves": {name: recall_curve(ref, order, queries, matrix)
                   for name, order in orders.items()},
        "dead": {d: dead_queries(ref, queries, d) for d in DIMS},
    }


def verify(result):
    alpha = result["curves"]["alphabetical"]
    docfreq, idf = result["curves"]["docfreq"], result["curves"]["idf"]
    return [
        practice.Check(
            "ANSWER: recall does not degrade with dimension; it wanders",
            all([alpha == [0.667, 0.833, 0.667, 1.0], alpha[2] < alpha[1]]),
            f"top-3 recall against the untruncated ranking at 50, 100, 200 and "
            f"{result['dims']} dimensions: {alpha}. Adding dimensions 101 to 200 makes "
            "retrieval worse. Whatever that is, it is not a degradation curve",
        ),
        practice.Check(
            "MECHANISM: the vocabulary is sorted, so the prefix is alphabetical",
            all([result["dims"] == 270, result["vocab_head"][0].startswith("$")]),
            f"the embedder sets vocab = sorted(vocab_set), so 'the first d dimensions' is "
            f"'the d alphabetically first words' and the first seven are "
            f"{result['vocab_head']}. Matryoshka's premise is that a prefix is sufficient; "
            f"here it is arbitrary, and {result['dims']} is the vocabulary size, not a "
            "dimensionality anyone chose",
        ),
        practice.Check(
            "FINDING: at 50 dimensions some queries truncate to the zero vector",
            all([result["dead"][50] > 0, result["dead"][200] == 0]),
            f"queries whose truncation is all zeros: {result['dead']} at 50, 100 and 200 "
            f"dimensions, out of {result['live']} live queries. Those score 0.0 against "
            "every chunk and fall back to insertion order, so part of the recall at the "
            "short end is scored against an ordering nothing produced",
        ),
        practice.Check(
            "CONTROL: order the dimensions by document frequency and the curve becomes one",
            all([docfreq == sorted(docfreq), docfreq[2] == 1.0]),
            f"document-frequency order gives {docfreq} -- monotone non-decreasing, full "
            f"recall at 200 of {result['dims']} dimensions -- against the alphabet's "
            f"{alpha}. That is how a Matryoshka prefix behaves, and it needed no training, "
            "only an ordering that means something",
        ),
        practice.Check(
            "FINDING: the obvious importance proxy makes it worse",
            all([idf[0] < alpha[0], idf[1] < alpha[1], idf[-1] == 1.0]),
            f"IDF order -- rarest dimensions first -- gives {idf}, below the alphabet at "
            f"two of the four points. The rarest words are the ones fewest chunks share, so "
            "ranking by surprise ranks by absence. Which prefix is sufficient is exactly "
            "what the training trick decides",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
