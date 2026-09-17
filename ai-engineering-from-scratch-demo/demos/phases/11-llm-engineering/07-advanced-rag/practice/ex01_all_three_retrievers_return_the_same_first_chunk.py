"""Exercise 1 — all three retrievers return the same first chunk, so hybrid wins 0 of 5.

    Compare BM25 vs vector search vs hybrid search on the sample documents. For
    each of the 5 test queries, record which approach returns the most relevant
    chunk in position #1. Hybrid search should win on at least 3 out of 5.

Reading of the exercise: "most relevant" is the document that actually answers
the query, recorded as a gold index per query, and the chunks are the lesson's
own at its default chunk size -- which is one chunk per document, six of them.
Hybrid is `hybrid_search` unchanged, pool and fusion constant included.

**ANSWER: hybrid wins 0 of 5, because it never differs.** Vector, BM25 and
hybrid return the identical rank-1 chunk on all five queries, and the same 4 of
5 are correct. There is no comparison to record.

**MECHANISM: the pool is larger than the corpus.** `hybrid_search` retrieves
`retrieval_pool=15` from each arm against a 6-chunk index, so both arms return
every chunk and the fusion is a rank-sum over identical candidate sets. Nothing
can enter or leave the pool.

**MECHANISM: `reciprocal_rank_fusion` discards the scores.** It reads only the
position in each list, so two ranked lists that differ by a factor of 99,000 in
score fuse to byte-identical values. Combined with `k=60`, the fused spread
across the whole six-chunk ranking is 0.00248 -- 7% of the top score -- so RRF
at this corpus size cannot express a preference either.

**FINDING: the query all three get wrong has one content word.** "how much does
professional cost" contributes `does` and `professional` to the embedding and
nothing else, and all three retrievers return the security document. The corpus never writes
"cost" or "much", so the ranking rests on term frequency over words the query
did not mean.

**CONTROL: the corpus needs 31 chunks before the arms disagree at all.**
Re-chunked without overlap: 13 chunks, 0 queries differ; 23 chunks, 0 differ; 31
chunks -- past the 15-item pool, five times the shipped corpus -- 1 differs. Only
then is there a comparison to record.

Structure: `arms` runs the three retrievers, `fused_spread` measures what RRF
can express, and `GOLD` records the answering document per query.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "07-advanced-rag"
QUERIES = ["what encryption is used at rest", "what is the enterprise refund window",
           "what is the starter rate limit", "what uptime is guaranteed",
           "how much does professional cost"]
GOLD = [2, 0, 3, 5, 1]


def build(ref, chunk_size=200, overlap=50):
    chunks = [c for d in ref.SAMPLE_DOCUMENTS for c in ref.chunk_text(d, chunk_size, overlap)]
    vocab = ref.build_vocabulary(chunks)
    idf = ref.compute_idf(chunks, vocab)
    bm25 = ref.BM25()
    bm25.index(chunks)
    return chunks, vocab, idf, [ref.tfidf_embed(c, vocab, idf) for c in chunks], bm25


def arms(ref, query, parts, k=1):
    chunks, vocab, idf, embeddings, bm25 = parts
    return {
        "vector": [i for i, _ in ref.vector_search(ref.tfidf_embed(query, vocab, idf),
                                                   embeddings, k)],
        "bm25": [i for i, _ in bm25.search(query, k)],
        "hybrid": [i for i, _ in ref.hybrid_search(query, chunks, embeddings, vocab, idf,
                                                   bm25, k)],
    }


def fused_spread(ref, query, parts, k=60):
    chunks, vocab, idf, embeddings, bm25 = parts
    vector = ref.vector_search(ref.tfidf_embed(query, vocab, idf), embeddings, 15)
    lexical = bm25.search(query, 15)
    fused = ref.reciprocal_rank_fusion([vector, lexical], k=k)
    return round(fused[0][1] - fused[-1][1], 5), [i for i, _ in fused]


def in_vocabulary(vocab, query):
    return sorted(w for w in query.lower().split() if w in vocab)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    parts = build(ref)
    chunks, vocab = parts[0], parts[1]
    first = [arms(ref, q, parts) for q in QUERIES]
    return {
        "chunks": len(chunks), "pool": 15, **agreement(first),
        "rrf_ignores_scores": (ref.reciprocal_rank_fusion([[(0, 99000.0), (1, 0.0)]])
                               == ref.reciprocal_rank_fusion([[(0, 0.001), (1, 0.0)]])),
        "spread_60": fused_spread(ref, QUERIES[0], parts)[0],
        "in_vocabulary": in_vocabulary(vocab, QUERIES[4]),
        "finer": {size: disagreements(ref, size) for size in (50, 30, 20)},
    }


def agreement(first):
    return {"identical": [len({tuple(row[a]) for a in row}) == 1 for row in first],
            "correct": {a: sum(row[a][0] == g for row, g in zip(first, GOLD))
                        for a in ("vector", "bm25", "hybrid")},
            "wrong_query": [i for i, (row, g) in enumerate(zip(first, GOLD))
                            if row["hybrid"][0] != g]}


def disagreements(ref, chunk_size):
    """How many queries the three arms rank differently, at a finer chunk size."""
    parts = build(ref, chunk_size, 0)
    rows = [arms(ref, q, parts) for q in QUERIES]
    return {"chunks": len(parts[0]),
            "differ": sum(len({tuple(row[a]) for a in row}) > 1 for row in rows)}


def verify(result):
    correct, finer = result["correct"], result["finer"]
    return [
        practice.Check(
            "ANSWER: hybrid wins 0 of 5, because it never differs",
            all([all(result["identical"]), len(set(correct.values())) == 1,
                 correct["hybrid"] == 4]),
            f"the three retrievers return the identical rank-1 chunk on "
            f"{sum(result['identical'])} of {len(QUERIES)} queries, and each is correct on "
            f"{correct}. Hybrid cannot win a comparison it is never in: the exercise asks "
            "for at least 3 of 5 and the answer is 0",
        ),
        practice.Check(
            "MECHANISM: the pool is larger than the corpus",
            result["pool"] > result["chunks"],
            f"`hybrid_search` retrieves retrieval_pool={result['pool']} from each arm "
            f"against a {result['chunks']}-chunk index, so both arms return every chunk and "
            "the fusion is a rank-sum over identical candidate sets. Nothing can enter the "
            "pool and nothing can be excluded from it",
        ),
        practice.Check(
            "MECHANISM: reciprocal rank fusion discards the scores",
            all([result["rrf_ignores_scores"], result["spread_60"] < 0.003]),
            f"RRF reads only the position in each list, so [(0, 99000.0), (1, 0.0)] and "
            f"[(0, 0.001), (1, 0.0)] fuse to identical values. With k=60 the fused spread "
            f"across the whole {result['chunks']}-chunk ranking is {result['spread_60']} -- "
            "6% of the top score -- so the fusion cannot express a preference either",
        ),
        practice.Check(
            "FINDING: the query all three get wrong is decided by stopwords",
            all([result["wrong_query"] == [4], result["in_vocabulary"] == ["does", "professional"]]),
            f"query {result['wrong_query']} -- {QUERIES[4]!r} -- has "
            f"{result['in_vocabulary']} as its only in-vocabulary words, one of them a "
            "stopword: the corpus never writes 'cost' or 'much'. All three retrievers "
            "return the security document, and they agree because there was nothing left "
            "to disagree about",
        ),
        practice.Check(
            "CONTROL: the corpus needs 31 chunks before the arms disagree at all",
            all([finer[50]["differ"] == 0, finer[30]["differ"] == 0,
                 finer[20]["differ"] == 1, finer[20]["chunks"] > result["pool"]]),
            f"re-chunking without overlap: {{size: (chunks, queries where the arms differ)}} "
            f"= { {k: (v['chunks'], v['differ']) for k, v in finer.items()} }. Not until "
            f"{finer[20]['chunks']} chunks -- past the {result['pool']}-item pool, five "
            "times the shipped corpus -- does any retriever rank something first that the "
            "others do not. Only then is there a comparison to record",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
