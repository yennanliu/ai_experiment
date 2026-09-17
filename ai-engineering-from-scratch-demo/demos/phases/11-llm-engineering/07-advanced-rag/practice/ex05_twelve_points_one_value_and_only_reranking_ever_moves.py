"""Exercise 5 — twelve points, one value, and only reranking ever moves.

    Create an evaluation dataset: 10 questions with known answer chunks.
    Measure Recall@3, Recall@5, and Recall@10 for (a) vector search only,
    (b) BM25 only, (c) hybrid search, (d) hybrid + reranking. Plot the results
    and identify where reranking helps most.

Reading of the exercise: the known answer chunks are derived rather than
asserted -- a chunk is relevant when it contains the question's answer string --
so the labels cannot drift from the chunking. Recall is the lesson's own
`evaluate_retrieval_recall`, and the four configurations are the lesson's own
functions unmodified.

**ANSWER: at the shipped chunk size the plot is twelve points at 1.000.** Six
chunks, ten questions, three values of k, four configurations: every cell is
perfect recall. k=10 against a 6-chunk index retrieves everything, k=5 retrieves
everything, and k=3 is enough because each answer sits in exactly one of six
documents. There is nowhere for reranking to help.

**FINDING: re-chunk to 31 and three of the four configurations are one line.**
Vector, BM25 and hybrid score 0.778, 0.889, 1.000 at k = 3, 5, 10 -- identical
to each other at every k. The exercise asks for four curves and the corpus
supports two.

**ANSWER: reranking helps most at k=3, and it is the only thing that helps.**
Hybrid + rerank scores 1.000 at every k, so it gains +0.222 at k=3, +0.111 at
k=5 and nothing at k=10. Reranking can only reorder within the pool, so its
value is exactly the recall the smaller k was throwing away.

**MECHANISM: rerank's `initial_score * 5.0` term weighs its input 66x
differently depending on who fed it.** On hybrid candidates the initial score is
an RRF value near 0.032, so the term spans 0.133 to 0.164 -- a range of 0.031
against term-overlap counts that are whole numbers. On vector candidates it is a
cosine, so the same term spans 0.000 to 2.043. The same function treats its
input ranking as a constant in one case and as a real signal in the other.

**FINDING: the labels themselves move with the chunking.** At 20-word chunks one
of the ten answer strings is split across a boundary and the question becomes
unlabellable -- "$99 per month" spans two chunks. A derived-label evaluation set
is a function of the chunk size it was derived at.

Structure: `grid` runs the four configurations at three k, `labels` derives the
relevant set from the answer strings, and `rerank_weight` measures the input
term's contribution under each retriever.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "07-advanced-rag"
QA = [("what encryption is used at rest", "AES-256"),
      ("what is the enterprise refund window", "60-day"),
      ("what is the starter rate limit", "100 requests"),
      ("what uptime is guaranteed", "99.9%"),
      ("how much does professional cost", "$99 per month"),
      ("how long do refunds take", "5-7 business days"),
      ("what compliance certification is held", "SOC 2"),
      ("what was Q3 revenue", "$47.2 million"),
      ("what happens when the rate limit is exceeded", "429"),
      ("how are service credits calculated", "10% credit")]
KS = (3, 5, 10)
POOL = 15


def index(ref, chunk_size, overlap):
    chunks = [c for d in ref.SAMPLE_DOCUMENTS
              for c in ref.chunk_text(" ".join(d.split()), chunk_size, overlap)]
    vocab = ref.build_vocabulary(chunks)
    idf = ref.compute_idf(chunks, vocab)
    bm25 = ref.BM25()
    bm25.index(chunks)
    return chunks, vocab, idf, [ref.tfidf_embed(c, vocab, idf) for c in chunks], bm25


def labels(chunks):
    """A chunk is relevant when it holds the question's answer string."""
    return [(question, [i for i, c in enumerate(chunks) if answer.lower() in c.lower()])
            for question, answer in QA]


def configurations(ref, parts):
    chunks, vocab, idf, embeddings, bm25 = parts
    return {
        "vector": lambda q, k: ref.vector_search(ref.tfidf_embed(q, vocab, idf),
                                                 embeddings, k),
        "bm25": lambda q, k: bm25.search(q, k),
        "hybrid": lambda q, k: ref.hybrid_search(q, chunks, embeddings, vocab, idf, bm25, k),
        "hybrid+rerank": lambda q, k: ref.rerank(
            q, ref.hybrid_search(q, chunks, embeddings, vocab, idf, bm25, POOL), chunks)[:k],
    }


def grid(ref, parts):
    pairs = [(q, relevant) for q, relevant in labels(parts[0]) if relevant]
    return {name: {k: round(ref.evaluate_retrieval_recall(pairs, fn, k)[0], 3) for k in KS}
            for name, fn in configurations(ref, parts).items()}, len(pairs)


def rerank_weight(ref, parts, query):
    """What `initial_score * 5.0` contributes, under each retriever."""
    chunks, vocab, idf, embeddings, bm25 = parts
    hybrid = ref.hybrid_search(query, chunks, embeddings, vocab, idf, bm25, POOL)
    vector = ref.vector_search(ref.tfidf_embed(query, vocab, idf), embeddings, POOL)
    return {name: (round(min(s for _, s in rows) * 5, 3), round(max(s for _, s in rows) * 5, 3))
            for name, rows in (("hybrid", hybrid), ("vector", vector))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, finer = index(ref, 200, 50), index(ref, 20, 0)
    coarse_grid, coarse_pairs = grid(ref, shipped)
    fine_grid, fine_pairs = grid(ref, finer)
    return {
        "chunks": (len(shipped[0]), len(finer[0])), "questions": len(QA),
        "shipped": coarse_grid, "finer": fine_grid,
        "pairs": (coarse_pairs, fine_pairs),
        "unlabelled": [q for q, relevant in labels(finer[0]) if not relevant],
        "weights": rerank_weight(ref, finer, QA[0][0]),
        "pool": POOL,
    }


def verify(result):
    shipped, finer = result["shipped"], result["finer"]
    plain = {name: finer[name] for name in ("vector", "bm25", "hybrid")}
    lift = {k: round(finer["hybrid+rerank"][k] - finer["hybrid"][k], 3) for k in KS}
    hybrid_span, vector_span = (result["weights"]["hybrid"], result["weights"]["vector"])
    return [
        practice.Check(
            "ANSWER: at the shipped chunk size the plot is twelve points at 1.000",
            all([{v for row in shipped.values() for v in row.values()} == {1.0},
                 result["chunks"][0] == 6]),
            f"{result['chunks'][0]} chunks, {result['questions']} questions, "
            f"{len(KS)} values of k, {len(shipped)} configurations: every cell is "
            f"{shipped['vector'][3]}. k=10 and k=5 retrieve the whole corpus, and k=3 is "
            "enough because each answer sits in exactly one of six documents",
        ),
        practice.Check(
            "FINDING: re-chunk to 31 and three of the four configurations are one line",
            all([len({tuple(row.items()) for row in plain.values()}) == 1,
                 finer["vector"][3] == 0.778, result["chunks"][1] == 31]),
            f"at {result['chunks'][1]} chunks, vector, BM25 and hybrid all score "
            f"{list(finer['vector'].values())} at k = {list(KS)} -- identical to each other "
            "at every k. The exercise asks for four curves and the corpus supports two",
        ),
        practice.Check(
            "ANSWER: reranking helps most at k=3, and it is the only thing that helps",
            all([set(finer["hybrid+rerank"].values()) == {1.0}, lift[3] == 0.222,
                 lift[10] == 0.0]),
            f"hybrid + rerank scores {list(finer['hybrid+rerank'].values())} at k = "
            f"{list(KS)}, so the lift over hybrid is {lift}. Reranking can only reorder "
            "within the pool, so its value is exactly the recall the smaller k was throwing "
            "away",
        ),
        practice.Check(
            "MECHANISM: rerank weighs its input 60x differently depending on who fed it",
            all([hybrid_span[1] - hybrid_span[0] < 0.05,
                 vector_span[1] - vector_span[0] > 1.0]),
            f"`initial_score * 5.0` spans {hybrid_span} on hybrid candidates -- a range of "
            f"{hybrid_span[1] - hybrid_span[0]:.3f} against term-overlap counts that are "
            f"whole numbers -- and {vector_span} on vector candidates, where the score is a "
            f"cosine: {(vector_span[1] - vector_span[0]) / (hybrid_span[1] - hybrid_span[0]):.0f}x "
            "wider. The same function treats its input as a constant in one case and a "
            "signal in the other",
        ),
        practice.Check(
            "FINDING: the labels themselves move with the chunking",
            all([len(result["unlabelled"]) == 1, result["pairs"] == (10, 9)]),
            f"at 20-word chunks {result['unlabelled']} becomes unlabellable, because its "
            f"answer string '$99 per month' is split across a chunk boundary: the pair count "
            f"goes {result['pairs'][0]} -> {result['pairs'][1]}. A derived-label evaluation "
            "set is a function of the chunk size it was derived at",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
