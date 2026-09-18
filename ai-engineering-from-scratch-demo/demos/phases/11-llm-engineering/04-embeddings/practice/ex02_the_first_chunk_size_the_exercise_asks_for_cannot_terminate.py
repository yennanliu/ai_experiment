"""Exercise 2 — the first chunk size the exercise asks for cannot terminate.

    **Chunk size experiment**: index the sample documents with chunk sizes of
    50, 100, 200, and 500 words. For each, run 5 queries and record the top-1
    similarity score. Plot the relationship between chunk size and retrieval
    quality. Find the point where larger chunks start hurting.

Reading of the exercise: the overlap is the one the engine defaults to, 50,
because the exercise names only the chunk size. `chunk_text` is never called at
(50, 50) -- its step is verified arithmetically and reproduced with a bounded
clone -- and the other three sizes are run through `SemanticSearchEngine`
end to end.

**ANSWER: chunk size 50 hangs.** `chunk_text` advances by
`chunk_size - overlap`, which is 0 at (50, 50), so the `while start < len(words)`
loop appends the same chunk forever. A bounded clone emits 1,000 identical
chunks with `start` still at 0. `SemanticSearchEngine(chunk_size=50)` therefore
never returns from `index_documents`, and the exercise's first data point is
unreachable.

**FINDING: two of the remaining three are the same experiment.** The longest
sample document is 102 words, so any chunk size at or above it gives one chunk
per document: 200 and 500 both produce 5 chunks and byte-identical indexes.
Of the four sizes requested, one hangs and two coincide -- there is one
comparison, not a curve.

**FINDING: the scores are not comparable across sizes anyway.**
`index_documents` calls `self.embedder.fit(all_chunks)`, so the IDF vector is a
function of the chunk count: its sum is 660.65 at 100 chunks-of-11 and 537.35 at
5. The vocabulary is identical (270 words) and the same query still embeds to a
different vector, cosine 0.998785 between the two.

**ANSWER: there is no point where larger chunks start hurting, and no
consistent direction either.** Going from 100 to 200, three of the five queries
improve (0.4502 -> 0.4714, 0.4152 -> 0.4389, 0.4070 -> 0.4366), one gets worse
(0.3614 -> 0.3539) and one is 0.0 both ways. 500 is identical to 200. Two usable
points, disagreeing.

**CONTROL: fit the embedder once and the curve flattens.** Holding the
embedding space fixed across chunkings, top-1 for query 1 is 0.4714 at all three
sizes, and the largest move across the five queries is 0.0030. What the
exercise's plot would have shown is the refit.

Structure: `bounded` is the non-terminating loop with a cap, `sweep` runs the
engine per size, and `shared` is the fixed-embedder control.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "04-embeddings"
QUERIES = ["refund policy for enterprise customers", "how do I reset my password",
           "what is the SLA uptime guarantee", "data retention and deletion",
           "api rate limits"]
SIZES = (100, 200, 500)
OVERLAP = 50


def bounded(text, chunk_size, overlap, cap=1000):
    """`chunk_text`, with a cap: returns (chunks, final start, whether it terminated."""
    words, chunks, start = text.split(), [], 0
    while start < len(words) and len(chunks) < cap:
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks, start, len(chunks) < cap


def index(ref, size):
    engine = ref.SemanticSearchEngine(chunk_size=size, overlap=OVERLAP)
    count = engine.index_documents(ref.SAMPLE_DOCUMENTS)
    return engine, count


def sweep(ref):
    rows = {}
    for size in SIZES:
        engine, count = index(ref, size)
        rows[size] = {"chunks": count, "idf_sum": round(float(engine.embedder.idf.sum()), 2),
                      "vocab": len(engine.embedder.vocab), "texts": list(engine.index.texts),
                      "top1": [round(engine.search(q, top_k=1)[0]["score"], 4)
                               for q in QUERIES]}
    return rows


def shared(ref):
    """The control: one embedder fitted on the documents, reused for every chunking."""
    embedder = ref.SimpleEmbedder()
    embedder.fit(ref.SAMPLE_DOCUMENTS)
    rows = {}
    for size in SIZES:
        chunks = [c for d in ref.SAMPLE_DOCUMENTS for c in ref.chunk_text(d, size, OVERLAP)]
        vectors = [embedder.embed(c) for c in chunks]
        rows[size] = [round(max(ref.cosine_similarity(embedder.embed(q), v) for v in vectors), 4)
                      for q in QUERIES]
    return rows


def summarise(rows):
    return {"chunks": {s: rows[s]["chunks"] for s in SIZES},
            "same_index": rows[200]["texts"] == rows[500]["texts"],
            "idf": {s: rows[s]["idf_sum"] for s in SIZES},
            "vocab": sorted({rows[s]["vocab"] for s in SIZES}),
            "top1": {s: rows[s]["top1"] for s in SIZES}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "embeddings")
    rows = sweep(ref)
    stuck, start, finished = bounded(ref.SAMPLE_DOCUMENTS[0], 50, OVERLAP)
    query = [index(ref, s)[0].embedder.embed(QUERIES[0]) for s in (100, 200)]
    return {
        "step": 50 - OVERLAP, "emitted": len(stuck), "start": start, "finished": finished,
        "identical_chunk": len(set(stuck)) == 1,
        "words": [len(d.split()) for d in ref.SAMPLE_DOCUMENTS],
        "query_drift": round(ref.cosine_similarity(*query), 6),
        **summarise(rows), "control": shared(ref),
    }


def verify(result):
    top1, control = result["top1"], result["control"]
    drift = [round(max(control[s][i] for s in SIZES) - min(control[s][i] for s in SIZES), 4)
             for i in range(len(QUERIES))]
    better = sum(b > a for a, b in zip(top1[100], top1[200]))
    worse = sum(b < a for a, b in zip(top1[100], top1[200]))
    return [
        practice.Check(
            "ANSWER: chunk size 50 hangs, because the step is zero",
            all([result["step"] == 0, not result["finished"], result["start"] == 0,
                 result["identical_chunk"]]),
            f"`chunk_text` advances by chunk_size - overlap = {result['step']} at (50, 50), "
            f"so the loop never moves: a bounded clone emits {result['emitted']} chunks, all "
            f"{'identical' if result['identical_chunk'] else 'distinct'}, with start still at "
            f"{result['start']}. index_documents(chunk_size=50) does not return",
        ),
        practice.Check(
            "FINDING: two of the remaining three sizes are the same experiment",
            all([result["chunks"][200] == result["chunks"][500], result["same_index"],
                 max(result["words"]) < 200]),
            f"the longest sample document is {max(result['words'])} words, so any size at "
            f"or above it is one chunk per document: {result['chunks']} chunks, and the 200 "
            "and 500 indexes hold byte-identical text. One hangs, two coincide -- there is "
            "one comparison to make, not a curve to plot",
        ),
        practice.Check(
            "FINDING: the scores are not comparable across sizes, because the space refits",
            all([len(result["vocab"]) == 1, result["idf"][100] != result["idf"][200],
                 result["query_drift"] < 1.0]),
            f"`index_documents` calls embedder.fit(all_chunks), so the IDF sum is "
            f"{result['idf']} across the three sizes while the vocabulary stays at "
            f"{result['vocab'][0]} words. The same query embeds to a different vector: "
            f"cosine {result['query_drift']} between the size-100 and size-200 versions",
        ),
        practice.Check(
            "ANSWER: no point where larger hurts, and no consistent direction either",
            all([top1[200] == top1[500], better == 3, worse == 1]),
            f"top-1 per query: 100 -> {top1[100]}, 200 -> {top1[200]}, 500 -> {top1[500]}. "
            f"From 100 to 200, {better} queries improve, {worse} gets worse and one is 0.0 "
            f"both ways; 500 is identical to 200. Two usable points that disagree, which is "
            "not a curve with a knee in it",
        ),
        practice.Check(
            "CONTROL: hold the embedding space fixed and the curve flattens",
            all([max(drift) < 0.01, control[200] == control[500]]),
            f"one embedder fitted on the documents and reused for every chunking gives "
            f"per-query spreads of {drift} across the three sizes, largest {max(drift)}. "
            "What the exercise's plot would have shown is the refit, not the chunking",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
