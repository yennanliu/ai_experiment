"""Exercise 4 — 90% overlap at k=10 is forced by an index of 11, and k=1 is 0%.

    **Binary quantization**: take the embeddings from the search engine, convert
    them to binary (1 if positive, 0 if negative), and implement Hamming
    distance search. Compare the top-10 results against full-precision cosine
    similarity. Measure the overlap percentage.

Reading of the exercise: `binarize` and `hamming_distance` already ship, and
`VectorIndex.search` already has a `"hamming"` branch, so the exercise is read
as *measure the overlap it asks for and then measure it at the values of k where
it can vary*. The index is the lesson's own at chunk size 100: 11 chunks.

**ANSWER: 90% on all five queries, and it cannot be less.** Two subsets of size
10 drawn from 11 items must share at least 2*10 - 11 = 9 of them. The overlap
the exercise asks for is pinned by the size of the fixture, and would read 90%
for any metric whatsoever, including a random one.

**FINDING: at every k where the number can move, it collapses.** Overlap is
0.00 at k=1 on all five queries, 0.33 at k=3 and 0.40 at k=5. Binary
quantization gets the top hit wrong every time.

**MECHANISM: tf-idf is non-negative, so `(vec > 0)` is a presence bitmap.** The
smallest IDF in the fitted embedder is 1.087 and the smallest vector entry is
0.0, so `binarize` is exactly "this word occurs" -- every weight, and therefore
every notion of importance, is discarded. Hamming distance over presence
bitmaps is the size of the symmetric difference of two word sets.

**FINDING: what survives is length.** The number of ones correlates 0.992 with
the chunk's word count, and a query has three or four ones against a chunk's
thirty-seven to seventy-four, so the nearest chunk in Hamming terms is the
shortest one. The returned order is almost exactly the chunks sorted by size.

**CONTROL: normalise for length and the bitmaps are fine.** Jaccard over the
same binary vectors -- intersection over union, the length-invariant version of
the same information -- scores 0.40 at k=1 and 0.867 at k=3. The quantization
was never the problem; the unnormalised distance was.

Structure: `bitmaps` binarizes the index, `overlap` measures against cosine at
each k, and `jaccard` is the control distance.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "04-embeddings"
QUERIES = ["refund policy for enterprise customers", "how do I reset my password",
           "what is the SLA uptime guarantee", "data retention and deletion",
           "api rate limits"]
KS = (1, 3, 5, 10)


def jaccard(a, b):
    """The control: the same bitmaps, with length divided out."""
    union = int(np.sum((a == 1) | (b == 1)))
    return int(np.sum((a == 1) & (b == 1))) / union if union else 0.0


def order_by(scores, k):
    return [i for i, _ in sorted(enumerate(scores), key=lambda p: -p[1])[:k]]


def rankings(ref, engine, query, bits, k):
    vector = engine.embedder.embed(query)
    query_bits = ref.binarize(vector)
    cosine = order_by([ref.cosine_similarity(vector, v) for v in engine.index.vectors], k)
    hamming = order_by([-ref.hamming_distance(query_bits, b) for b in bits], k)
    return cosine, hamming, order_by([jaccard(query_bits, b) for b in bits], k)


def overlaps(ref, engine, bits, k):
    rows = [rankings(ref, engine, q, bits, k) for q in QUERIES]
    return {"hamming": [len(set(c) & set(h)) / k for c, h, _ in rows],
            "jaccard": [len(set(c) & set(j)) / k for c, _, j in rows]}


def presence(ref, vectors):
    """Is binarize's threshold doing anything, given non-negative tf-idf?"""
    return all((ref.binarize(v) == (v != 0)).all() for v in vectors)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "embeddings")
    engine = ref.SemanticSearchEngine(chunk_size=100, overlap=50)
    size = engine.index_documents(ref.SAMPLE_DOCUMENTS)
    bits = [ref.binarize(v) for v in engine.index.vectors]
    ones = [int(b.sum()) for b in bits]
    words = [len(t.split()) for t in engine.index.texts]
    by_k = {k: overlaps(ref, engine, bits, k) for k in KS}
    hamming_order = rankings(ref, engine, QUERIES[0], bits, size)[1]
    return {
        "size": size, "floor": (2 * 10 - size) / 10, **means(by_k),
        "presence": presence(ref, engine.index.vectors),
        "min_idf": round(float(engine.embedder.idf.min()), 3),
        "min_entry": min(float(v.min()) for v in engine.index.vectors),
        "ones": ones, "correlation": round(float(np.corrcoef(ones, words)[0, 1]), 4),
        "by_size": [ones[i] for i in hamming_order],
    }


def means(by_k):
    return {"hamming": {k: round(sum(by_k[k]["hamming"]) / len(QUERIES), 3) for k in KS},
            "jaccard": {k: round(sum(by_k[k]["jaccard"]) / len(QUERIES), 3) for k in KS},
            "at_ten": by_k[10]["hamming"], "at_one": by_k[1]["hamming"]}


def verify(result):
    hamming, jaccard_scores = result["hamming"], result["jaccard"]
    ordered = result["by_size"]
    inversions = sum(a > b for a, b in zip(ordered, ordered[1:]))
    return [
        practice.Check(
            "ANSWER: 90% at k=10 on all five queries, and it cannot be less",
            all([set(result["at_ten"]) == {0.9}, result["floor"] == 0.9]),
            f"the index holds {result['size']} chunks, so two subsets of size 10 must share "
            f"at least 2*10 - {result['size']} = 9 of them: the floor is "
            f"{result['floor']:.0%} and the measurement is {set(result['at_ten'])}. The "
            "number the exercise asks for is pinned by its own fixture",
        ),
        practice.Check(
            "FINDING: at every k where the number can move, it collapses",
            all([set(result["at_one"]) == {0.0}, hamming[3] < 0.5, hamming[5] < 0.5]),
            f"mean overlap by k: {hamming}. At k=1 it is {set(result['at_one'])} on all "
            f"five queries -- binary quantization gets the top hit wrong every time -- and "
            "it only reaches 0.9 where the pigeonhole forces it",
        ),
        practice.Check(
            "MECHANISM: tf-idf is non-negative, so (vec > 0) is a presence bitmap",
            all([result["presence"], result["min_idf"] > 1, result["min_entry"] == 0.0]),
            f"the smallest IDF is {result['min_idf']} and the smallest vector entry is "
            f"{result['min_entry']}, so binarize(v) equals (v != 0) on every stored vector. "
            "Every weight is discarded, and Hamming distance over presence bitmaps is the "
            "size of the symmetric difference of two word sets",
        ),
        practice.Check(
            "FINDING: what survives is length",
            all([result["correlation"] > 0.99, inversions <= 1]),
            f"the number of ones correlates {result['correlation']} with the chunk's word "
            f"count, and the Hamming ranking returned for query 1 has ones "
            f"{ordered} -- {inversions} inversion from sorted. A query has a handful of "
            "ones, so the nearest chunk in Hamming terms is the shortest chunk",
        ),
        practice.Check(
            "CONTROL: divide the length out and the same bitmaps are fine",
            all([jaccard_scores[1] > 0.3, jaccard_scores[3] > 0.8]),
            f"Jaccard over the identical binary vectors -- intersection over union, the "
            f"length-invariant reading of the same bits -- scores {jaccard_scores} against "
            f"Hamming's {hamming}. The quantization was never the problem; the "
            "unnormalised distance was",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
