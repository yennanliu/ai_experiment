"""Exercise 3 -- no offline stand-in moves semantic recall by more than 1 of 9, and a wider hash lifts every other strategy to 9 of 9.

    Replace the deterministic embedding with the one from your project's real provider. Measure the semantic-clustering recall delta. Report whether the spread between strategies widens or narrows.

Reading of the exercise: no provider is reachable offline, and the repo may not
download a model at run time, so the provider is played by five stand-ins
fitted to the fixture's own 23 sentences: the lesson's `mock_embed` at 4,096
dimensions instead of 96, an exact bag of words over the 195-word vocabulary,
TF-IDF, and two 8-dimension LSA projections of the TF-IDF rows (unit-length
rows, and raw rows). They go in where a provider would: `mock_embed` is a
module global, and `run` replaces it for both `DenseIndex` and
`semantic_chunks`. A learner with a key can pass a real embed function to
`run`. The delta and the spread (best minus worst strategy) are counted in
hits out of the fixture's 9 queries.

**ANSWER: semantic moves by at most 1 of 9 queries, and the spread has no
direction.** Hits@1 for semantic go 6 -> 7, 7, 6, 7 and 5 (delta -0.11 to
+0.11). The k=1 spread is 3 with the lesson's hash; it narrows to 2 under
three stand-ins, stays 3 under TF-IDF, and widens to 4 under the raw-row
LSA. Normalising the rows before the SVD is enough to flip that last verdict
from widen to narrow. At k=3 and k=5 the spread is 0 or 1 under every
embedding, because every strategy already finds nearly all 9.

**FINDING: the lesson's table ranks hash collisions, not strategies.** The
same `mock_embed` at 4,096 dimensions takes fixed 8 -> 9 and sentence 7 -> 9,
so every strategy except semantic reaches 9/9, the same result as the exact
bag of words. At 96 dimensions, 195 words share the buckets.

**FINDING: swapping the embedding re-chunks the corpus.** Semantic makes 20
chunks with the lesson's hash, 23 under all three lexical stand-ins, and 17
and 15 under the two LSAs. At 23, the 0.55 threshold cuts before every one
of the 23 sentences, so the "semantic" chunker is a one-sentence splitter.
This is the doc's "stale embeddings" failure mode, measured.

Structure: `stand_ins` fits the five embeddings; `run` patches the
lesson's `mock_embed`, restores it, and returns hits@1,3,5, the spread, and
the semantic chunk count under one embedding.
"""

from __future__ import annotations

import collections
import math
import re

from harness import parity, practice

import numpy as np

PHASE, LESSON = "19-capstone-projects", "64-chunking-strategies-advanced"
KS, LSA_RANK, WIDE_DIM = (1, 3, 5), 8, 4096
ORDER = ("fixed", "sentence", "recursive", "semantic", "structural")


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def corpus(ref, fixture):
    sents = [s for d in fixture for _, _, s in ref.split_sentences(d["text"])]
    queries = [q for d in fixture for q, _ in d["queries"]]
    vocab = sorted({w for t in sents + queries for w in words(t)})
    return sents, {w: i for i, w in enumerate(vocab)}


def idf_weights(sents, index):
    df = collections.Counter(w for s in sents for w in set(words(s)))
    return np.array([math.log((1 + len(sents)) / (1 + df[w])) + 1 for w in index])


def counts(index, text, weight=None):
    v = np.zeros(len(index))
    for w in words(text):
        if w in index:  # fixed windows cut words in half; the halves are not in the vocabulary
            v[index[w]] += 1
    return v if weight is None else v * weight


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n else v


def lsa_basis(rows):
    return np.linalg.svd(np.stack(rows), full_matrices=False)[2][:LSA_RANK].T


def stand_ins(ref, fixture):
    """Four embeddings fitted to the fixture's sentences, plus the lesson's hash made wide."""
    sents, index = corpus(ref, fixture)
    idf = idf_weights(sents, index)
    raw = [counts(index, s, idf) for s in sents]
    basis, basis_raw = lsa_basis([unit(r) for r in raw]), lsa_basis(raw)
    original = ref.mock_embed
    return (len(sents), len(index)), {
        "hash96": original,
        "hash4096": lambda t, dim=96: original(t, WIDE_DIM),
        "bow": lambda t, dim=96: list(unit(counts(index, t))),
        "tfidf": lambda t, dim=96: list(unit(counts(index, t, idf))),
        "lsa8": lambda t, dim=96: list(unit(counts(index, t, idf) @ basis)),
        "lsa8_raw_rows": lambda t, dim=96: list(unit(counts(index, t, idf) @ basis_raw)),
    }


def run(ref, fixture, embed):
    """`DenseIndex` and `semantic_chunks` both read the module global `mock_embed`."""
    original, ref.mock_embed = ref.mock_embed, embed
    try:
        hits = {n: [round(v * 9) for v in ref.eval_recall(fn, fixture, KS).values()]
                for n, fn in ref.STRATEGIES.items()}
        semantic = sum(len(ref.STRATEGIES["semantic"](d["doc_id"], d["text"])) for d in fixture)
    finally:
        ref.mock_embed = original
    spread = [max(h[i] for h in hits.values()) - min(h[i] for h in hits.values()) for i in range(3)]
    return {"hits": hits, "spread": spread, "semantic_chunks": semantic}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fixture = ref.build_fixture()
    sizes, embeds = stand_ins(ref, fixture)
    out = {name: run(ref, fixture, e) for name, e in embeds.items()}
    return {"sizes": sizes, "runs": out, "restored": ref.mock_embed is embeds["hash96"]}


def verify(result):
    runs = result["runs"]
    at1 = {e: [r["hits"][n][0] for n in ORDER] for e, r in runs.items()}
    spread, chunks = {e: r["spread"] for e, r in runs.items()}, {e: r["semantic_chunks"] for e, r in runs.items()}
    sem = {e: row[3] for e, row in at1.items()}
    return [
        practice.Check(
            "ANSWER: semantic moves by at most 1 of 9 and the k=1 spread narrows, holds or widens",
            (sem, spread, result["restored"])
            == ({"hash96": 6, "hash4096": 7, "bow": 7, "tfidf": 6, "lsa8": 7, "lsa8_raw_rows": 5},
                {"hash96": [3, 0, 0], "hash4096": [2, 0, 0], "bow": [2, 0, 0], "tfidf": [3, 0, 0],
                 "lsa8": [2, 0, 0], "lsa8_raw_rows": [4, 1, 0]}, True),
            f"semantic hits@1 {sem}; spread@1,3,5 {spread}",
        ),
        practice.Check(
            "FINDING: the lesson's table ranks hash collisions -- at 4,096 dims all but semantic hit 9/9",
            (at1["hash96"], at1["hash4096"], at1["bow"], result["sizes"])
            == ([8, 7, 9, 6, 9], [9, 9, 9, 7, 9], [9, 9, 9, 7, 9], (23, 195)),
            f"hits@1 {ORDER}: {at1}; sentences, vocabulary {result['sizes']}",
        ),
        practice.Check(
            "FINDING: swapping the embedding re-chunks -- 0.55 then cuts before all 23 sentences",
            chunks == {"hash96": 20, "hash4096": 23, "bow": 23, "tfidf": 23, "lsa8": 17, "lsa8_raw_rows": 15},
            f"semantic chunks per embedding {chunks} of {result['sizes'][0]} sentences",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
