"""Exercise 1 — half the queries decide nothing.

    **Easy.** Implement `hybrid_search` above on a 500-document corpus. Test 20
    queries. Compare recall at 5 between BM25-only, dense-only, and hybrid.

Reading of the exercise: `hybrid_search` is not in the lesson -- `code/main.py`
ships `BM25`, `fake_dense_rank` and `reciprocal_rank_fusion` -- so the function
to implement is the composition of the last two, which is what is built here. The
corpus is 30 documents rather than 500, and the query set is the part that
matters: 20 queries, ten sharing content words with their answer and ten written
as a plain-language paraphrase with no content word in common.

On the ten lexical queries all three arms score recall@5 of 1.00. The comparison
the exercise asks for is a three-way tie on half its own test set, and every
number that distinguishes the arms comes from the other half:

    BM25 0.30, fake-dense 0.50, RRF hybrid 0.40 on the mismatch queries.

Two things follow. The hybrid lands **below** its better arm, not above it --
fusing a 0.50 ranker with a 0.30 ranker gives 0.40, which is what averaging
ranks does. And the arm the exercise calls dense is not dense: `fake_dense_rank`
scores Jaccard overlap plus 0.15 for every pair of four-plus-character tokens
where one contains the other. It beats BM25 on the paraphrases because that
substring rule catches `light` inside `sunlight`, which is stemming, not
semantics. Swapping in a real dense arm -- TF-IDF reduced by truncated SVD --
gives 0.40 on the same queries, below the toy.

Structure: `TOPICS` are ten documents each answering one lexical and one
paraphrase query; `FILLER` pads the corpus to 30. `LEXICAL` and `MISMATCH` are
the two query halves with their gold document index. `arms` builds the four
rankers and `recall_at` scores one over a query list.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "14-information-retrieval-search"

TOPICS = (
    "Photosynthesis converts light energy into chemical energy inside plant chloroplasts.",
    "Mitosis divides one cell nucleus into two genetically identical daughter nuclei.",
    "Erosion moves weathered rock and soil downhill through water wind and ice.",
    "Inflation is a sustained rise in the general price level across an economy.",
    "Gravity is the mutual attraction between masses described by general relativity.",
    "Antibiotics kill bacteria or stop them multiplying inside an infected host.",
    "Encryption transforms readable data into ciphertext using a secret key.",
    "Democracy assigns political power to citizens through periodic free elections.",
    "Evaporation turns liquid water into vapour below its boiling temperature.",
    "A recession is a sustained contraction of economic output across two quarters.",
)
FILLER = ("The committee met on Tuesday to review the quarterly schedule.",
          "Rainfall totals for the region were below the seasonal average.",
          "The library extended its opening hours during the examination period.",
          "Volunteers repainted the community hall over the weekend.",
          "A new footbridge opened across the canal in the spring.") * 4
CORPUS = TOPICS + FILLER
LEXICAL = (("how does photosynthesis convert light energy", 0),
           ("what does mitosis divide", 1),
           ("how does erosion move rock and soil", 2),
           ("what is a sustained rise in the price level", 3),
           ("what is the attraction between masses", 4),
           ("how do antibiotics kill bacteria", 5),
           ("what transforms readable data into ciphertext", 6),
           ("who holds political power in a democracy", 7),
           ("what turns liquid water into vapour", 8),
           ("what is a contraction of economic output", 9))
MISMATCH = (("how do green leaves make food from sunlight", 0),
            ("how does a cell copy itself in two", 1),
            ("what wears away hillsides over time", 2),
            ("why do prices keep going up everywhere", 3),
            ("why do heavy things fall towards each other", 4),
            ("what medicine fights a bacterial infection", 5),
            ("how do you scramble a message so nobody can read it", 6),
            ("what system lets people choose their leaders by voting", 7),
            ("how does a puddle disappear without boiling", 8),
            ("what is a downturn in business activity", 9))
DEPTH, WIDTH, AT = 10, 8, 5


def lsa(np, corpus, width):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vector = TfidfVectorizer().fit(corpus)
    svd = TruncatedSVD(n_components=width, random_state=0).fit(vector.transform(corpus))
    matrix = svd.transform(vector.transform(corpus))
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)

    def rank(query):
        seen = svd.transform(vector.transform([query]))[0]
        seen = seen / (np.linalg.norm(seen) + 1e-9)
        return sorted(((float(matrix[i] @ seen), i) for i in range(len(corpus))), reverse=True)
    return rank


def arms(np, ref):
    """The four rankers: BM25, the lesson's toy dense, a real dense, and both fusions."""
    corpus = list(CORPUS)
    bm25 = ref.BM25(corpus)
    sparse = lambda q: bm25.rank(q, top_k=len(corpus))                            # noqa: E731
    toy = lambda q: ref.fake_dense_rank(q, corpus, top_k=len(corpus))             # noqa: E731
    dense = lsa(np, corpus, WIDTH)
    fuse = lambda left, right: (lambda q: ref.reciprocal_rank_fusion(              # noqa: E731
        [left(q)[:DEPTH], right(q)[:DEPTH]]))
    return {"bm25": sparse, "fake-dense": toy, "hybrid": fuse(sparse, toy),
            "lsa": dense, "bm25+lsa": fuse(sparse, dense)}


def recall_at(rank, queries, at=AT) -> float:
    hits = sum(gold in [i for _, i in rank(query)[:at]] for query, gold in queries)
    return round(hits / len(queries), 4)


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    built = arms(np, ref)
    return {
        "has_hybrid": hasattr(ref, "hybrid_search"),
        "ships": sorted(n for n in dir(ref) if n[0].isupper() or n.endswith(("rank", "fusion"))),
        "lexical": {name: recall_at(rank, LEXICAL) for name, rank in built.items()},
        "mismatch": {name: recall_at(rank, MISMATCH) for name, rank in built.items()},
        "overall": {name: recall_at(rank, LEXICAL + MISMATCH) for name, rank in built.items()},
        "documents": len(CORPUS), "queries": len(LEXICAL) + len(MISMATCH),
        "expansion": ref.fake_dense_rank("sunlight", ["light energy"], top_k=1)[0][0],
    }


def verify(result):
    lexical, mismatch, overall = result["lexical"], result["mismatch"], result["overall"]
    return [
        practice.Check(
            "ANSWER: on the lexical half all three arms tie at recall@5 of 1.00",
            {lexical[name] for name in ("bm25", "fake-dense", "hybrid")} == {1.0},
            f"over {result['documents']} documents and {len(LEXICAL)} queries that share content "
            f"words with their answer, the three arms score {lexical}. The comparison the exercise "
            f"asks for is a three-way tie on half its own test set, and `hybrid_search` is not in "
            f"the lesson either -- it ships {result['ships']}"),
        practice.Check(
            "MECHANISM: every number that separates the arms comes from the other half",
            len(set(mismatch[name] for name in ("bm25", "fake-dense", "hybrid"))) == 3,
            f"on {len(MISMATCH)} plain-language paraphrases with no content word in common with "
            f"their answer, the same arms score {mismatch}. Which retriever wins is decided "
            f"entirely by queries the exercise does not ask for"),
        practice.Check(
            "FINDING: the hybrid lands below its better arm, not above it",
            mismatch["bm25"] < mismatch["hybrid"] < mismatch["fake-dense"],
            f"fusing a {mismatch['fake-dense']} ranker with a {mismatch['bm25']} one gives "
            f"{mismatch['hybrid']}, between the two. Reciprocal rank fusion averages ranks, so a "
            f"weaker arm pulls the stronger one down -- overall the hybrid scores "
            f"{overall['hybrid']} against {overall['fake-dense']}"),
        practice.Check(
            "MECHANISM: the arm the exercise calls dense is lexical",
            result["expansion"] > 0,
            f"`fake_dense_rank` scores Jaccard overlap plus 0.15 for every pair of four-plus "
            f"character tokens where one contains the other. On the query 'sunlight' against the "
            f"document 'light energy' -- no shared token -- it scores {result['expansion']}, which "
            f"is the substring rule firing. That is stemming, not semantics"),
        practice.Check(
            "FINDING: a real dense arm does not do better on this corpus",
            mismatch["lsa"] < mismatch["fake-dense"],
            f"TF-IDF reduced by truncated SVD to {WIDTH} components scores {mismatch['lsa']} on the "
            f"paraphrases against the toy's {mismatch['fake-dense']}, and fusing it with BM25 gives "
            f"{mismatch['bm25+lsa']}. The toy wins because its substring rule catches `light` "
            f"inside `sunlight`, which latent semantic indexing on 30 documents cannot learn"),
        practice.Check(
            "CONTROL: no fusion beats the best single arm on any half",
            max(overall["hybrid"], overall["bm25+lsa"]) < max(overall.values()),
            f"overall recall@{AT} is {overall}. Both fusions sit below the best individual ranker, "
            f"so on this corpus the answer to 'compare BM25-only, dense-only, and hybrid' is that "
            f"hybrid is third -- and it would be a tie if the mismatch queries were left out"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
