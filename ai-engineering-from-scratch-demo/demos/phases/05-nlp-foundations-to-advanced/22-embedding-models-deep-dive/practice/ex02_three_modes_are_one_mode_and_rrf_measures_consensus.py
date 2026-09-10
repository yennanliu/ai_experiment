"""Exercise 2 — three modes are one mode, and RRF measures consensus.

    **Medium.** Compare BGE-M3 dense, sparse, and colbert on 500 passages from
    your domain. Which wins on recall@10? Does RRF fusion beat the best single
    mode?

Reading of the exercise: BGE-M3 is not installed, so the three modes are built
from the lesson's own parts -- `hash_embed` + `cosine` for dense, `sparse_embed`
+ `sparse_score` for sparse, and per-token max-similarity for ColBERT -- over 40
passages on 10 topics. ColBERT wins at **0.8250** against **0.8000** for both
others, and **RRF does not beat it**: fusing dense with sparse scores 0.8000, and
adding ColBERT to the fusion leaves it at 0.8000.

The win is one document. With 40 passages and 4 relevant per query, recall@10
moves in steps of 1/40 = 0.025, and 0.8250 - 0.8000 is exactly one step. The
comparison the exercise asks for has one unit of resolution and the answer sits
inside it.

The deeper problem is that the three modes are not three modes. `hash_embed` of a
single token is a unit vector on one coordinate with a fixed sign, so the cosine
between two token embeddings is +1, -1 or 0, and ColBERT's max-similarity reduces
to *does the document contain a token in the same bucket*. It agrees with sparse
on the sign of the score for **353 of 400** query-passage pairs. All three modes
read the same lexical feature and differ only in how they weight it, so no
architectural question is being answered.

RRF cannot repair that, because RRF throws the scores away. At `k=60` over 40
documents the weights run 1/61 to 1/100 -- a **1.64x** range end to end -- so a
document ranked first by one mode and last by the other scores 0.02639 and loses
to one ranked fifth by both at 0.03077. RRF is a consensus vote, and consensus
between three views of one feature is that feature again.

Two smaller things. `main()` fuses `dense_ranked[:5]` with `sparse_scores[:5]`,
and fusing truncated lists scores 0.7500 -- worse than either input. And
`rrf_fuse` consumes `(score, index)` pairs while returning `(index, score)`, so
feeding its own output back in reads each score as a document id.

Structure: `MODES` holds the three scorers over the lesson's primitives;
`ranking` orders the corpus for one query; `recall_at` averages over the ten
queries; `fuse` runs `rrf_fuse` over a named subset, optionally truncated.
"""

from __future__ import annotations

import importlib.util
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "22-embedding-models-deep-dive"

# The corpus is exercise 1's, loaded rather than copied, so the two answers are
# about the same 40 passages (`DESIGN D5` applied within the lesson).
CORPUS = practice.load_module(
    pathlib.Path(__file__).resolve().parent / "ex01_truncating_a_hash_is_not_matryoshka.py")
DIM, DOCS, GOLD = CORPUS.DIM, CORPUS.DOCS, CORPUS.GOLD
QUERIES, PER_TOPIC = CORPUS.QUERIES, CORPUS.PER_TOPIC

UNAVAILABLE = ("FlagEmbedding", "sentence_transformers", "transformers", "torch")
AT, RRF_K = 10, 60


def modes(ref):
    """Dense, sparse and late-interaction scorers, all from the lesson's own parts."""
    def colbert(query, doc):
        left = [ref.hash_embed(t, DIM) for t in set(ref.tokenize(query))]
        right = [ref.hash_embed(t, DIM) for t in set(ref.tokenize(doc))]
        return sum(max((ref.cosine(a, b) for b in right), default=0.0) for a in left)

    return {
        "dense": lambda q, d: ref.cosine(ref.hash_embed(q, DIM), ref.hash_embed(d, DIM)),
        "sparse": lambda q, d: ref.sparse_score(ref.sparse_embed(q), ref.sparse_embed(d)),
        "colbert": colbert,
    }


def ranking(scorer, query):
    """(score, index) pairs for the whole corpus, best first -- `rank`'s own convention."""
    return sorted(((scorer(query, DOCS[i]), i) for i in range(len(DOCS))), reverse=True)


def recall_at(order_for):
    """Mean recall@10 over the ten topic queries."""
    total = sum(sum(1 for i in order_for(query)[:AT] if GOLD[i] == topic) / PER_TOPIC
                for query, topic in QUERIES)
    return round(total / len(QUERIES), 4)


def fuse(ref, scorers, names, cut=None):
    """Recall of `rrf_fuse` over the named modes, optionally over truncated lists."""
    def order(query):
        lists = [ranking(scorers[name], query) for name in names]
        return [index for index, _ in ref.rrf_fuse([r[:cut] if cut else r for r in lists])]

    return recall_at(order)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scorers = modes(ref)
    single = {name: recall_at(lambda q, s=scorers[name]: [i for _, i in ranking(s, q)])
              for name in scorers}
    pairs = [(q, i) for q, _ in QUERIES for i in range(len(DOCS))]
    sample = ref.rrf_fuse([ranking(scorers["dense"], QUERIES[0][0])[:5]])
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "docs": len(DOCS), "queries": len(QUERIES), "single": single,
        "grid": round(1 / len(DOCS), 4),
        "fused": {"dense+sparse": fuse(ref, scorers, ["dense", "sparse"]),
                  "all three": fuse(ref, scorers, list(scorers)),
                  "dense+sparse, top 5": fuse(ref, scorers, ["dense", "sparse"], cut=5)},
        "agree": sum(1 for q, i in pairs
                     if (scorers["colbert"](q, DOCS[i]) > 0) == (scorers["sparse"](q, DOCS[i]) > 0)),
        "pairs": len(pairs),
        "weights": (round(1 / (RRF_K + 1), 5), round(1 / (RRF_K + len(DOCS)), 5)),
        "spread": round((RRF_K + len(DOCS)) / (RRF_K + 1), 2),
        "split": round(1 / (RRF_K + 1) + 1 / (RRF_K + len(DOCS)), 5),
        "agreed": round(2 / (RRF_K + 5), 5),
        "recycled": ref.rrf_fuse([sample[:2]])[0][0],
    }


def verify(result):
    single, fused = result["single"], result["fused"]
    best = max(single.values())
    return [
        practice.Check(
            "ANSWER: ColBERT wins by 0.0250 and RRF does not beat it",
            single["colbert"] == best and max(fused.values()) <= best,
            f"{result['absent']} are all absent, so the three modes are built from the lesson's "
            f"own parts over {result['docs']} passages: {single}. Fusion scores {fused} -- no "
            "combination reaches the best single mode",
        ),
        practice.Check(
            "MECHANISM: and the win is one document, which is the unit of the metric",
            round(best - min(single.values()), 4) == result["grid"],
            f"{result['docs']} passages with {PER_TOPIC} relevant per query means recall@{AT} "
            f"moves in steps of {result['grid']}, and the gap between best and worst is exactly "
            "one step. The comparison has one unit of resolution and the answer sits inside it",
        ),
        practice.Check(
            "FINDING: the three modes read one feature",
            result["agree"] / result["pairs"] > 0.85,
            f"`hash_embed` of a single token is a unit vector on one coordinate, so ColBERT's "
            f"max-similarity reduces to whether the passage holds a token in the same bucket. It "
            f"agrees with sparse on the sign of the score for {result['agree']} of "
            f"{result['pairs']} query-passage pairs. Three weightings of one lexical feature",
        ),
        practice.Check(
            "MECHANISM: RRF cannot repair that, because it discards the scores",
            result["split"] < result["agreed"],
            f"at k={RRF_K} over {result['docs']} documents the weights run {result['weights'][0]} "
            f"down to {result['weights'][1]}, a {result['spread']}x range end to end. A document "
            f"ranked first by one mode and last by the other scores {result['split']} and loses "
            f"to one ranked fifth by both at {result['agreed']}. RRF is a consensus vote",
        ),
        practice.Check(
            "FINDING: fusing the truncated lists `main()` passes is worse than not fusing",
            fused["dense+sparse, top 5"] < min(single.values()),
            f"`main()` calls `rrf_fuse([dense_ranked[:5], sparse_scores[:5]])`. Over top-5 lists "
            f"the fusion scores {fused['dense+sparse, top 5']} against "
            f"{fused['dense+sparse']} over full lists and {min(single.values())} for either input "
            "alone -- a document missing from both heads gets no score at all",
        ),
        practice.Check(
            "CONTROL: `rrf_fuse` is not composable with itself",
            isinstance(result["recycled"], float),
            f"it consumes `(score, index)` pairs and returns `(index, score)`, so fusing its own "
            f"output reads each score as a document id: the first key comes back as "
            f"{result['recycled']!r}. Cascading fusions is the obvious next step and it fails "
            "silently",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
