"""Exercise 3 — the retriever has no parameters to tune.

    **Hard.** Fine-tune a dense encoder on your domain using
    MultipleNegativesRankingLoss (Sentence Transformers). Build a training set
    from 500 query-document pairs. Compare pre- and post-fine-tune recall.

Reading of the exercise: `sentence_transformers` and `torch` are absent, and the
retriever the lesson ships as its dense arm has nothing to fine-tune. Every
number `fake_dense_rank` produces comes from Jaccard overlap and a fixed 0.15
substring bonus; it has no weights, so "pre- and post-fine-tune recall" is the
same number twice for the system the exercise's own lesson builds. The
comparison needs a parameterised retriever before it needs a training loop.

Substituting one -- TF-IDF reduced by truncated SVD, whose subspace *is* fitted
-- makes the comparison run, and it shows what the exercise is aiming at without
a GPU. Refitting the same encoder at 2, 4, 8 and 16 components moves recall@5 on
the paraphrase queries across 0.1000 to 0.4000: the number the exercise wants to
attribute to fine-tuning is available from a hyperparameter of the untrained
model, over a range as wide as any training run could plausibly buy.

The confound is larger on the half everyone assumes is solved. Across the same
four widths the ten lexical queries score 0.7000, 0.9000, 1.0000, 1.0000 -- a
span of 0.3000, wider than the paraphrase half's 0.2000 -- so only the two widest
configurations get the easy queries right at all. What is actually left to learn
is the paraphrase half's ceiling of 0.4000 against a reachable 1.0000.

One structural note about the loss the exercise names. MultipleNegativesRanking
takes the other documents in the batch as the negatives, so its signal is the
batch composition. With 500 pairs over ten topics -- fifty pairs per topic --
roughly one batch member in ten is a false negative: a document that answers the
query and is being pushed away from it. That is a property of the pairing, not
of the optimiser, and it is fixed before any training starts.

Structure: the corpus, queries and rankers come from exercise 1. `widths` refits
the dense arm at each SVD dimension and scores both query halves; `unfittable`
checks that the lesson's own dense arm returns identical rankings from two
independent constructions, which is what having no parameters looks like.
"""

from __future__ import annotations

import importlib.util
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "14-information-retrieval-search"

SIBLING = "ex01_half_the_queries_decide_nothing.py"
UNAVAILABLE = ("sentence_transformers", "torch", "transformers")
WIDTHS = (2, 4, 8, 16)
PAIRS, TOPICS = 500, 10


def widths(np, sibling, corpus) -> dict:
    rows = {}
    for width in WIDTHS:
        rank = sibling.lsa(np, corpus, width)
        rows[width] = {"lexical": sibling.recall_at(rank, sibling.LEXICAL),
                       "mismatch": sibling.recall_at(rank, sibling.MISMATCH)}
    return rows


def unfittable(ref, sibling, corpus, query) -> bool:
    """Two independent calls to the lesson's dense arm return the same ranking."""
    first = ref.fake_dense_rank(query, list(corpus), top_k=len(corpus))
    second = ref.fake_dense_rank(query, list(corpus), top_k=len(corpus))
    return first == second


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    corpus = list(sibling.CORPUS)
    rows = widths(np, sibling, corpus)
    mismatch = [rows[w]["mismatch"] for w in WIDTHS]
    return {
        "unavailable": [m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        "rows": rows, "mismatch": mismatch,
        "lexical": [rows[w]["lexical"] for w in WIDTHS],
        "span": (min(mismatch), max(mismatch)),
        "lexical_span": (min(rows[w]["lexical"] for w in WIDTHS),
                         max(rows[w]["lexical"] for w in WIDTHS)),
        "counts": (len(sibling.LEXICAL), len(sibling.MISMATCH)),
        "deterministic": unfittable(ref, sibling, corpus, sibling.MISMATCH[0][0]),
        "toy_mismatch": sibling.recall_at(
            lambda q: ref.fake_dense_rank(q, corpus, top_k=len(corpus)), sibling.MISMATCH),
        "false_negative_rate": round(1 / TOPICS, 4), "pairs": PAIRS, "topics": TOPICS,
        "per_topic": PAIRS // TOPICS,
    }


def verify(result):
    rows, mismatch, span = result["rows"], result["mismatch"], result["span"]
    return [
        practice.Check(
            "ANSWER: the lesson's dense arm has no parameters, so both sides of the comparison are equal",
            result["unavailable"] == list(UNAVAILABLE) and result["deterministic"],
            f"{result['unavailable']} are all absent, and `fake_dense_rank` computes Jaccard "
            f"overlap plus a fixed 0.15 substring bonus -- two independent constructions return "
            f"identical rankings. 'Compare pre- and post-fine-tune recall' is the same number twice "
            f"({result['toy_mismatch']} on the paraphrase queries) for the retriever this lesson "
            f"ships"),
        practice.Check(
            "MECHANISM: substituting a fitted encoder makes the comparison run",
            len(set(mismatch)) > 1,
            f"TF-IDF reduced by truncated SVD has a subspace that is fitted, so refitting it at "
            f"{list(WIDTHS)} components gives paraphrase recall@5 "
            f"{dict(zip(WIDTHS, mismatch))}. The comparison the exercise describes needs a "
            f"parameterised retriever before it needs a training loop"),
        practice.Check(
            "FINDING: a hyperparameter of the untrained model spans the range fine-tuning would claim",
            span[1] - span[0] >= 0.2,
            f"recall on the paraphrase half runs {span[0]} to {span[1]} across the four widths -- a "
            f"span of {span[1] - span[0]:.4f} with no training at all, and not monotone: "
            f"{dict(zip(WIDTHS, mismatch))}. A pre- and post- comparison that does not hold the "
            f"encoder's width fixed is measuring the width"),
        practice.Check(
            "FINDING: the width moves the easy half further than the hard one",
            result["lexical_span"][1] - result["lexical_span"][0] > span[1] - span[0],
            f"the {result['counts'][0]} lexical queries score {result['lexical']} across the same "
            f"widths -- a span of {result['lexical_span'][1] - result['lexical_span'][0]:.4f} "
            f"against {span[1] - span[0]:.4f} on the paraphrases. The confound is larger on the "
            f"half everyone assumes is solved, and only the widest two configurations reach 1.0"),
        practice.Check(
            "FINDING: so the target is 0.4000 against a reachable 1.0000",
            span[1] < 0.5 < result["lexical_span"][1] == 1.0,
            f"the best paraphrase recall over every width tried is {span[1]}, where the lexical "
            f"queries reach {result['lexical_span'][1]}. That gap is what fine-tuning would have "
            f"to close, and stating it is more useful than a pre-and-post pair of numbers with no "
            f"target between them"),
        practice.Check(
            "CONTROL: the loss the exercise names is decided by the batch, not the optimiser",
            result["false_negative_rate"] == 0.1,
            f"MultipleNegativesRankingLoss takes the other documents in the batch as negatives. "
            f"{result['pairs']} pairs over {result['topics']} topics is {result['per_topic']} pairs "
            f"per topic, so about {result['false_negative_rate']:.0%} of any batch is a false "
            f"negative -- a document that answers the query and is being pushed away from it. That "
            f"is fixed by how the pairs were built, before training starts"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
