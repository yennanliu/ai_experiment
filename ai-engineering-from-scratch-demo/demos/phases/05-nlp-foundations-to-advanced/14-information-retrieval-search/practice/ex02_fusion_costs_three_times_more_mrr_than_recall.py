"""Exercise 2 — fusion costs three times more MRR than recall.

    **Medium.** Add MRR calculation. For each test query with a known correct
    document, find the rank of the correct doc in BM25, dense, and hybrid
    rankings. Report the MRR for each.

Reading of the exercise: adding MRR does not add a second opinion, it adds a
second magnitude, and on the fusion the two magnitudes are very different. On
exercise 1's mismatch queries the hybrid loses 0.1000 of recall@5 against the
better of its two arms and 0.3016 of MRR -- three times as much. Recall@5 asks
only whether the answer is somewhere in the top five; MRR asks where, and
reciprocal rank fusion moves the answer down inside the window without moving it
out.

The full table, on the ten paraphrase queries:

    BM25 recall 0.30 / MRR 0.2891, fake-dense 0.50 / 0.4649,
    hybrid 0.40 / 0.1633, real dense 0.40 / 0.3033, BM25+dense 0.30 / 0.2800.

The hybrid is second on recall and last on MRR, below both of the arms it is
made of. That is not a contradiction: a document ranked 4th by one arm and 9th
by the other fuses to something near 6th, which is inside recall@5's window on
one arm's ranking and outside it after fusion, while MRR charges the full
distance either way. And the direction is not fixed -- fusing BM25 with the real
dense arm costs 0.1000 of recall and only 0.0233 of MRR, the reverse ratio. Where
fusion pushes the answer out of the window it costs recall; where it pushes the
answer down inside the window it costs MRR, and neither number predicts the
other.

On the lexical half every arm scores MRR 1.0000 exactly -- the correct document
is first for all five rankers on all ten queries. So MRR, like recall, is
constant across the arms on half the query set, and the exercise's instruction
to "report the MRR for each" produces one informative column and one column of
ones.

Structure: the corpus, the query halves and the rankers come from exercise 1 via
`practice.load_module`. `mrr` is the mean reciprocal rank over a query list, with
0 for a document that never appears; `cost` compares each fusion against the
better of its two inputs on both metrics.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "14-information-retrieval-search"

SIBLING = "ex01_half_the_queries_decide_nothing.py"
PAIRS = {"hybrid": ("bm25", "fake-dense"), "bm25+lsa": ("bm25", "lsa")}


def mrr(rank, queries) -> float:
    total = 0.0
    for query, gold in queries:
        order = [index for _, index in rank(query)]
        total += 1 / (order.index(gold) + 1) if gold in order else 0.0
    return round(total / len(queries), 4)


def table(sibling, built, queries) -> dict:
    return {"mrr": {name: mrr(rank, queries) for name, rank in built.items()},
            "recall": {name: sibling.recall_at(rank, queries) for name, rank in built.items()}}


def cost(scores, fused, parts) -> float:
    return round(max(scores[name] for name in parts) - scores[fused], 4)


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    built = sibling.arms(np, ref)
    halves = {"lexical": sibling.LEXICAL, "mismatch": sibling.MISMATCH}
    scores = {half: table(sibling, built, queries) for half, queries in halves.items()}
    return {
        "scores": scores, "queries": {half: len(q) for half, q in halves.items()},
        "cost": {metric: {fused: cost(scores["mismatch"][metric], fused, parts)
                          for fused, parts in PAIRS.items()}
                 for metric in ("recall", "mrr")},
        "ranked": {metric: sorted(scores["mismatch"][metric],
                                  key=lambda n: -scores["mismatch"][metric][n])
                   for metric in ("recall", "mrr")},
        "at": sibling.AT,
    }


def verify(result):
    lexical, mismatch = result["scores"]["lexical"], result["scores"]["mismatch"]
    loss, ranked = result["cost"], result["ranked"]
    return [
        practice.Check(
            "ANSWER: the fusion loses 0.1000 of recall and 0.3016 of MRR against its better arm",
            loss["mrr"]["hybrid"] > 2 * loss["recall"]["hybrid"] > 0,
            f"on the {result['queries']['mismatch']} paraphrase queries the hybrid gives up "
            f"{loss['recall']['hybrid']} of recall@{result['at']} and {loss['mrr']['hybrid']} of "
            f"MRR against the better of the two arms it is made of -- "
            f"{loss['mrr']['hybrid'] / loss['recall']['hybrid']:.1f} times as much. Adding MRR adds "
            f"a magnitude, not a second opinion"),
        practice.Check(
            "MECHANISM: recall asks whether the answer is in the window, MRR asks where",
            mismatch["recall"]["hybrid"] > mismatch["recall"]["bm25"]
            and mismatch["mrr"]["hybrid"] < mismatch["mrr"]["bm25"],
            f"recall@{result['at']} on the mismatch half reads {mismatch['recall']} and MRR reads "
            f"{mismatch['mrr']}. The hybrid is above BM25 on one and below it on the other: fusion "
            f"moves the answer down inside the window without always moving it out"),
        practice.Check(
            "FINDING: the hybrid ranks second on recall and last on MRR",
            ranked["recall"].index("hybrid") < ranked["mrr"].index("hybrid")
            and ranked["mrr"][-1] == "hybrid",
            f"by recall the order is {ranked['recall']} and by MRR it is {ranked['mrr']}. The two "
            f"metrics the exercise asks for do not agree on where the fusion belongs, and the one "
            f"it adds second is the one that ranks it worst"),
        practice.Check(
            "FINDING: on the lexical half MRR is 1.0000 for every arm",
            set(lexical["mrr"].values()) == {1.0},
            f"all five rankers put the correct document first on all "
            f"{result['queries']['lexical']} lexical queries: {lexical['mrr']}. 'Report the MRR for "
            f"each' produces one informative column and one column of ones, exactly as recall did"),
        practice.Check(
            "FINDING: the second fusion pays the other way round, so neither loss predicts the other",
            loss["mrr"]["bm25+lsa"] < loss["recall"]["bm25+lsa"],
            f"fusing BM25 with the real dense arm costs {loss['recall']['bm25+lsa']} of recall and "
            f"only {loss['mrr']['bm25+lsa']} of MRR -- the reverse of the hybrid's "
            f"{loss['recall']['hybrid']} and {loss['mrr']['hybrid']}. Where fusion demotes the "
            f"answer out of the window it costs recall; where it demotes the answer inside the "
            f"window it costs MRR. Reporting one number cannot stand in for the other"),
        practice.Check(
            "CONTROL: no arm is best on both metrics except the one that is best on both halves",
            ranked["recall"][0] == ranked["mrr"][0],
            f"the toy dense arm tops both mismatch columns ({ranked['recall'][0]}), so the "
            f"disagreement between recall and MRR is about the fusions rather than about the "
            f"individual rankers. Adding MRR changes the verdict on hybrid retrieval and on nothing "
            f"else"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
