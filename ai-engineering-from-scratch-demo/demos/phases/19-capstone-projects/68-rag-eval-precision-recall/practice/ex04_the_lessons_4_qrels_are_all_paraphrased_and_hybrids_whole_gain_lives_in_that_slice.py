"""Exercise 4 -- the lesson's 4 qrels are all paraphrased, and hybrid's whole gain lives in that slice.

    Add a query-class slice ("literal", "paraphrased", "multi-topic"). Report per-slice metrics.

Reading of the exercise: the slice is a label per qrel. The lesson's 4
`QRELS` are measured first: each contains a query word that none of its gold
docs contains, so all 4 go in "paraphrased". Four "literal" queries (every
content word in the gold doc) and four "multi-topic" queries (two unrelated
corpus topics, one gold doc each) are added over the same 12-doc `CORPUS`.
Each slice is run through the lesson's own `evaluate_pipeline` with its three
pipelines at k = 1, 3, 5, so one metrics row per (slice, pipeline).

**ANSWER: per-slice metrics** (recall@1 / MRR / nDCG@3 / answer relevance):

| slice | baseline | hybrid | hybrid+rerank |
|---|---|---|---|
| literal | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 0.996 / 1.0 | 1.0 / 1.0 / 0.996 / 1.0 |
| paraphrased | 0.375 / 0.75 / 0.724 / 0.75 | 0.875 / 1.0 / 0.882 / 1.0 | same as hybrid |
| multi-topic | 0.5 / 1.0 / 0.99 / 1.0 | same | same |

recall@3 and faithfulness are 1.0 in every cell.

**FINDING: the lesson's qrels are one slice.** q1-q4 each have at least one
query word absent from their gold docs ("dropped", "central gate", "fuse",
"stop"). No literal or multi-topic query is in the lesson's fixture.

**FINDING: hybrid's gain over baseline is all in the paraphrased slice.**
Pooled over 12 queries, recall@1 is 0.625 for baseline and 0.792 for
hybrid, and the paraphrased slice accounts for all of the gap. On literal queries
the two tie on recall and MRR, and hybrid loses nDCG@3 (0.996 against 1.0).
On multi-topic they are identical.

**FINDING: multi-topic recall@1 is capped at 0.5.** Two gold docs and one
slot give 1/2 at best, and all three pipelines reach it. The reranker changes
the order of one list out of 12 (m3) and no metric in any slice.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "68-rag-eval-precision-recall"
NAMES = ("baseline", "hybrid", "hybrid+rerank")
METRICS = ("recall@1", "recall@3", "mrr", "ndcg@3", "faithfulness", "answer_relevance")
# (qid, query, gold doc ids, gold answer substring, graded relevance)
LITERAL = [
    ("l1", "AbortMultipartOnFail retry budget", ["d1"], "retry budget", {"d1": 3}),
    ("l2", "check_permission policy principal resource action", ["d4"], "check_permission", {"d4": 3, "d5": 1}),
    ("l3", "reciprocal rank fusion lexical semantic retrieval", ["d6"], "reciprocal rank fusion", {"d6": 3, "d12": 1}),
    ("l4", "cancellation signal releases the queue slot", ["d7"], "cancellation signal", {"d7": 3}),
]
MULTI_TOPIC = [  # two unrelated corpus topics in one question, one gold doc each
    ("m1", "abort threshold for uploads and the authorization check", ["d3", "d4"], "three failed parts",
     {"d3": 3, "d4": 3, "d1": 1}),
    ("m2", "cancel a long-running job and size the worker pool", ["d7", "d10"], "cancellation signal",
     {"d7": 3, "d10": 3}),
    ("m3", "rank fusion and memory per vector in the ANN index", ["d6", "d8"], "reciprocal rank fusion",
     {"d6": 3, "d8": 3}),
    ("m4", "policy engine TTL and stale record TTL", ["d5", "d9"], "TTL", {"d5": 3, "d9": 3}),
]


def slices(ref):
    """The lesson's 4 QRELS are the paraphrased slice; the other two are added."""
    return {"literal": [ref.Qrel(*q) for q in LITERAL], "paraphrased": list(ref.QRELS),
            "multi-topic": [ref.Qrel(*q) for q in MULTI_TOPIC]}


def missing_words(ref, qrel):
    """Query content words (the baseline's tokenizer) that no gold doc contains."""
    by_id = {d.doc_id: d for d in ref.CORPUS}
    gold = {t for g in qrel.gold_doc_ids for t in ref._baseline_tokens(by_id[g].text())}
    return sorted(set(ref._baseline_tokens(qrel.query)) - gold)


def metrics(ref, fn, qrels):
    res = ref.evaluate_pipeline(fn, qrels)
    return {m: round(res[m], 3) for m in METRICS}


def moved(before, after, qrels):
    """Queries whose top-5 list the second pipeline changes."""
    return [q.qid for q in qrels if before(q.query, 5)[0] != after(q.query, 5)[0]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pipes = {n: getattr(ref, f + "_pipeline") for n, f in zip(NAMES, ("baseline", "hybrid", "hybrid_plus_rerank"))}
    sl = slices(ref)
    every = sum(sl.values(), [])
    return {
        "table": {(s, n): metrics(ref, fn, qs) for s, qs in sl.items() for n, fn in pipes.items()},
        "pooled": {n: metrics(ref, fn, every) for n, fn in pipes.items()},
        "rerank_moves": moved(pipes["hybrid"], pipes["hybrid+rerank"], every),
        "missing": {q.qid: missing_words(ref, q) for q in every},
    }


def row(r, sl, name, keys=("recall@1", "mrr", "ndcg@3", "answer_relevance")):
    return tuple(r["table"][(sl, name)][m] for m in keys)


def summary(r):
    return {
        "got": {sl: [row(r, sl, n) for n in NAMES] for sl in ("literal", "paraphrased", "multi-topic")},
        "ndcg": [row(r, "literal", n, ("ndcg@3",))[0] for n in NAMES[:2]],
        "pooled": [r["pooled"][n]["recall@1"] for n in NAMES],
        "same": [r["table"][(sl, NAMES[1])] == r["table"][(sl, NAMES[2])] for sl in ("literal", "paraphrased", "multi-topic")],
        "r3_faith": {(v["recall@3"], v["faithfulness"]) for v in r["table"].values()},
    }


def verify(result):
    r, m, s = result, result["missing"], summary(result)
    got, ndcg = s["got"], s["ndcg"]
    return [
        practice.Check(
            "ANSWER: per slice (recall@1, MRR, nDCG@3, relevance) for baseline / hybrid / hybrid+rerank",
            (got, s["r3_faith"])
            == ({"literal": [(1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.996, 1.0), (1.0, 1.0, 0.996, 1.0)],
                 "paraphrased": [(0.375, 0.75, 0.724, 0.75), (0.875, 1.0, 0.882, 1.0), (0.875, 1.0, 0.882, 1.0)],
                 "multi-topic": [(0.5, 1.0, 0.99, 1.0)] * 3}, {(1.0, 1.0)}),
            f"{got}; recall@3 and faithfulness 1.0 in every cell",
        ),
        practice.Check(
            "FINDING: the lesson's 4 qrels are all one class -- each has a query word no gold doc contains",
            ([bool(m[q]) for q in ("q1", "q2", "q3", "q4")], [m[q] for q in ("l1", "l2", "l3", "l4")]) == ([True] * 4, [[]] * 4),
            f"missing words: {m}",
        ),
        practice.Check(
            "FINDING: hybrid's whole gain is in the paraphrased slice; on literal queries it loses nDCG",
            (ndcg, s["pooled"]) == ([1.0, 0.996], [0.625, 0.792, 0.792]),
            f"literal nDCG@3 baseline vs hybrid {ndcg}; pooled 12-query recall@1 {s['pooled']}",
        ),
        practice.Check(
            "FINDING: multi-topic recall@1 is capped at 0.5 and the reranker moves one list and no metric",
            (got["multi-topic"][0][0], r["rerank_moves"], s["same"]) == (0.5, ["m3"], [True] * 3),
            f"two gold docs, one slot: recall@1 {got['multi-topic'][0][0]} for all three; rerank reorders "
            f"{r['rerank_moves']} of 12 and every cell equals hybrid's",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
