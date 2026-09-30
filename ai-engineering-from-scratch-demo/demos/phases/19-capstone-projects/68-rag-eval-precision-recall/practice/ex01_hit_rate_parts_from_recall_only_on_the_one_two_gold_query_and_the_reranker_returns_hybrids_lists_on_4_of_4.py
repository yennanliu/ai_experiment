"""Exercise 1 -- hit-rate@k and recall@k part only on the one two-gold query, and the reranker changes no list.

    Add a fifth retrieval metric: hit-rate@k. Compare it against recall@k. Explain when they differ.

Reading of the exercise: hit-rate@k is 1 for a query when at least one gold
doc is in the top k and 0 otherwise, averaged over queries. It is added
beside the lesson's `recall_at_k` and both are run on the lesson's own 4
`QRELS` with its three pipelines at k = 1, 3 and 5, as `main()` does. When
they differ is then checked exhaustively: every ranking of 5 docs, every gold
set of 1 to 3 of them, every k from 1 to 5 (15,000 cases).

**ANSWER: hit-rate@k and recall@k differ only at k = 1, and only on q1.**

| pipeline | hit-rate@1 | recall@1 | @3 and @5 |
|---|---:|---:|---:|
| baseline | 0.5 | 0.375 | 1.0 / 1.0 |
| hybrid | 1.0 | 0.875 | 1.0 / 1.0 |
| hybrid+rerank | 1.0 | 0.875 | 1.0 / 1.0 |

q1 is the one qrel with two gold docs (d3, d1). Its top 1 holds one of them,
so it is a hit (1) but half the recall (0.5). The other three qrels have one
gold doc each, and for them the two metrics are the same number.

**FINDING: hit-rate is never below recall, and equals it whenever there is
one gold doc.** Of the 15,000 cases, 9,000 are equal and 6,000 have hit-rate
above recall, none of them with a single gold doc. They differ exactly when
the top k holds some but not all of two or more gold docs. Hit-rate asks
"did the generator see anything useful"; recall asks "did it see everything".

**FINDING: the rerank row cannot beat hybrid on MRR, as the doc says it
does.** `hybrid_plus_rerank_pipeline` returns hybrid's exact list on 4 of 4
queries, so both have MRR 1.0 (baseline 0.75). "Hybrid beats baseline on
recall" holds only at k = 1 (0.875 against 0.375); at k = 3 and 5 all three
rows are 1.0. At k = 5 every pipeline returns only 3, 2, 2 and 2 docs.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "68-rag-eval-precision-recall"
KS = (1, 3, 5)
NAMES = ("baseline", "hybrid", "hybrid+rerank")


def hit_rate_at_k(retrieved, gold, k):
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def pipelines(ref):
    return {"baseline": ref.baseline_pipeline, "hybrid": ref.hybrid_pipeline,
            "hybrid+rerank": ref.hybrid_plus_rerank_pipeline}


def table(ref):
    """Per pipeline and k: (mean hit-rate, mean recall, queries where they differ)."""
    out = {}
    for name, fn in pipelines(ref).items():
        runs = [(fn(q.query, max(KS))[0], set(q.gold_doc_ids)) for q in ref.QRELS]
        for k in KS:
            hit = [hit_rate_at_k(r, g, k) for r, g in runs]
            rec = [ref.recall_at_k(r, g, k) for r, g in runs]
            diff = [q.qid for q, h, c in zip(ref.QRELS, hit, rec) if h != c]
            out[f"{name}@{k}"] = (sum(hit) / len(hit), sum(rec) / len(rec), diff)
    return out


def lesson_rows(ref):
    """The lesson's evaluator on its own pipelines: recall@1,3,5, MRR, docs returned."""
    evals = {n: ref.evaluate_pipeline(fn, ref.QRELS) for n, fn in pipelines(ref).items()}
    return {n: ([round(e[f"recall@{k}"], 3) for k in KS], round(e["mrr"], 3), [len(r) for r in e["per_query_retrieved"]],
                e["per_query_retrieved"]) for n, e in evals.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = lesson_rows(ref)
    return {
        "table": table(ref),
        "gold_sizes": {q.qid: len(q.gold_doc_ids) for q in ref.QRELS},
        "rows": {n: v[:3] for n, v in rows.items()},
        "same_lists": sum(a == b for a, b in zip(rows["hybrid"][3], rows["hybrid+rerank"][3])),
        "exhaustive": exhaustive(ref),
    }


def exhaustive(ref):
    """Every ranking of 5 docs, every gold set of 1-3 of them, k = 1-5: hit vs recall."""
    docs, count = "abcde", {"equal": 0, "hit_above": 0, "hit_below": 0, "above_single_gold": 0}
    for rank in itertools.permutations(docs):
        for size in (1, 2, 3):
            for gold in itertools.combinations(docs, size):
                for k in range(1, 6):
                    h, c = hit_rate_at_k(list(rank), gold, k), ref.recall_at_k(list(rank), set(gold), k)
                    key = "equal" if h == c else "hit_above" if h > c else "hit_below"
                    count[key] += 1
                    count["above_single_gold"] += h > c and size == 1
    return count


def verify(result):
    r, t = result, result["table"]
    at = {k: {n: t[f"{n}@{k}"][:2] for n in NAMES} for k in KS}
    differ = sorted({q for v in t.values() for q in v[2]})
    return [
        practice.Check(
            "ANSWER: hit-rate@k parts from recall@k only on the one query with two gold docs",
            (at[1], at[3], at[5], differ, r["gold_sizes"])
            == ({"baseline": (0.5, 0.375), "hybrid": (1.0, 0.875), "hybrid+rerank": (1.0, 0.875)},
                *[{n: (1.0, 1.0) for n in NAMES}] * 2, ["q1"], {"q1": 2, "q2": 1, "q3": 1, "q4": 1}),
            f"(hit-rate, recall)@1 {at[1]}; @3 {at[3]}; @5 {at[5]}; they differ only on {differ}, gold sizes {r['gold_sizes']}",
        ),
        practice.Check(
            "FINDING: hit-rate is never below recall, and equals it whenever the query has one gold doc",
            r["exhaustive"] == {"equal": 9000, "hit_above": 6000, "hit_below": 0, "above_single_gold": 0},
            f"over 120 rankings x 25 gold sets x 5 k: {r['exhaustive']}",
        ),
        practice.Check(
            "FINDING: the rerank row cannot beat hybrid on MRR -- it returns hybrid's lists on 4 of 4 queries",
            (r["rows"], r["same_lists"]) == ({"baseline": ([0.375, 1.0, 1.0], 0.75, [3, 2, 2, 2]),
                                              "hybrid": ([0.875, 1.0, 1.0], 1.0, [3, 2, 2, 2]),
                                              "hybrid+rerank": ([0.875, 1.0, 1.0], 1.0, [3, 2, 2, 2])}, 4),
            f"(recall@1,3,5, MRR, docs returned at k = 5) {r['rows']}; identical top-5 lists on {r['same_lists']}/4",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
