"""Exercise 1 — lesson 66's reranker puts "Cancelling jobs" first on all 3 queries and drops gold from rank 1 to 6.

    Implement RAG-Fusion (a 2024 variant of multi-query) where the rewriter's paraphrases are intentionally diverse, then the rerank step (lesson 66) picks the final list.

Reading of the exercise: "intentionally diverse" is made mechanical with the
lesson's own advice, "detect duplicates by Jaccard". The candidate pool is
what the lesson's `MockLLM` offers for a query: its three paraphrases plus
its decomposition. A candidate is kept only if its token Jaccard with the
query and with every kept rewrite is below 0.5, up to four rewrites. Each
rewrite goes through the lesson's `HybridRetriever`, the lists are merged
with the lesson's `rrf`, and lesson 66's `CrossEncoder` (trained with its
own `train_tiny` on its own `TRAIN_TRIPLES`, seed 19660101) reranks the fused
top 8. Two rerank variants are scored: by the user's query, as lesson 66
does, and by the best score over all the fused queries. The fixture is the
lesson's three `GOLD` queries, scored by the gold document's rank.

**ANSWER: `rag_fusion` below. The diversity filter works, and the lesson 66
rerank then makes the final list worse.** The filter removes all three of
the lesson's paraphrases for the multi-topic query. They have pairwise
Jaccard 0.60 to 1.00, and two of them are the same string. The two
sub-questions replace them. Before the rerank, the gold rank on the three
queries is 1 / 2 / 1, the same as the lesson's multi-query. After a rerank by
the user's query it is 6 / 2 / 6.

**FINDING: lesson 66's reranker memorised its 14 training triples, wording
included.** Both lessons' corpora cover the same eight topics under the same
ids, but lesson 67 rewords every document. Given lesson 66's wording, the
reranker puts d1 first for its own training query "how do we abort a
multipart upload". Given lesson 67's wording, it puts d8, "Cancelling jobs",
first for that same query and for all three fixture queries. Reranking by
the best score over the fused queries gives 6 / 1 / 4, which does not fix
it.

**FINDING: the lesson's paraphrase fallback collapses on a real query.** For
the multi-topic query, the third paraphrase repeats the first, and both say
"an transfer". This is the "rewrites all converge" failure the lesson warns
about, and it happens in the lesson's own demo.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "67-query-rewriting-hyde"
CAP, MAX_REWRITES, TOP = 0.5, 4, 8


def jaccard(ref, a, b):
    x, y = set(ref.tokenize(a)), set(ref.tokenize(b))
    return len(x & y) / len(x | y)


def diverse_rewrites(ref, llm, query):
    kept = [query]
    for cand in llm.paraphrase(query, 3) + llm.decompose(query):
        if len(kept) <= MAX_REWRITES and all(jaccard(ref, cand, k) < CAP for k in kept):
            kept.append(cand)
    return kept


def rerank_ids(r66, model, queries, docs):
    cands = [r66.Candidate(d.doc_id, d.field_text()) for d in docs]
    best = {}
    for q in queries:
        for cand, score in r66.rerank(model, q, cands, top_k=len(cands)):
            best[cand.doc_id] = max(best.get(cand.doc_id, float("-inf")), score)
    return sorted(best, key=lambda k: -best[k])


def rag_fusion(ref, r66, model, retriever, llm, query):
    queries = diverse_rewrites(ref, llm, query)
    fused = [d for d, _ in ref.rrf([retriever.search(q, k_each=TOP, k_out=TOP) for q in queries])[:TOP]]
    return {
        "queries": queries,
        "fused": [d.doc_id for d in fused],
        "by_query": rerank_ids(r66, model, [query], fused),
        "by_max": rerank_ids(r66, model, queries, fused),
    }


def pairwise(ref, texts):
    return [round(jaccard(ref, a, b), 2) for i, a in enumerate(texts) for b in texts[i + 1 :]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    r66 = parity.load_reference(PHASE, "66-reranker-cross-encoder", "main")
    r66._set_seed()
    model = r66.CrossEncoder()
    r66.train_tiny(model, r66.TRAIN_TRIPLES, epochs=60)
    retriever, llm = ref.build_retriever(), ref.MockLLM()
    mq = ref.MultiQueryRewriter(llm=llm, n=3)
    rows = []
    for query, gold, _ in ref.GOLD:
        rf = rag_fusion(ref, r66, model, retriever, llm, query)
        lesson = [d.doc_id for d, _ in ref.retrieve_with_rewriter(query, mq, retriever, TOP, TOP)["results"]]
        ranks = {k: rf[k].index(gold) + 1 for k in ("fused", "by_query", "by_max")}
        rows.append({"lesson_mq": lesson.index(gold) + 1, **ranks, "n_rewrites": len(rf["queries"]) - 1,
                     "top1": rf["by_query"][0], "lesson_para": [query] + llm.paraphrase(query, 3)})
    q66 = ["how do we abort a multipart upload"]
    own = [ref.Doc(c.doc_id, "", c.text) for c in r66.CORPUS]
    train = (rerank_ids(r66, model, q66, own)[0], rerank_ids(r66, model, q66, ref.CORPUS)[0])
    return {"rows": rows, "train_top1": train, "pairwise": pairwise(ref, rows[2]["lesson_para"])}


def verify(result):
    rows = result["rows"]
    col = {k: [r[k] for r in rows] for k in ("lesson_mq", "fused", "by_query", "by_max", "n_rewrites", "top1")}
    para3 = rows[2]["lesson_para"]
    return [
        practice.Check(
            "ANSWER: diverse rewrites fused by RRF match the lesson's multi-query; the lesson 66 rerank degrades it",
            (col["lesson_mq"], col["fused"], col["by_query"], col["n_rewrites"]) == ([1, 2, 1], [1, 2, 1], [6, 2, 6], [3, 3, 2]),
            f"gold rank: lesson multi-query {col['lesson_mq']}, RAG-Fusion before rerank {col['fused']}, "
            f"after rerank by query {col['by_query']}, by max over queries {col['by_max']}; rewrites kept {col['n_rewrites']}",
        ),
        practice.Check(
            "FINDING: lesson 66's reranker memorised its training triples and ranks d8 first on every fixture query",
            (result["train_top1"], col["top1"], col["by_max"]) == (("d1", "d8"), ["d8", "d8", "d8"], [6, 1, 4]),
            f"training query top-1 on lesson 66 / lesson 67 wording {result['train_top1']}; fixture top-1 {col['top1']}; max-rerank gold ranks {col['by_max']}",
        ),
        practice.Check(
            "FINDING: the lesson's paraphrase fallback converges on the multi-topic query",
            para3[1] == para3[3] and "an transfer" in para3[1] and result["pairwise"] == [0.71, 0.85, 0.71, 0.6, 1.0, 0.6],
            f"paraphrases {para3[1:]}; pairwise Jaccard {result['pairwise']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
