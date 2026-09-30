"""Exercise 5 — a similarity confidence gate only lowers recall, because the rewrites that find gold score lower.

    Add a confidence score per rewrite. Drop rewrites below the threshold. Measure the impact on recall.

Reading of the exercise: a rewrite's confidence is how well it keeps the
user's intent. It is measured here as the cosine between the lesson's
`mock_embed` of the rewrite and of the query, so the original query always
scores 1.0 and is never dropped. The gate filters the rewrites of the
lesson's `MultiQueryRewriter` (n = 3) and `DecomposeRewriter`. If a
decomposition loses every sub-question, the gate falls back to the query.
Retrieval is the lesson's `retrieve_with_rewriter` with k = 8. Recall@3 is
measured over the lesson's three `GOLD` queries. Their relevant sets are the
gold document, plus d3 "Quota cooldown" for the multi-topic query, whose
second clause it answers: 4 (query, document) pairs in all. The thresholds
swept are 0, 0.3, 0.5, 0.7 and 0.9.

**ANSWER: raising the threshold never raises recall, and above 0.5 it
costs multi-query a quarter of it.** Recall@3 by threshold:

    threshold     0     0.3   0.5   0.7   0.9
    multi-query   1.00  1.00  1.00  0.75  0.75
    decompose     0.75  0.75  0.75  0.75  0.75

The decomposer's recall does not move. Its sub-questions only matter on the
multi-topic query, and there the gate keeps the original query's result:
d1 and d3 are already both in the top 3 from the question as asked.

**FINDING: similarity to the query scores the rewrites backwards.** Of the
nine paraphrases, the five that on their own put the gold document first
average 0.501 confidence; the four that do not average 0.658. The two
lowest-scoring rewrites that reach gold, "how is an in-flight multipart
upload aborted on persistent failure" (0.279) and "how is lexical and
semantic retrieval combined" (0.290), are the lowest and third-lowest of
all nine.
The duplicated "an transfer" paraphrase scores 0.907 and misses. A rewrite
helps by leaving the query's vocabulary, and that is what this score
penalises.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "67-query-rewriting-hyde"
THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9)
EXTRA_RELEVANT = {2: {"d3"}}
K = 3


def confidence(ref, query, rewrite):
    return ref.cosine(ref.mock_embed(query), ref.mock_embed(rewrite))


def gated(ref, rewriter, threshold):
    """Wrap a lesson rewriter so rewrites under `threshold` are dropped."""
    inner = rewriter.rewrite

    def rewrite(query):
        out = inner(query)
        kept = [r for r in out.rewrites if confidence(ref, query, r) >= threshold]
        return ref.RewriteResult(strategy=out.strategy, rewrites=kept or [query], hypothetical=out.hypothetical)

    wrapper = ref._IdentityRewriter()
    wrapper.rewrite = rewrite
    return wrapper


def recall(ref, retriever, rewriter):
    hit = total = 0
    for i, (query, gold, _) in enumerate(ref.GOLD):
        relevant = {gold} | EXTRA_RELEVANT.get(i, set())
        out = ref.retrieve_with_rewriter(query, rewriter, retriever, k_each=8, k_out=K)
        hit += len(relevant & {d.doc_id for d, _ in out["results"]})
        total += len(relevant)
    return hit / total


def paraphrase_scores(ref, retriever, llm):
    rows = []
    for query, gold, _ in ref.GOLD:
        for p in llm.paraphrase(query, 3):
            top1 = retriever.search(p, k_each=8, k_out=1)[0][0].doc_id
            rows.append((round(confidence(ref, query, p), 3), top1 == gold, p))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    retriever, llm = ref.build_retriever(), ref.MockLLM()
    curves = {
        name: [round(recall(ref, retriever, gated(ref, make(), t)), 2) for t in THRESHOLDS]
        for name, make in (("multiquery", lambda: ref.MultiQueryRewriter(llm=llm, n=3)),
                           ("decompose", lambda: ref.DecomposeRewriter(llm=llm)))
    }
    return {"curves": curves, "scores": paraphrase_scores(ref, retriever, llm)}


def verify(result):
    c, rows = result["curves"], result["scores"]
    hits = sorted(s for s, hit, _ in rows if hit)
    misses = sorted(s for s, hit, _ in rows if not hit)
    dup = [s for s, hit, p in rows if "an transfer" in p]
    mean = lambda xs: round(sum(xs) / len(xs), 3)  # noqa: E731
    return [
        practice.Check(
            "ANSWER: raising the confidence threshold never raises recall@3 and costs multi-query a quarter above 0.5",
            c == {"multiquery": [1.0, 1.0, 1.0, 0.75, 0.75], "decompose": [0.75] * 5},
            f"recall@3 at thresholds {list(THRESHOLDS)}: {c}",
        ),
        practice.Check(
            "FINDING: the paraphrases that find the gold document score lower on similarity to the query than those that miss",
            (len(hits), mean(hits), len(misses), mean(misses), hits[:2], misses[0], dup)
            == (5, 0.501, 4, 0.658, [0.279, 0.29], 0.282, [0.907, 0.907])
            and not any(hit for _, hit, p in rows if "an transfer" in p),
            f"gold-at-1 paraphrases {hits} (mean {mean(hits)}); the rest {misses} (mean {mean(misses)}); "
            f"'an transfer' duplicates {dup}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
