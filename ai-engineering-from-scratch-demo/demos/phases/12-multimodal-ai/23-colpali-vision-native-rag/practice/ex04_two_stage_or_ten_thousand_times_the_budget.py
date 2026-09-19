"""Exercise 4 — two-stage, or ten thousand times the budget.

    Design the end-to-end pipeline for a 1M-page corpus with a latency budget of
    500ms per query. Pick ColQwen2 / VisRAG and justify.

Reading of the exercise: the single-stage cost is computed first, because it
settles the design before any model is picked -- a full MaxSim over a million
pages is four orders of magnitude outside the budget, so the only question left
is what the first stage is and how deep the second one goes.

**ANSWER: VisRAG first, ColQwen2 second, top-100 rerank.** A pooled one-vector
page index is **3.1 GB** for a million pages and answers an ANN query in
milliseconds; MaxSim then reranks the top **100** at **186.6 million** multiply-
adds, which fits the budget with room for the generator.

**FINDING: single-stage MaxSim is 10,000x too much work.** A 20-token query
against 1M pages of 729 patches at 128 dims is **1.87 trillion** multiply-adds
per query. The rerank is **186.6 million** -- the same arithmetic on 0.01% of the
corpus.

**FINDING: and single-stage is unaffordable in storage before it is in compute.**
The patch index is **373.2 GB** raw and **46.7 GB** at PQ 8x, against the pooled
index's **3.1 GB**. The first stage is not a speed optimisation; it is the only
stage that fits in memory.

**ANSWER: so the pick is both, and the justification is the recall hand-off.**
VisRAG's single vector per page is the thing Exercise 2 shows a mean similarity
does -- it asks whether a page is *about* the query, which is exactly the right
question for a first stage that must not miss. ColQwen2's MaxSim then asks
whether the answer is *on* the page, which is the right question once there are
only a hundred candidates. The failure mode to watch is the first stage's recall
at k=100, because nothing downstream can recover a page it did not return.

Structure: `maxsim_ops` prices a MaxSim pass over a candidate set, `index_bytes`
prices each index, and `STAGES` is the two-stage design being compared against
the single-stage one.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "23-colpali-vision-native-rag"
CORPUS = 1_000_000
PATCHES, DIM, FLOAT_BYTES = 729, 128, 4
QUERY_TOKENS, RERANK_K = 20, 100
PQ_FACTOR = 8
POOLED_KIB = 3.0          # the lesson's own VisRAG / text-RAG row
BUDGET_MS = 500


def maxsim_ops(pages, patches=PATCHES, tokens=QUERY_TOKENS, dim=DIM):
    return pages * patches * tokens * dim


def index_bytes(pages, per_page):
    return pages * per_page


def gb(value):
    return round(value / 1e9, 1)


FIRST_STAGE_RECALL = 0.95


def end_to_end(first_recall, rerank_quality):
    """Recall after reranking. The second stage reorders candidates; it never adds one."""
    return round(first_recall * rerank_quality, 4)


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    full = maxsim_ops(CORPUS)
    rerank = maxsim_ops(RERANK_K)
    patch_page = PATCHES * DIM * FLOAT_BYTES
    return {
        "corpus": CORPUS, "budget_ms": BUDGET_MS,
        "single_stage_ops": full, "rerank_ops": rerank,
        "ops_ratio": full // rerank,
        "candidate_share_pct": round(RERANK_K / CORPUS * 100, 4),
        "raw_gb": gb(index_bytes(CORPUS, patch_page)),
        "pq_gb": gb(index_bytes(CORPUS, patch_page // PQ_FACTOR)),
        "pooled_gb": gb(index_bytes(CORPUS, int(POOLED_KIB * 1024))),
        "storage_ratio": round(index_bytes(CORPUS, patch_page // PQ_FACTOR)
                               / index_bytes(CORPUS, int(POOLED_KIB * 1024)), 1),
        "rerank_k": RERANK_K,
        "stages": ("VisRAG pooled ANN", "ColQwen2 MaxSim rerank"),
        "recall_gate": "first stage recall@100",
        "first_recall": FIRST_STAGE_RECALL,
        "perfect_rerank": end_to_end(FIRST_STAGE_RECALL, 1.0),
        "half_rerank": end_to_end(FIRST_STAGE_RECALL, 0.5),
        "capped": end_to_end(FIRST_STAGE_RECALL, 1.0) <= FIRST_STAGE_RECALL,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: VisRAG first, ColQwen2 second, top-100 rerank",
            all([result["stages"] == ("VisRAG pooled ANN", "ColQwen2 MaxSim rerank"),
                 result["pooled_gb"] == 3.1, result["rerank_ops"] == 186_624_000,
                 result["rerank_k"] == RERANK_K]),
            f"a pooled one-vector index is {result['pooled_gb']} GB for "
            f"{result['corpus']:,} pages and answers an ANN query in milliseconds; MaxSim "
            f"then reranks the top {result['rerank_k']} at {result['rerank_ops']:,} "
            f"multiply-adds, inside a {result['budget_ms']} ms budget with room for the "
            "generator",
        ),
        practice.Check(
            "FINDING: single-stage MaxSim is 10,000x too much work",
            all([result["single_stage_ops"] == 1_866_240_000_000,
                 result["ops_ratio"] == 10_000,
                 result["candidate_share_pct"] == 0.01]),
            f"a {QUERY_TOKENS}-token query against {result['corpus']:,} pages of {PATCHES} "
            f"patches at {DIM} dims is {result['single_stage_ops']:,} multiply-adds. The "
            f"rerank is {result['rerank_ops']:,} -- {result['ops_ratio']:,}x less, the same "
            f"arithmetic on {result['candidate_share_pct']}% of the corpus",
        ),
        practice.Check(
            "FINDING: single-stage is unaffordable in storage before it is in compute",
            all([result["raw_gb"] == 373.2, result["pq_gb"] == 46.7,
                 result["pooled_gb"] == 3.1, result["storage_ratio"] == 15.2]),
            f"the patch index is {result['raw_gb']} GB raw and {result['pq_gb']} GB at PQ "
            f"{PQ_FACTOR}x, against the pooled index's {result['pooled_gb']} GB -- "
            f"{result['storage_ratio']}x. The first stage is not a speed optimisation; it is "
            "the only stage that fits in memory",
        ),
        practice.Check(
            "ANSWER: the justification is the recall hand-off",
            all([result["recall_gate"] == "first stage recall@100",
                 len(result["stages"]) == 2, result["capped"],
                 result["perfect_rerank"] == 0.95, result["half_rerank"] == 0.475]),
            "VisRAG's single vector per page asks whether a page is ABOUT the query -- what "
            "Exercise 2 shows a mean similarity does -- which is the right question for a "
            "stage that must not miss. ColQwen2's MaxSim then asks whether the answer is ON "
            f"the page, the right question once there are {RERANK_K} candidates. The number "
            f"to watch is {result['recall_gate']}, because nothing downstream recovers a "
            f"page it did not return: at a first stage of {result['first_recall']:.0%}, a "
            f"perfect reranker ends at {result['perfect_rerank']:.0%} and a half-useless one "
            f"at {result['half_rerank']:.1%}. The second stage reorders candidates and never "
            "adds one, so its quality can only subtract",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
