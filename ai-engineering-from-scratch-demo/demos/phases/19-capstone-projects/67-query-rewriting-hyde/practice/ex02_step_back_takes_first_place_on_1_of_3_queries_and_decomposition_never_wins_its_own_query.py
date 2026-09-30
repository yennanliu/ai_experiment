"""Exercise 2 — step-back takes first place on 1 of 3 queries, and decomposition never wins its own query.

    Add a fourth strategy: step-back prompting (ask the LLM for the more general question, retrieve on that, then narrow). Compare on the fixture.

Reading of the exercise: the "LLM" is a step-back table written in the same
style as the lesson's `MockLLM` tables. Each of the lesson's three `GOLD`
queries gets one more general question. A query outside the table is
returned unchanged. "Narrow" follows Zheng et al.,
"Take a Step Back" (arXiv:2310.06117, read 2026-09-29 at
https://arxiv.org/html/2310.06117v2), which retrieves on both the original
and the step-back question. Here the two ranked lists are merged with the
lesson's `rrf`. All five strategies (no-rewrite, HyDE, multi-query,
decompose, step-back) run through the lesson's `retrieve_with_rewriter` with
k = 8. Each is scored by the gold document's rank. The multi-topic query is
also scored on its second fact, d3 "Quota cooldown".

**ANSWER: step-back takes first place on one query of three and hurts one.**
Gold ranks for the three queries (transfer / merge / multi-topic):

    no-rewrite 3/5/1, HyDE 2/1/2, multi-query 1/2/1, decompose 3/5/1,
    step-back 3/1/2 (retrieving on the step-back question alone: 3/1/3)

Step-back ties HyDE on "merge two retrievers", where the general question
reaches "combine results" wording. It leaves "transfer breaks halfway"
unchanged. On the multi-topic query it pushes d1 down to rank 2 behind d3,
because the general question is about retries, not the upload.

**FINDING: the lesson's demo does not back "decomposition wins on the
multi-topic query".** No-rewrite and multi-query also rank d1 first there.
The second fact, d3, is at rank 2 for no-rewrite and decompose alike.
Decomposition is the only strategy that never takes first place outright.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "67-query-rewriting-hyde"
K = 8
STEP_BACK = {
    "what do we do when a transfer breaks halfway": "how does the storage service handle failed transfers",
    "how does the search service merge two retrievers": "how does search combine results from multiple retrievers",
    "what happens when an upload fails and the retry budget is exhausted": "how does the storage service handle failures and retries",
}


def step_back(query):
    key = query.lower().strip().rstrip("?").strip()
    return STEP_BACK.get(key, query)


def make_step_back_rewriter(ref, narrow=True):
    def rewrite(query):
        general = step_back(query)
        return ref.RewriteResult(strategy="step-back", rewrites=[general, query] if narrow else [general])

    rewriter = ref._IdentityRewriter()
    rewriter.rewrite = rewrite
    return rewriter


def ranks(ref, retriever, rewriter, query, docs):
    out = ref.retrieve_with_rewriter(query, rewriter, retriever, k_each=K, k_out=K)
    ids = [d.doc_id for d, _ in out["results"]]
    return [ids.index(d) + 1 if d in ids else -1 for d in docs]


def outright_wins(gold, names):
    """Queries on which a strategy alone ranks gold first."""
    firsts = [[n for n in names if gold[n][i] == 1] for i in range(len(gold[names[0]]))]
    return {n: sum(f == [n] for f in firsts) for n in names}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    retriever, llm = ref.build_retriever(), ref.MockLLM()
    strategies = {
        "no-rewrite": ref._IdentityRewriter(),
        "hyde": ref.HyDERewriter(llm=llm),
        "multiquery": ref.MultiQueryRewriter(llm=llm, n=3),
        "decompose": ref.DecomposeRewriter(llm=llm),
        "step-back": make_step_back_rewriter(ref),
        "step-back-only": make_step_back_rewriter(ref, narrow=False),
    }
    gold = {name: [ranks(ref, retriever, rw, q, [g])[0] for q, g, _ in ref.GOLD] for name, rw in strategies.items()}
    multi = ref.GOLD[2][0]
    d3 = {name: ranks(ref, retriever, strategies[name], multi, ["d3"])[0] for name in ("no-rewrite", "decompose")}
    return {"gold": gold, "d3": d3, "outright": outright_wins(gold, ["no-rewrite", "hyde", "multiquery", "decompose"])}


def verify(result):
    g = result["gold"]
    return [
        practice.Check(
            "ANSWER: step-back takes first place on 1 of 3 fixture queries (tied with HyDE) and hurts the multi-topic one",
            g == {"no-rewrite": [3, 5, 1], "hyde": [2, 1, 2], "multiquery": [1, 2, 1], "decompose": [3, 5, 1],
                  "step-back": [3, 1, 2], "step-back-only": [3, 1, 3]},
            f"gold rank per query (transfer / merge / multi-topic): {g}",
        ),
        practice.Check(
            "FINDING: decomposition never wins its own query outright; no-rewrite ties it on d1 and d3",
            result["outright"] == {"no-rewrite": 0, "hyde": 1, "multiquery": 1, "decompose": 0}
            and result["d3"] == {"no-rewrite": 2, "decompose": 2},
            f"outright first places {result['outright']}; d3 rank on the multi-topic query {result['d3']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
