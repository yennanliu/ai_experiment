"""Exercise 1 — one word apart scores lower than completely unrelated.

    **Add RAG integration.** Build a simple in-memory vector store with 20
    documents. When the template is `rag_answer`, embed the query, find the 3
    most similar documents, and inject them as context. Measure how response
    quality changes with and without RAG context. Track retrieval latency
    separately from LLM latency.

Reading of the exercise: the store is 20 one-line documents about this very
service, and the 10 queries each have exactly one right answer among them, so
"the 3 most similar" has something to be right or wrong about. Retrieval and
the LLM call are timed separately, which is what the exercise asks for and also
the only way to see that one of them is a `random.uniform`.

**ANSWER: recall@3 is 3 of 10, and the mean rank of the right document is 8.8
of 20 against 10.5 for a coin.** Ranks: 14, 3, 5, 13, 5, 3, 14, 1, 17, 13.

**MECHANISM: `simple_embedding` is `sha256` of the whole string.** It is not a
function of the words, so it cannot be a function of the meaning. Six
single-word rewrites of the first document score **0.7427** against it on
average; the other nineteen documents, which share nothing, score **0.7560**.
The near set scores *lower*, and the two ranges overlap almost completely.
Retrieval here is noise with a sorted order.

**FINDING: every pair of texts sits at 0.756 with sd 0.037.** The entries are
`int(hex_pair, 16) / 255`, all non-negative, so every vector lives in one
corner of the space and no two of them can be far apart. The maximum over the
190 document pairs is 0.8480, which is below `SemanticCache`'s 0.92 threshold:
the "semantic" cache can only ever hit on an exact repeat, and `similarity` is
1.0 on every hit it reports.

**ANSWER: "quality with and without RAG" compares two constants.**
`call_llm_with_retry` picks `SIMULATED_RESPONSES["rag"]` when the prompt
contains "context" and `["general"]` otherwise, and the `rag_answer` template
always renders the word "Context:". So the RAG answer is one fixed string and
the non-RAG answer is another, whatever the retrieved documents were.

**ANSWER: retrieval is under a tenth of a millisecond and the LLM call is
130.2 ms of `asyncio.sleep`.** More than a thousand to one. The number the
exercise asks to track separately turns out to be the only one that measures
anything real, and it is the small one.

Structure: `DOCS` is the store, `NEAR` six single-word rewrites of `DOCS[0]`,
`QUERIES` the ten labelled queries. `run` drives one coroutine with the
lesson's `asyncio.sleep` recording its delays instead of serving them and its
`random` pinned to a seed, restoring both afterwards, and returns the
simulated milliseconds alongside the result. `ranked` is the retrieval the
exercise asks for, `geometry` measures the embedding space and `latencies` the
two halves of a request.
"""

from __future__ import annotations

import asyncio
import itertools
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "13-production-app"
DOCS = [
    "caching reduces latency for repeated queries", "rate limiting protects the API",
    "retries use exponential backoff", "the fallback chain tries three models",
    "guardrails block prompt injection", "PII is redacted before the prompt is built",
    "cost is tracked per user and per model", "the semantic cache uses cosine similarity",
    "prompt templates are versioned by name", "A/B experiments split traffic by user hash",
    "streaming yields one token at a time", "health checks report cache statistics",
    "the eval log records output length", "tokens are four thirds of the word count",
    "gpt-4o-mini is the cheapest model", "claude-sonnet-5 is the default primary",
    "the request log carries a request id", "latency is measured in milliseconds",
    "emergency mode serves only cached responses", "traces show each component's duration",
]
NEAR = [DOCS[0].replace(a, b) for a, b in (
    ("caching", "the cache"), ("reduces", "cuts"), ("latency", "delay"),
    ("repeated", "repeat"), ("queries", "query"), ("for", "on"))]
QUERIES = [("how does caching help latency", 0), ("what protects the API", 1),
           ("how do retries work", 2), ("how many models in the fallback chain", 3),
           ("what blocks prompt injection", 4), ("when is PII redacted", 5),
           ("how is cost tracked", 6), ("what does the semantic cache use", 7),
           ("are prompts versioned", 8), ("how is A/B traffic split", 9)]


def run(ref, make, seed=7):
    slept, real, state = [], ref.asyncio.sleep, ref.random.getstate()

    async def instant(seconds, *_, **__):
        slept.append(seconds)

    ref.asyncio.sleep = instant
    ref.random.seed(seed)
    try:
        return asyncio.run(make()), round(sum(slept) * 1000, 1)
    finally:
        ref.asyncio.sleep, _ = real, ref.random.setstate(state)


def ranked(ref, query, store):
    embedded = ref.simple_embedding(query)
    return [d for _, d in sorted(((ref.cosine_similarity(embedded, e), d)
                                  for e, d in store), reverse=True)]


def geometry(ref, store):
    base = store[0][0]
    near = [ref.cosine_similarity(base, ref.simple_embedding(t)) for t in NEAR]
    far = [ref.cosine_similarity(base, e) for e, _ in store[1:]]
    pairs = [ref.cosine_similarity(a, b) for (a, _), (b, _) in itertools.combinations(store, 2)]
    return {"near_mean": round(statistics.mean(near), 4), "pair_max": round(max(pairs), 4),
            "far_mean": round(statistics.mean(far), 4),
            "overlap": min(max(near), max(far)) > max(min(near), min(far)),
            "pair_mean": round(statistics.mean(pairs), 4),
            "pair_sd": round(statistics.pstdev(pairs), 4),
            "cache_threshold": ref.SemanticCache().threshold}


def latencies(ref, store):
    start = time.perf_counter()
    context = "\n".join(ranked(ref, QUERIES[0][0], store)[:3])
    retrieval_ms, svc = (time.perf_counter() - start) * 1000, ref.ProductionLLMService()
    rag, llm_ms = run(ref, lambda: svc.handle_request(
        "u1", QUERIES[0][0], "rag_answer", variables={"context": context}))
    plain, _ = run(ref, lambda: svc.handle_request("u2", "tell me about the system"))
    return {"retrieval_ms": round(retrieval_ms, 4), "llm_ms": llm_ms,
            "rag_is_constant": rag["response"] == ref.SIMULATED_RESPONSES["rag"],
            "plain_is_constant": plain["response"] == ref.SIMULATED_RESPONSES["general"],
            "same_response": rag["response"] == plain["response"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "production_app")
    store = [(ref.simple_embedding(d), d) for d in DOCS]
    ranks = [ranked(ref, query, store).index(DOCS[gold]) + 1 for query, gold in QUERIES]
    return {"documents": len(DOCS), "queries": len(QUERIES), "ranks": ranks,
            "recall_at_3": sum(1 for r in ranks if r <= 3), "chance_rank": (len(DOCS) + 1) / 2,
            "mean_rank": round(statistics.mean(ranks), 1),
            **geometry(ref, store), **latencies(ref, store)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: recall@3 is 3 of 10 and the mean rank is 8.8 of 20",
            all([result["recall_at_3"] == 3, result["documents"] == 20,
                 result["mean_rank"] < result["chance_rank"]]),
            f"over {result['documents']} documents and {result['queries']} labelled queries "
            f"the right one ranks {result['ranks']} -- recall@3 "
            f"{result['recall_at_3']}/{result['queries']}, mean rank {result['mean_rank']} "
            f"against {result['chance_rank']} for a coin",
        ),
        practice.Check(
            "MECHANISM: one word apart scores lower than completely unrelated",
            all([result["near_mean"] < result["far_mean"], result["overlap"]]),
            f"six single-word rewrites of the first document score {result['near_mean']} "
            f"against it; the {result['documents'] - 1} documents that share nothing score "
            f"{result['far_mean']}, and the ranges overlap. `simple_embedding` is sha256 of "
            "the whole string: not a function of the words, so not one of the meaning",
        ),
        practice.Check(
            "FINDING: every pair sits at 0.756, and 0.92 is out of reach",
            all([result["pair_sd"] < 0.05, result["pair_max"] < result["cache_threshold"]]),
            f"the 190 document pairs have mean {result['pair_mean']} and sd "
            f"{result['pair_sd']}: the entries are int(hex, 16)/255, all non-negative, so no "
            f"two vectors can be far apart. The maximum, {result['pair_max']}, is below "
            f"SemanticCache's {result['cache_threshold']} -- it can only hit an exact repeat",
        ),
        practice.Check(
            "ANSWER: 'quality with and without RAG' compares two constants",
            all([result["rag_is_constant"], result["plain_is_constant"],
                 not result["same_response"]]),
            "`call_llm_with_retry` returns SIMULATED_RESPONSES['rag'] when the prompt "
            "contains 'context' and ['general'] otherwise, and the rag_answer template "
            "always renders 'Context:'. The RAG answer is one fixed string and the non-RAG "
            "answer is another, whatever was retrieved",
        ),
        practice.Check(
            "ANSWER: retrieval is under 0.1 ms against 130.2 ms of asyncio.sleep",
            all([result["retrieval_ms"] < 1.0, result["llm_ms"] > 100,
                 result["llm_ms"] / result["retrieval_ms"] > 100]),
            f"retrieval over {result['documents']} documents takes "
            f"{result['retrieval_ms']} ms; the LLM call asks for {result['llm_ms']} ms of "
            f"asyncio.sleep -- {result['llm_ms'] / result['retrieval_ms']:.0f}x. The only "
            "number here that measures real work is the small one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
