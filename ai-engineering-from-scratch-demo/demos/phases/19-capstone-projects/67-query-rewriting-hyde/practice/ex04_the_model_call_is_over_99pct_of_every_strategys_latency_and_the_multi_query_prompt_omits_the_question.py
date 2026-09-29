"""Exercise 4 — the model call is over 99% of every strategy's latency, and the multi-query prompt omits the question.

    Replace the mock LLM with a real model call. Measure the latency-per-strategy on your stack.

Reading of the exercise: a hosted model would need a network and a key, and
this repo runs offline. So the "real model" is the transformer from Phase 19
lesson 35: its `GPTModel` and `generate`, built at a small CPU size (4
layers, d_model 128, vocab 2048, seed 0) and run as a real forward pass per
token. Latency depends on a model's shape and on how many tokens go in and
come out, not on its weights. For each strategy, the prompt is the lesson's
own template, taken from docs/en.md and filled with the query. The output
length is the length of the mock's answer. Tokens are the lesson's
`tokenize` words, hashed into the vocab. Each of the three `GOLD` queries
runs through each rewriter: model call, then retrieval through the lesson's
`HybridRetriever`, each timed as the median of 3 runs. The latencies are
printed, not asserted. The checks hold the deterministic cost drivers.

**ANSWER: on this stack, the LLM call is over 99% of every rewriting
strategy's latency, and decomposition is the cheapest.** Summed over the
three queries, the tokens generated are HyDE 78, multi-query 88 and
decompose 33. `generate` has no KV cache, so each step re-reads the whole
window, and the prompt length counts as well. The token positions the model
processes are 4,899 / 3,408 / 1,676. Multi-query generates the most tokens
but processes fewer positions than HyDE, because its prompt is shorter. On
one run here the three took about 129 / 112 / 50 ms. Every strategy makes
exactly one LLM call; retrieval costs under 1 ms.

**FINDING: the retrieval counts are not the ones the lesson states, and
the retrievals do not run in parallel.** HyDE runs 2 hybrid retrievals, not
1: `search_vec` plus a plain search on the original query. That leaves
one of its four underlying rankings from the hypothetical document. Multi-query
runs N + 1 = 4 retrievals, because the original query is kept. The lesson says
"the retrievals run in parallel", but `retrieve_with_rewriter` loops over
them in sequence.

**FINDING: the lesson's multi-query prompt has no slot for the question.**
The HyDE and decomposition templates carry `{user_query}`. The multi-query
template has only `{N}`, so as written the model is never shown the query
it is asked to rewrite.
"""

from __future__ import annotations

import re
import statistics
import time

import torch

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "67-query-rewriting-hyde"
REPEATS = 3


def templates():
    fenced = re.findall(r"^```(\w*)\n(.*?)^```$", parity.doc_text(PHASE, LESSON), flags=re.S | re.M)
    blocks = [body for lang, body in fenced if not lang]
    return {"hyde": blocks[0], "multiquery": blocks[1], "decompose": blocks[2]}


def llm_output(llm, name, query):
    if name == "hyde":
        return llm.generate_hypothetical(query)
    return " ".join(llm.paraphrase(query, 3) if name == "multiquery" else llm.decompose(query))


def ids(ref, text, vocab):
    return [sum(map(ord, t)) * 2654435761 % vocab for t in ref.tokenize(text)]


def timed(fn):
    runs = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(runs)


def one_run(ref, gpt, model, name, query, rewriter, retriever):
    prompt = templates()[name].replace("{user_query}", query).replace("{N}", "3")
    p_ids = ids(ref, prompt, model.cfg.vocab_size)
    new = len(ref.tokenize(llm_output(ref.MockLLM(), name, query)))
    tensor = torch.tensor([p_ids])
    llm_ms = timed(lambda: gpt.generate(model, tensor, max_new_tokens=new, top_k=50, seed=0))
    calls = {"n": 0}
    search, search_vec = retriever.search, retriever.search_vec
    retriever.search = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), search(*a, **k))[1]
    retriever.search_vec = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), search_vec(*a, **k))[1]
    ret_ms = timed(lambda: ref.retrieve_with_rewriter(query, rewriter, retriever, 8, 8))
    retriever.search, retriever.search_vec = search, search_vec
    return {"new": new, "positions": sum(len(p_ids) + i for i in range(new)), "llm_ms": llm_ms,
            "ret_ms": ret_ms, "retrievals": calls["n"] // REPEATS}


def summarise(rows):
    out = {k: sum(r[k] for r in rows) for k in ("new", "positions", "llm_ms", "ret_ms")}
    out["retrievals"] = [r["retrievals"] for r in rows]
    out["llm_share"] = out["llm_ms"] / (out["llm_ms"] + out["ret_ms"])
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gpt = parity.load_reference(PHASE, "35-gpt-model-assembly", "main")
    torch.manual_seed(0)
    cfg = gpt.GPTConfig(vocab_size=2048, context_length=256, d_model=128, num_heads=4, num_layers=4, dropout=0.0)
    model = gpt.GPTModel(cfg)
    llm, retriever = ref.MockLLM(), ref.build_retriever()
    rewriters = {"hyde": ref.HyDERewriter(llm=llm), "multiquery": ref.MultiQueryRewriter(llm=llm, n=3),
                 "decompose": ref.DecomposeRewriter(llm=llm)}
    out = {n: summarise([one_run(ref, gpt, model, n, q, rw, retriever) for q, _, _ in ref.GOLD])
           for n, rw in rewriters.items()}
    cols = {k: [out[n][k] for n in rewriters] for k in ("new", "positions", "retrievals", "llm_share")}
    cols["llm_ms"] = [round(out[n]["llm_ms"], 1) for n in rewriters]
    cols["ret_ms"] = [round(out[n]["ret_ms"], 2) for n in rewriters]
    return {**cols, "slots": {n: sorted(set(re.findall(r"\{\w+\}", t))) for n, t in templates().items()}}


def verify(result):
    r = result
    hyde_ms, mq_ms, dec_ms = r["llm_ms"]
    return [
        practice.Check(
            "ANSWER: the model call is >= 99% of every strategy's latency; decompose is the cheapest",
            (r["new"], r["positions"]) == ([78, 88, 33], [4899, 3408, 1676])
            and min(r["llm_share"]) >= 0.99 and dec_ms < min(hyde_ms, mq_ms),
            f"HyDE / multi-query / decompose: tokens generated {r['new']}, positions processed {r['positions']}; "
            f"LLM ms {r['llm_ms']}, retrieval ms {r['ret_ms']}, LLM share {[round(x, 3) for x in r['llm_share']]}",
        ),
        practice.Check(
            "FINDING: HyDE runs 2 retrievals and multi-query N + 1, one after another",
            r["retrievals"] == [[2, 2, 2], [4, 4, 4], [1, 1, 2]],
            f"hybrid retrievals per query {r['retrievals']}",
        ),
        practice.Check(
            "FINDING: the lesson's multi-query prompt never includes the question",
            r["slots"] == {"hyde": ["{user_query}"], "multiquery": ["{N}"], "decompose": ["{user_query}"]},
            f"template placeholders {r['slots']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
