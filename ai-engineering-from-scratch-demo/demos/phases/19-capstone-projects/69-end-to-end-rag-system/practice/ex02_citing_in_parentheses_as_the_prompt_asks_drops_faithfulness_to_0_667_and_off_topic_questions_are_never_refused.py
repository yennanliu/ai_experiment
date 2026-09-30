"""Exercise 2 — a real model call is the whole generate stage, and anchors in parentheses as the prompt asks fail the faithfulness gate.

    Add a real LLM call for the generator behind an env flag. Default to the mock. Measure the latency delta.

Reading of the exercise: a hosted model would need a network and a key, and
this repo runs offline. So the real LLM is the transformer from Phase 19
lesson 35: its `GPTModel` and `generate`, built at a small CPU size (4
layers, d_model 128, vocab 2048, seed 0), with a real forward pass per
token. Latency depends on the model's shape and on the token counts, not on
its weights. `generate` reads the env flag `RAG_REAL_LLM`. Unset, it calls the
lesson's mock `generate_answer`. Set to 1, it fills the lesson's own prompt
template (from docs/en.md) with the question and the top-k snippets. The
model then generates as many tokens as the mock's answer has. It is
installed in place of `generate_answer`, so the latency is the pipeline's own
`latency_ms["generate"]`, the median of 3 runs per eval query. The
milliseconds are printed; the checks hold the token counts and a wide ratio.

**ANSWER: the default stays the mock, and the flag makes generation the
whole cost.** Over the 4 eval queries the prompts are 127 to 135 tokens and
the answers 16 to 30 tokens. Generation goes from about 0.03 ms to about
100-230 ms per query on one local run, a ratio in the thousands (the check
holds only 20x). With the flag set, generation is about 97% of each query's
latency. Retrieval and rerank together stay at a few milliseconds.

**FINDING: an answer that follows the lesson's prompt fails the lesson's
faithfulness gate.** The template says "Cite every claim with the anchor in
parentheses". The mock writes `[d3:0]`, and `faithfulness_score` strips only
square brackets. The same 4 mock answers with `(d3:0)` anchors score 0.667,
0.500, 1.000 and 0.500, a mean of 0.667 against the 0.75 threshold. The
split leaves each anchor behind as a claim, and "d3" is in no context.

**FINDING: the refuse-on-low-confidence path never fires.** 6 off-topic
questions ("who wrote hamlet", "best pizza recipe with basil", ...) get 0
refusals. Each gets a cited answer and faithfulness 1.0, because the answer
is a sentence copied from a chunk. The lowest top-1 blended score is over
10 times `REFUSE_THRESHOLD` = 0.05. A refusal on an on-topic query would
score faithfulness 0.0 on all 4, so the metric punishes the safety valve.
"""

from __future__ import annotations

import os
import re
import statistics

import torch

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "69-end-to-end-rag-system"
FLAG, REPEATS = "RAG_REAL_LLM", 3
OFF_TOPIC = ["what is the capital of france", "how many legs does a spider have", "who wrote hamlet",
             "best pizza recipe with basil", "what time is it in tokyo", "xyzzy plugh"]


def template():
    fenced = re.findall(r"^```(\w*)\n(.*?)^```$", parity.doc_text(PHASE, LESSON), flags=re.S | re.M)
    return next(body for lang, body in fenced if not lang)


def make_generate(ref, gpt, model, mock, stats):
    vocab = sorted({t for _, text in ref.CORPUS for t in ref.tokenize(text)})

    def generate(query, ranked):
        answer = mock(query, ranked)
        if os.environ.get(FLAG) != "1":
            return answer
        snippets = "\n".join(f"({c.anchor()}) {c.text}" for c, _ in ranked)
        prompt = template().replace("{query}", query).replace("{enumerated chunks with anchors}", snippets)
        ids = [ref._token_to_id(t) % model.cfg.vocab_size for t in ref.tokenize(prompt)]
        new = len(ref.tokenize(answer[0]))
        out = gpt.generate(model, torch.tensor([ids]), max_new_tokens=new, top_k=50, seed=0)[0, len(ids):]
        stats.append((len(ids), new))
        return " ".join(vocab[i % len(vocab)] for i in out.tolist()), []

    return generate


def timed(p, query, real):
    os.environ[FLAG] = "1" if real else "0"
    runs = [p.query(query).latency_ms for _ in range(REPEATS)]
    return {k: statistics.median(r[k] for r in runs) for k in runs[0]}


def latency(ref, gpt, p):
    torch.manual_seed(0)
    cfg = gpt.GPTConfig(vocab_size=2048, context_length=256, d_model=128, num_heads=4, num_layers=4, dropout=0.0)
    stats, mock = [], ref.generate_answer
    ref.generate_answer = make_generate(ref, gpt, gpt.GPTModel(cfg), mock, stats)
    try:
        rows = [(timed(p, e.query, False), timed(p, e.query, True)) for e in ref.EVAL_QUERIES]
    finally:
        ref.generate_answer = mock
        os.environ.pop(FLAG, None)
    return {
        "prompt_tokens": sorted({s[0] for s in stats}), "new_tokens": sorted({s[1] for s in stats}),
        "mock_ms": [round(m["generate"], 3) for m, _ in rows], "real_ms": [round(r["generate"], 1) for _, r in rows],
        "ratio": min(r["generate"] / max(m["generate"], 1e-6) for m, r in rows),
        "real_share": min(r["generate"] / sum(r.values()) for _, r in rows),
    }


def audits(ref, p):
    results = [p.query(e.query) for e in ref.EVAL_QUERIES]
    paren = [ref.faithfulness_score(re.sub(r"\[([^\]]+)\]", r"(\1)", r.answer), r.top_k) for r in results]
    return {
        "paren": [round(x, 3) for x in paren], "threshold": ref.THRESHOLDS["faithfulness"],
        "refusal_faith": [ref.faithfulness_score(ref.REFUSE_TEXT, r.top_k) for r in results],
    }


def refusal_audit(ref, p):
    off = [p.query(q) for q in OFF_TOPIC]
    return {
        "refused": sum(r.answer == ref.REFUSE_TEXT for r in off),
        "off_faith": [ref.faithfulness_score(r.answer, r.top_k) for r in off],
        "off_min_top": min(r.top_k[0][1] for r in off), "refuse_at": ref.REFUSE_THRESHOLD,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gpt = parity.load_reference(PHASE, "35-gpt-model-assembly", "main")
    p = ref.build_pipeline()
    return {**latency(ref, gpt, p), **audits(ref, p), **refusal_audit(ref, p)}


def verify(result):
    r = result
    mean_paren = sum(r["paren"]) / len(r["paren"])
    return [
        practice.Check(
            "ANSWER: the flag swaps in a real forward pass, >= 20x the mock and > half of each query",
            (min(r["prompt_tokens"]), max(r["prompt_tokens"]), min(r["new_tokens"]), max(r["new_tokens"])) == (127, 135, 16, 30)
            and r["ratio"] >= 20 and r["real_share"] > 0.5,
            f"prompt tokens {r['prompt_tokens']}, answer tokens {r['new_tokens']}; generate ms mock {r['mock_ms']} "
            f"vs real {r['real_ms']}; min ratio {r['ratio']:.0f}x, min share of query {r['real_share']:.2f}",
        ),
        practice.Check(
            "FINDING: anchors in parentheses, as the lesson's prompt asks, drop faithfulness below its threshold",
            r["paren"] == [0.667, 0.5, 1.0, 0.5] and mean_paren < r["threshold"],
            f"faithfulness with (anchor) citations {r['paren']}, mean {mean_paren:.3f} vs threshold {r['threshold']}",
        ),
        practice.Check(
            "FINDING: refuse-on-low-confidence never fires, and a refusal scores faithfulness 0",
            (r["refused"], r["off_faith"], r["refusal_faith"]) == (0, [1.0] * 6, [0.0] * 4)
            and r["off_min_top"] > 10 * r["refuse_at"],
            f"{r['refused']}/6 off-topic queries refused, faithfulness {r['off_faith']}; lowest top-1 score "
            f"{r['off_min_top']:.3f} vs threshold {r['refuse_at']}; refusal faithfulness on eval {r['refusal_faith']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
