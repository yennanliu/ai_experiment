"""Exercise 5 — the five ms gap is TTFT spread over the output, so it needs a 506 ms TTFT at 100 tokens.

    GenAI-Perf reports TPOT=6ms; LLMPerf reports TPOT=11ms on the same server.
    Explain.

Reading of the exercise: "explain" is made quantitative with the two
formulas as lesson 08's `RequestTrace` implements them, checked against the
tools' own definitions: NVIDIA's NIM benchmarking metrics page gives ITL =
(e2e_latency - TTFT) / (output_tokens - 1) -- the page now names the tool
AIPerf -- and LLMPerf sums a `time_to_next_token` list whose first entry is
the TTFT, then divides by the output token count (`llmperf` source). Both
read 2026-09-26.

**ANSWER: LLMPerf folds the TTFT into its per-token average, GenAI-Perf does
not.** Per request, LLMPerf minus GenAI-Perf = (TTFT - GenAI TPOT) / N
exactly. So a 6 -> 11 ms gap is a TTFT of 5N + 6 ms: 506 ms at 100 output
tokens, 1006 ms at 200. A trace with TTFT 506 ms and 100 tokens over 594 ms
of decode reproduces 6.0 and 11.0 on lesson 08's code. The server is the
same; the 5 ms is prefill and queueing, divided across the output.

**FINDING: on lesson 08's own workload the gap is 1.03 ms, not 5.** Its
1000 synthetic requests (mean TTFT 154 ms, 50-300 tokens) give GenAI-Perf
7.70 and LLMPerf 8.73 ms, and the identity above reproduces the delta to
1e-9. A 5 ms gap needs TTFTs about 5x longer, or much shorter outputs, so it
is itself a clue: long prompts or a queue.

**FINDING: lesson 08's GenAI-Perf divides N decode gaps by N - 1.** Its trace
stores `output_tokens` decode steps *after* the TTFT, then divides their sum
by N - 1, so its GenAI-Perf TPOT is 0.71% above the mean decode step (7.70 vs
7.64 ms). The real tool has N - 1 gaps for N tokens.

**FINDING: LLMPerf's divisor is not the server's token count either.** It
counts output tokens by re-tokenizing the text with
`hf-internal-testing/llama-tokenizer`, so the N in its denominator depends on
that tokenizer, not on the served model's. That is a second, model-dependent
divergence; it was read in the source but not measured here.

Structure: `gap()` is the closed form; `solve()` runs lesson 08's
`RequestTrace` and `synth_workload` against it.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "22-load-testing-llm-apis"
METRICS = "08-inference-metrics-goodput"
GENAI, LLMPERF = 6.0, 11.0


def ttft_for(n, genai=GENAI, llmperf=LLMPERF):
    """TTFT at which the two tools report `genai` and `llmperf` for N tokens."""
    return llmperf * n - genai * (n - 1)


def gap(trace):
    return (trace.ttft_ms - trace.tpot_genaiperf()) / trace.output_tokens


def solve():
    ref = parity.load_reference(PHASE, METRICS, "main")
    n = 100
    ttft = ttft_for(n)
    decode = GENAI * (n - 1)
    trace = ref.RequestTrace(0.0, ttft, [decode / n] * n, n)
    traces = ref.synth_workload()
    genai = statistics.mean(t.tpot_genaiperf() for t in traces)
    llm = statistics.mean(t.tpot_llmperf() for t in traces)
    step = statistics.mean(sum(t.decode_ms_per_token) / t.output_tokens for t in traces)
    return {
        "ttft": {k: ttft_for(k) for k in (100, 200)},
        "reproduced": (round(trace.tpot_genaiperf(), 9), round(trace.tpot_llmperf(), 9)),
        "workload": (round(genai, 2), round(llm, 2), round(llm - genai, 2)),
        "identity": abs((llm - genai) - statistics.mean(gap(t) for t in traces)),
        "mean_ttft": round(statistics.mean(t.ttft_ms for t in traces)),
        "overstate": round(genai / step - 1, 4), "step": round(step, 2),
        "lesson": "ITL excludes TTFT; LLMPerf's includes it" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    w = result["workload"]
    return [
        practice.Check(
            "ANSWER: LLMPerf folds the TTFT into its per-token average, GenAI-Perf does not",
            result["lesson"] and result["ttft"] == {100: 506.0, 200: 1006.0}
            and result["reproduced"] == (6.0, 11.0),
            f"6 vs 11 ms needs TTFT {result['ttft']} ms by output length; lesson 08's "
            f"RequestTrace at 506 ms / 100 tokens reports {result['reproduced']}",
        ),
        practice.Check(
            "FINDING: on lesson 08's own workload the gap is 1.03 ms, not 5",
            w == (7.7, 8.73, 1.03) and result["identity"] < 1e-9 and result["mean_ttft"] == 154,
            f"GenAI-Perf {w[0]}, LLMPerf {w[1]}, delta {w[2]} ms at mean TTFT "
            f"{result['mean_ttft']} ms; (TTFT - TPOT)/N matches to {result['identity']:.1e}",
        ),
        practice.Check(
            "FINDING: lesson 08's GenAI-Perf divides N decode gaps by N - 1",
            result["overstate"] == 0.0071,
            f"its GenAI-Perf mean {w[0]} ms against a mean decode step of "
            f"{result['step']} ms, +{result['overstate']:.2%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
