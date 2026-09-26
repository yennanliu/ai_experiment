"""Exercise 5 — the tools also disagree on per-request throughput, token count and the percentile itself.

    Read the LLMPerf README and the GenAI-Perf docs. Identify three other
    metrics where the tools disagree.

Reading of the exercise: the documents were read on 2026-09-26: the LLMPerf
README, plus its `token_benchmark_ray.py` and OpenAI client (the README
defines almost nothing), the GenAI-Perf README metrics table, and NVIDIA's
NIM metrics page. Each disagreement is then applied to the lesson's own 2000
traces so its size is measured, not asserted. The lesson's ITL claim holds in
the source: LLMPerf appends TTFT as the first entry of `time_to_next_token`,
and NIM's ITL is `(e2e - TTFT) / (tokens - 1)`.

**ANSWER: per-request output throughput, the output token count, and the
percentile.**

(1) *Per-request throughput.* LLMPerf's `request_output_throughput` is
`num_output_tokens / e2e`, so it includes TTFT. GenAI-Perf's "Output Token
Throughput Per User" excludes the first token and divides by the generation
phase only. On the lesson's traces the means are 117.3 and 130.4 tok/s, and
the gap reaches 2.50x on one request.

(2) *Output token count.* LLMPerf counts tokens with `LlamaTokenizer`
"regardless of which LLM API is being tested", from
`hf-internal-testing/llama-tokenizer`, a 32K-vocab Llama-2-style tokenizer.
GenAI-Perf defaults to the model's own tokenizer. The Llama 3 paper reports
3.17 vs 3.94 English characters per token for the Llama 2 and Llama 3
tokenizers, so on a Llama 3.x model LLMPerf's output length and throughput
read 1.243x GenAI-Perf's for identical text.

(3) *The percentile.* LLMPerf takes pandas quantiles, which interpolate
linearly, at p25/50/75/90/95/99. GenAI-Perf's table lists p99/p90/p75. The
lesson's `percentiles()` picks index `int(p * n)`, a third rule. On 50
requests, the size the lesson's "30-50 iterations" suggests, the lesson's P99
TTFT is the sample maximum, 564.7 ms, and linear interpolation gives 558.3 ms.

**FINDING: LLMPerf's ITL and throughput use two different token counts.** It
divides the summed gaps by the client's streamed-chunk count, then
overwrites `NUM_OUTPUT_TOKENS` with the tokenizer count before computing
throughput. On a Llama 3 model, ITL x throughput is therefore 1.243, not
1.0.

**FINDING: the lesson's trace charges one decode gap too many.** Each trace
has as many decode gaps as output tokens, all after TTFT. GenAI-Perf's
formula divides them by N - 1, so `tpot_genaiperf()` overstates the
generating per-token mean by 0.055 ms on average.

Structure: one function per tool definition over reference traces;
`statistics.quantiles(method="inclusive")` is pandas' linear rule.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "08-inference-metrics-goodput"
LLAMA2_CPT, LLAMA3_CPT = 3.17, 3.94  # English chars/token, Llama 3 paper (arXiv:2407.21783)
TOKEN_RATIO = LLAMA3_CPT / LLAMA2_CPT  # Llama-2 tokens per Llama-3 token


def llmperf_throughput(t, ratio=1.0):
    return t.output_tokens * ratio / t.e2e_ms * 1000


def genaiperf_per_user(t):
    return (t.output_tokens - 1) / sum(t.decode_ms_per_token) * 1000


def llmperf_itl_times_throughput(t, ratio):
    """ITL over chunks (one token per chunk) times throughput over tokenizer tokens."""
    itl_s = t.e2e_ms / t.output_tokens / 1000
    return itl_s * llmperf_throughput(t, ratio)


def linear_p99(values):
    return statistics.quantiles(values, n=100, method="inclusive")[98]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traces = ref.synth_workload(n=2000)
    first50 = [t.ttft_ms for t in traces[:50]]
    overstate = [t.tpot_genaiperf() - sum(t.decode_ms_per_token) / t.output_tokens
                 for t in traces]
    return {
        "llmperf": statistics.mean(llmperf_throughput(t) for t in traces),
        "per_user": statistics.mean(genaiperf_per_user(t) for t in traces),
        "worst": max(genaiperf_per_user(t) / llmperf_throughput(t) for t in traces),
        "ratio": TOKEN_RATIO,
        "lesson_p99": ref.percentiles(first50, [0.99])[0], "max50": max(first50),
        "linear_p99": linear_p99(first50),
        "product": llmperf_itl_times_throughput(traces[0], TOKEN_RATIO),
        "overstate": statistics.mean(overstate),
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: per-request throughput, the output token count, and the percentile",
            all([round(r["llmperf"], 1) == 117.3, round(r["per_user"], 1) == 130.4,
                 r["worst"] > 2.4, round(r["ratio"], 3) == 1.243,
                 r["lesson_p99"] == r["max50"] > r["linear_p99"]]),
            f"mean per-request throughput LLMPerf {r['llmperf']:.1f} vs GenAI-Perf "
            f"{r['per_user']:.1f} tok/s, up to {r['worst']:.2f}x on one request; the Llama-2 "
            f"tokenizer counts {r['ratio']:.3f}x a Llama 3 model's tokens; on 50 requests P99 "
            f"TTFT is {r['lesson_p99']:.1f} ms by the lesson's rule (the maximum) and "
            f"{r['linear_p99']:.1f} ms by linear interpolation",
        ),
        practice.Check(
            "FINDING: LLMPerf's ITL and throughput use two different token counts",
            round(r["product"], 3) == round(r["ratio"], 3),
            f"ITL x throughput on one trace is {r['product']:.3f}, not 1.0",
        ),
        practice.Check(
            "FINDING: the lesson's trace charges one decode gap too many",
            0.03 < r["overstate"] < 0.07,
            f"N gaps divided by N - 1 overstate the per-token mean by {r['overstate']:.3f} ms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
