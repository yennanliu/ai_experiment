"""Exercise 3 — chunked prefill moves the stall rather than removing it.

    Why does chunked prefill protect P99 TPOT but not mean TPOT?

Reading of the exercise: the "why" is shown by measuring it. A decode batch
steps every 7 ms, the lesson's mean TPOT. Every `period` steps a 32k-token
prompt arrives needing 800 ms of prefill, the lesson's Llama-3.3-70B H100
figure. Unchunked, one step absorbs the whole 800 ms. Chunked at 2048 tokens,
16 steps absorb 50 ms each. Step cost is additive, 7 ms plus the prefill work
scheduled in it, and every token in the batch waits for its step, so the
per-token ITL distribution is the step distribution. It is summarised with
the reference's `percentiles()`.

**ANSWER: chunking cannot change the mean, because the prefill work is the
same 800 ms either way.** It only changes how that work is spread across
steps. With one long prompt every 50 steps, mean ITL is 23.0 ms both ways.
P99 falls from 807 ms to 57 ms because the 800 ms stall is cut into 16 pieces.
The pieces are not free: P90 rises from 7 ms to 57 ms, since 32% of steps now
carry a chunk instead of 2%. The tail gets shorter by making more of the
distribution slow.

**FINDING: chunking protects P99 only when the stall is already above the
99th percentile's reach.** With one long prompt every 200 steps, the stall
occupies 0.5% of steps, so unchunked P99 is 7 ms. Chunked, it occupies 8%,
and P99 *rises* to 57 ms. The mean is 11.0 ms both ways, again.

**FINDING: the lesson's per-request TPOT cannot see chunking at all.**
`goodput()` checks each request's mean TPOT. Over a 175-token request that
averages the stall away: at period 50 P99 request TPOT is 25.29 ms both
unchunked and chunked, while per-token P99 differs 14x. Only a per-token ITL percentile shows the protection.

**FINDING: the lesson's own example has no 65 ms P99.** "500 tokens with
TPOT ~7 ms and 20 tokens with TPOT ~60 ms. Mean TPOT is 9 ms. P99 TPOT is
65 ms": the mean is 9.04 ms, but no token is slower than 60 ms, so P99 is
60 ms. The code's prefill cost also disagrees with the prose: `synth_workload`
charges 0.05 ms per prompt token, so a 32k prompt would take 1638 ms, not 800.

Structure: `steps()` builds the step-time series; `stats()` and
`request_p99()` summarise it per token and per request.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "08-inference-metrics-goodput"
BASE_MS, PREFILL_MS, PROMPT, CHUNK, STEPS, REQUEST = 7.0, 800.0, 32_768, 2048, 20_000, 175


def steps(period, chunked):
    """Per-step latency with a long prefill arriving every `period` steps."""
    pieces = PROMPT // CHUNK if chunked else 1
    extra = PREFILL_MS / pieces
    return [BASE_MS + (extra if i % period < pieces else 0.0) for i in range(STEPS)]


def stats(ref, series):
    p50, p90, p99 = ref.percentiles(series, [0.5, 0.9, 0.99])
    return {"mean": round(statistics.mean(series), 2), "p50": p50, "p90": p90, "p99": p99}


def request_p99(ref, series):
    """P99 of the lesson's per-request TPOT: the mean over each 175-step window."""
    window, means = sum(series[:REQUEST]), []
    for i in range(REQUEST, len(series)):
        means.append(window / REQUEST)
        window += series[i] - series[i - REQUEST]
    return round(ref.percentiles(means, [0.99])[0], 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {(p, c): steps(p, c) for p in (50, 200) for c in (False, True)}
    example = [7.0] * 500 + [60.0] * 20
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "tokens": {k: stats(ref, v) for k, v in runs.items()},
        "requests": {k: request_p99(ref, v) for k, v in runs.items()},
        "example": (round(statistics.mean(example), 2), ref.percentiles(example, [0.99])[0]),
        "doc_65": "P99 TPOT is 65 ms" in doc,
        "code_32k_ms": PROMPT * 0.05,  # synth_workload's prefill = prompt_len * 0.05
    }


def verify(r):
    t, q = r["tokens"], r["requests"]
    off, on, rare_off, rare_on = t[50, False], t[50, True], t[200, False], t[200, True]
    return [
        practice.Check(
            "ANSWER: chunking cannot change the mean; it spreads the same 800 ms",
            on["mean"] == off["mean"] == 23.0 and on["p99"] < off["p99"] / 10
            and on["p90"] > off["p90"],
            f"period 50: unchunked {off}, chunked {on}",
        ),
        practice.Check(
            "FINDING: chunking protects P99 only when the stall is already above it",
            rare_off["p99"] == BASE_MS < rare_on["p99"] and rare_off["mean"] == rare_on["mean"],
            f"period 200: unchunked {rare_off}, chunked {rare_on}",
        ),
        practice.Check(
            "FINDING: the lesson's per-request TPOT cannot see chunking at all",
            q[50, True] == q[50, False],
            f"per-request P99 TPOT {q[50, False]} ms unchunked vs {q[50, True]} chunked, "
            f"against per-token P99 {off['p99']} vs {on['p99']}",
        ),
        practice.Check(
            "FINDING: the lesson's own example has no 65 ms P99",
            r["doc_65"] and r["example"] == (9.04, 60.0) and r["code_32k_ms"] > 2 * PREFILL_MS,
            f"500 x 7 ms + 20 x 60 ms: mean {r['example'][0]}, P99 {r['example'][1]}; the "
            f"code's 0.05 ms/token prefills 32k tokens in {r['code_32k_ms']:.0f} ms, not 800",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
