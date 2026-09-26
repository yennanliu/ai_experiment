"""Exercise 5 — 20 ms is fine with headroom above 300 ms; at 100 ms the SLA is already lost, so log async.

    Helicone is 20ms proxy overhead. At P99 TTFT 300 ms, is that acceptable?
    What if SLA is 100 ms?

Reading of the exercise: a constant 20 ms proxy shifts every TTFT, so it
moves P99 by exactly 20 ms; what it costs is how many requests cross the SLA
line, which depends on how dense the distribution is there. This lesson's
`code/` has no latency model, so the distribution is lesson 08's
`synth_workload()` (2,000 requests, seed 7), scaled so its P99 TTFT is the
exercise's 300 ms.

**ANSWER: at 300 ms it is acceptable if the SLA has 20 ms of headroom; at a
100 ms SLA it is not, and the SLA is already lost without it.** P99 goes 300 ->
320 ms, +6.7%. Against an SLA at 320 ms or above it passes; the Phase 17 · 19
rule "For P99 < 500 ms, any" gateway agrees. Against an SLA *equal* to the 300
ms baseline it breaks: attainment within 300 ms falls from 99.05% to 97.1%.
At a 100 ms SLA the baseline P99 is already 3x over; the proxy takes 20% of
the budget, so the model would need P99 <= 80 ms; and attainment within 100 ms
falls 75.8% -> 69.6%. Remedy: Helicone's async integration, which its docs
say is "not on the critical path" -- at the price of the proxy-only Bucket
Cache, Retries and Custom rate limiting that make it "a gateway too".

**FINDING: the same 20 ms costs 12x more attainment at 100 ms than at 320 ms.**
It drops 6.2 points of requests at a 100 ms line, where the scaled workload's
body sits (median 48.4 ms), and 0.5 points at 320 ms, in the thin tail. The
overhead's share of the budget (20% vs 6%) understates the gap; the density
at the line is what sets it.

**FINDING: the curriculum already rules a 20 ms gateway out at 100 ms.**
Phase 17 · 19 says "For TTFT P99 < 100 ms SLA, Kong or Cloudflare" (3-8 and
1-3 ms) and lists Portkey at 20-40 ms; Helicone's 20 ms is Portkey's floor.
Helicone's own proxy-vs-async page (read 2026-09-26) gives no millisecond
figure at all, so the 20 ms is the lesson's number, not the vendor's.

Structure: `scaled()` rescales lesson 08's TTFTs; `within()` is the fraction
of requests under a line, before and after the shift.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "13-llm-observability"
GOODPUT = "08-inference-metrics-goodput"
P99, OVERHEAD = 300.0, 20.0
LINES = (100, 300, 320)


def scaled(l08):
    ttft = [t.ttft_ms for t in l08.synth_workload(n=2000)]
    k = P99 / l08.percentiles(ttft, [0.99])[0]
    return [x * k for x in ttft]


def within(ttft, line, shift=0.0):
    return sum(x + shift <= line for x in ttft) / len(ttft)


def solve():
    l08 = parity.load_reference(PHASE, GOODPUT, "main")
    ttft = scaled(l08)
    p50, p99 = l08.percentiles(ttft, [0.5, 0.99])
    after = l08.percentiles([x + OVERHEAD for x in ttft], [0.99])[0]
    gateways = parity.doc_text(PHASE, "19-ai-gateways")
    names = dir(parity.load_reference(PHASE, LESSON, "main"))
    return {
        "p50": p50,
        "p99": p99,
        "p99_after": after,
        "lines": {
            line: (within(ttft, line), within(ttft, line, OVERHEAD)) for line in LINES
        },
        "l19": (
            "For TTFT P99 < 100 ms SLA, Kong or Cloudflare" in gateways,
            "Portkey: 20-40 ms overhead" in gateways,
        ),
        "latency_names": [
            n for n in names if "latency" in n.lower() or "ttft" in n.lower()
        ],
    }


def verify(result):
    lines = result["lines"]
    loss = {line: round(before - after, 4) for line, (before, after) in lines.items()}
    return [
        practice.Check(
            "ANSWER: acceptable at 300 ms given 20 ms of headroom; at 100 ms the SLA is already lost",
            all(
                [
                    round(result["p99"], 6) == P99,
                    round(result["p99_after"], 6) == 320,
                    lines[300] == (0.9905, 0.971),
                    lines[100] == (0.758, 0.696),
                ]
            ),
            f"P99 {result['p99']:.0f} -> {result['p99_after']:.0f} ms (+{OVERHEAD / P99:.1%}); "
            f"within 300 ms {lines[300][0]:.2%} -> {lines[300][1]:.2%}; within 100 ms "
            f"{lines[100][0]:.1%} -> {lines[100][1]:.1%}; a 100 ms SLA leaves the model 80 ms",
        ),
        practice.Check(
            "FINDING: the same 20 ms costs 12x more attainment at 100 ms than at 320 ms",
            loss[100] == 0.062
            and loss[320] == 0.005
            and round(result["p50"], 1) == 48.4,
            f"attainment lost per line {loss}; median TTFT {result['p50']:.1f} ms",
        ),
        practice.Check(
            "FINDING: the curriculum already rules a 20 ms gateway out at 100 ms",
            all(result["l19"]) and result["latency_names"] == [],
            "lesson 19: 'For TTFT P99 < 100 ms SLA, Kong or Cloudflare', Portkey 20-40 ms; this "
            "lesson's code has no latency model, so lesson 08's workload stands in",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
