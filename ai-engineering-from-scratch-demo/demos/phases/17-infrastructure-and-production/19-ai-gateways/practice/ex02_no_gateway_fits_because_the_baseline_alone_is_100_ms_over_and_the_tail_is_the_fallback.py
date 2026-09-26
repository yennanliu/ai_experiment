"""Exercise 2 — no gateway fits, because the baseline alone is 100 ms over, and the tail is the fallback.

    Your SLA is TTFT P99 < 200 ms on a 300 ms baseline. Which gateways stay
    within budget?

Reading of the exercise: read literally, the provider's P99 TTFT is 300 ms
and the SLA is P99 < 200 ms, so a gateway's budget is 200 - 300 ms. The other
reading -- 200 ms of headroom on top of the 300 ms baseline -- is the lesson's
own "For P99 < 500 ms, any" line, and it is tested too. Gateway P99 overhead is
the top of each range the lesson's "Latency budget" gives (LiteLLM 15, Portkey
40, Kong 8, Cloudflare 3 ms), since a P99 budget needs the worst case.

**ANSWER: none.** The budget is -100 ms: with no gateway at all the SLA is
missed by 100 ms, and every gateway only adds to that. Under the headroom
reading every gateway fits -- 300 + 40 = 340 ms for Portkey, the worst, against
500 -- so the gateway choice decides nothing either way.

**FINDING: on the lesson's simulator the P99 is set by the fallback, not the
gateway.** Its latencies are deterministic per path: a first-try OpenAI hit is
180 ms, a fallback to Anthropic 54 + 220 = 274 ms. 3% of requests fall back,
more than 1%, so the P99 is 274 ms plus the gateway -- 276 ms for Cloudflare,
304 ms for Portkey. Fallback adds 94 ms to the tail; the gateway adds 2 to 30.
The code reports only the mean, which hides it (185 to 213 ms).

**FINDING: the lesson's "P99 < 100 ms: Kong or Cloudflare" rule assumes a
baseline of 85 to 91 ms.** Kong and Cloudflare fit and LiteLLM and Portkey miss
only when baseline + 8 < 100 <= baseline + 15. None of the simulator's
providers is that fast: the quickest, self-hosted, is 100 ms before any gateway.

Structure: `paths()` enumerates the simulator's chain exactly; `p99()` reads
the percentile off that distribution; `fits()` applies a budget.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "19-ai-gateways"
SLA, BASELINE = 200, 300
P99_OVERHEAD = {"LiteLLM": 15, "Portkey": 40, "Kong": 8, "Cloudflare": 3}


def paths(providers):
    """(latency, probability) for each way a request can end."""
    out, reach, spent = [], 1.0, 0.0
    for p in providers:
        out.append((spent + p.base_latency_ms, reach * (1 - p.error_rate)))
        spent += 0.3 * p.base_latency_ms
        reach *= p.error_rate
    return out + [(spent, reach)]


def p99(dist, overhead=0.0):
    total = 0.0
    for latency, prob in sorted(dist):
        total += prob
        if total >= 0.99:
            return round(latency + overhead, 6)
    return None


def fits(baseline, sla):
    return [g for g, ovh in P99_OVERHEAD.items() if baseline + ovh < sla]


def rule_baselines(sla=100):
    """Integer baselines at which exactly Kong and Cloudflare meet the SLA."""
    return [b for b in range(sla) if fits(b, sla) == ["Kong", "Cloudflare"]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    dist = paths(ref.PROVIDERS)
    return {
        "literal": fits(BASELINE, SLA), "budget": SLA - BASELINE,
        "headroom": fits(BASELINE, BASELINE + SLA),
        "sim_p99": {g: p99(dist, o) for g, o in ref.GATEWAY_OVERHEAD.items()},
        "bare_p99": p99(dist), "primary": dist[0][0],
        "fallback_share": round(1 - dist[0][1], 6),
        "mean": {g: round(ref.simulate_fallback(g)["mean_latency"]) for g in ref.GATEWAY_OVERHEAD},
        "rule": rule_baselines(),
        "fastest": min(p.base_latency_ms for p in ref.PROVIDERS),
        "doc_any": "For P99 < 500 ms, any" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    sim, rule = result["sim_p99"], result["rule"]
    return [
        practice.Check(
            "ANSWER: none -- the baseline alone misses the SLA by 100 ms",
            result["literal"] == [] and result["budget"] == -100
            and result["headroom"] == list(P99_OVERHEAD) and result["doc_any"],
            f"budget {result['budget']} ms leaves {result['literal']}; as 200 ms of "
            f"headroom on 300 every gateway fits: {result['headroom']}",
        ),
        practice.Check(
            "FINDING: on the lesson's simulator the P99 is set by the fallback, not the gateway",
            all([result["bare_p99"] == 274.0, result["fallback_share"] == 0.03,
                 sim == {"LiteLLM": 284.0, "Portkey": 304.0, "Kong": 279.0, "Cloudflare": 276.0}]),
            f"{result['fallback_share']:.2%} fall back, so P99 is {result['bare_p99']} ms "
            f"with no gateway against a {result['primary']} ms first try; with gateways "
            f"{sim}; the reported means are {result['mean']}",
        ),
        practice.Check(
            "FINDING: the 'P99 < 100 ms: Kong or Cloudflare' rule assumes an 85-91 ms baseline",
            rule == list(range(85, 92)) and result["fastest"] >= 100,
            f"exactly Kong and Cloudflare fit at baselines {rule[0]}..{rule[-1]} ms; the "
            f"simulator's fastest provider is {result['fastest']} ms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
