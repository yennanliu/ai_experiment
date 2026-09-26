"""Exercise 1 — a 1% token spike never reaches a 15 ms TPOT SLO, so goodput does not move.

    Run `code/main.py`. Generate a distribution with 1% tail spike. How does
    goodput change when you tighten P99 TPOT from 30 ms to 15 ms?

Reading of the exercise: "1% tail spike" is the reference's own knob,
`synth_workload(tail_spike_rate=0.01)`, and "tighten TPOT" is the `slo_tpot`
argument of its `goodput()`, with TTFT and E2E held at the lesson's target
profile (500 ms, 2000 ms). Because that knob spikes single *tokens*, a second
workload spikes whole *requests* -- 1% of them decode 3-8x slower, the
long-prefill-neighbour case the lesson describes -- built from the
reference's `RequestTrace`.

**ANSWER: it does not change at all -- 80.10% at 30 ms and 80.10% at 15 ms.**
The reference spikes each token independently, and `goodput()` tests the
request's *mean* TPOT, which averages ~175 tokens. The worst request in 2000
has TPOT 9.18 ms and the P99 is 8.30 ms, so no request is anywhere near 15 ms.
Tokens do spike -- per-token P99 is 18.17 ms -- but no per-request check can
see a token. The TPOT constraint first bites at 15 ms when the spike rate
reaches 20% (TPOT-only goodput 92.35%); at 5% it is still 100%.

**FINDING: goodput is set by E2E and TTFT; TPOT fails no request.** Under
the target profile the shipped run (2% spikes) fails 22.75% of requests on
E2E, 2.30% on TTFT and 0% on TPOT, and even the tight profile (TPOT 10 ms)
fails 0.15% on TPOT against 51.2% on E2E. The spikes still cost goodput --
it falls from 84.45% at rate 0 to 80.10% at 1% -- but they do it through E2E.

**FINDING: with whole-request spikes, tightening costs 0.5 points alone and
0.1 jointly.** 26 of 2000 requests spike (TPOT 21.9-55.5 ms, P99 26.70 ms).
TPOT-only goodput goes 99.20% -> 98.70% from 30 to 15 ms; under the joint
target SLO it goes 83.30% -> 83.20%, because E2E already fails 24 of the 26.

**FINDING: the module's printed KEY FINDING contradicts its own output.** It
says "P99 TPOT ~25-40 ms" -- the run prints 8.99 ms -- and "goodput collapses
from 99%", while the loose profile it prints is 100.00%.

Structure: `profile()` computes goodput and per-constraint failure rates with
the reference's functions; `request_spikes()` rescales reference traces.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "08-inference-metrics-goodput"
TTFT, E2E, INF, N = 500, 2000, float("inf"), 2000


def tpots(traces):
    return [t.tpot_genaiperf() for t in traces]


def profile(ref, traces, ttft, tpot, e2e):
    """Goodput plus the share failing each constraint on its own."""
    def fail(ok):
        return round(sum(not ok(t) for t in traces) / len(traces), 4)

    return {
        "goodput": round(ref.goodput(traces, ttft, tpot, e2e), 4),
        "ttft": fail(lambda t: t.ttft_ms <= ttft),
        "tpot": fail(lambda t: t.tpot_genaiperf() <= tpot),
        "e2e": fail(lambda t: t.e2e_ms <= e2e),
    }


def request_spikes(ref, rate=0.01, seed=1):
    """1% of requests decode 3-8x slower from start to end."""
    rng, out = random.Random(seed), []
    for t in ref.synth_workload(n=N, tail_spike_rate=0.0):
        k = rng.uniform(3, 8) if rng.random() < rate else 1.0
        decodes = [d * k for d in t.decode_ms_per_token]
        out.append(ref.RequestTrace(t.queue_ms, t.prefill_ms, decodes, t.output_tokens))
    return out


def tighten(ref, traces, ttft=TTFT, e2e=E2E):
    return {x: ref.goodput(traces, ttft, x, e2e) for x in (30, 15)}


def bite(ref, rate):
    return ref.goodput(ref.synth_workload(n=N, tail_spike_rate=rate), INF, 15, INF)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    one = ref.synth_workload(n=N, tail_spike_rate=0.01)
    shipped = ref.synth_workload(n=N)
    spiked = request_spikes(ref)
    tokens = [d for t in one for d in t.decode_ms_per_token]
    return {
        "one": tighten(ref, one),
        "max_tpot": max(tpots(one)), "p99_tpot": ref.percentiles(tpots(one), [0.99])[0],
        "token_p99": ref.percentiles(tokens, [0.99])[0],
        "bite": {0.05: bite(ref, 0.05), 0.2: bite(ref, 0.2)},
        "zero": ref.goodput(ref.synth_workload(n=N, tail_spike_rate=0.0), TTFT, 15, E2E),
        "target": profile(ref, shipped, TTFT, 15, E2E),
        "tight": profile(ref, shipped, 300, 10, 1500),
        "loose": ref.goodput(shipped, 800, 25, 3000),
        "shipped_p99": ref.percentiles(tpots(shipped), [0.99])[0],
        "spiked": sorted(x for x in tpots(spiked) if x > 15),
        "alone": tighten(ref, spiked, INF, INF),
        "joint": tighten(ref, spiked),
        "caught": sum(t.e2e_ms > E2E for t in spiked if t.tpot_genaiperf() > 15),
    }


def verify(r):
    tgt, spk = r["target"], r["spiked"]
    return [
        practice.Check(
            "ANSWER: it does not change at all -- 80.10% at 30 ms and at 15 ms",
            all([r["one"][30] == r["one"][15] == 0.801, r["max_tpot"] < 15,
                 r["token_p99"] > 15, r["bite"][0.05] == 1.0, r["bite"][0.2] < 1.0]),
            f"goodput {r['one'][30]:.2%} at both; worst request TPOT {r['max_tpot']:.2f} ms, "
            f"P99 {r['p99_tpot']:.2f}, per-token P99 {r['token_p99']:.2f}; TPOT-only goodput "
            f"at 15 ms is {r['bite'][0.05]:.2%} at 5% spikes and {r['bite'][0.2]:.2%} at 20%",
        ),
        practice.Check(
            "FINDING: goodput is set by E2E and TTFT; TPOT fails no request",
            tgt["tpot"] == 0 and tgt["e2e"] > 5 * tgt["ttft"] > 0
            and r["tight"]["tpot"] < 0.01 < r["tight"]["e2e"] and r["zero"] > r["one"][15],
            f"target profile fails {tgt['e2e']:.2%} on E2E, {tgt['ttft']:.2%} on TTFT, "
            f"{tgt['tpot']:.2%} on TPOT; tight {r['tight']}; goodput {r['zero']:.2%} at "
            f"rate 0 -> {r['one'][15]:.2%} at 1%",
        ),
        practice.Check(
            "FINDING: with whole-request spikes, tightening costs 0.5 points alone and 0.1 jointly",
            all([len(spk) == 26, round(r["alone"][30] - r["alone"][15], 4) == 0.005,
                 round(r["joint"][30] - r["joint"][15], 4) == 0.001, r["caught"] == 24]),
            f"{len(spk)} spiked requests at TPOT {spk[0]:.1f}-{spk[-1]:.1f} ms; TPOT-only "
            f"{r['alone'][30]:.2%} -> {r['alone'][15]:.2%}, joint {r['joint'][30]:.2%} -> "
            f"{r['joint'][15]:.2%}; E2E already fails {r['caught']} of them",
        ),
        practice.Check(
            "FINDING: the module's printed KEY FINDING contradicts its own output",
            r["shipped_p99"] < 25 and r["loose"] == 1.0,
            f"it claims P99 TPOT ~25-40 ms and a collapse from 99%; the run has P99 "
            f"{r['shipped_p99']:.2f} ms and loose goodput {r['loose']:.2%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
