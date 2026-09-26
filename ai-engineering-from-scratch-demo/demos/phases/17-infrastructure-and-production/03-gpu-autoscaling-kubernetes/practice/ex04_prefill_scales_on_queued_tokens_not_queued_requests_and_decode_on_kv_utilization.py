"""Exercise 4 — prefill scales on queued tokens, not queued requests, and decode on KV utilization.

    Pick a signal to autoscale disaggregated prefill pods and a different signal
    for decode pods. Justify both.

Reading of the exercise: "justify" is done by measurement. A seeded prefill
queue shows which backlog measure predicts TTFT, and a Little's-law decode
pool shows which signal each workload shape moves. Throughputs are
illustrative: 16k prompt tokens/s per prefill pod, 40 tokens/s per decoding
sequence, and a KV budget of 256k tokens (a TP=2 Llama 3.3 70B FP8 replica on
H100 80 GiB at 0.9 memory utilization).

**ANSWER: prefill on queued prompt tokens, decode on KV-cache utilization.**
This is the pair NVIDIA Dynamo's load planner uses: `prefill_scale_up_queue_tokens`
is a "prefill queue-token count" and `decode_scale_up_kv_rate` a "decode
KV-utilization percentage" (Planner configuration reference, fetched
2026-09-26). The shapes show why. At 1.5 req/s a RAG load (8000 in, 200 out)
runs prefill at 75% and KV at 24%. A reasoning load (500 in, 4000 out) runs
prefill at 5% and needs 146% of KV. Each signal is blind to the other role.

**FINDING: queued requests do not predict TTFT when prompts vary; queued
tokens do exactly.** 4000 seeded arrivals, 80% 500-token and 20%
16000-token prompts, at 70% prefill load: the correlation of TTFT wait
with requests ahead is 0.80, and with tokens ahead it is 1.0. With exactly 2
requests ahead the wait runs from 0.03s to 1.97s, 63x. The lesson says
"prefill pods scale on queue depth"; the depth has to be in tokens.

**FINDING: on decode pods a queue signal fires only after KV is full.** Decode
admits sequences while KV blocks remain, so nothing queues until KV
utilization reaches 100%, and beyond that vLLM preempts. For the reasoning
shape, KV passes an 80% scale-up line at 0.82 req/s and the queue first
appears at 1.02 req/s. The lesson's own simulator cannot show either signal:
every request costs a fixed REQUEST_PREFILL_SEC 0.6 + REQUEST_DECODE_SEC 1.8,
with no tokens and no KV.

Structure: `prefill_queue()` is a FIFO Lindley recursion; `decode_kv()` is
Little's law over mean context.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "03-gpu-autoscaling-kubernetes"
PREFILL_TPS, DECODE_TPS, KV_TOKENS = 16_000.0, 40.0, 256_000
SHAPES = {"rag": (8000, 200), "reasoning": (500, 4000)}  # (prompt, output) tokens
RATE, KV_UP = 1.5, 0.8


def prefill_queue(n=4000, seed=0, load=0.7):
    """Per arrival: (requests ahead, tokens ahead, wait seconds)."""
    rng = random.Random(seed)
    rate = load * PREFILL_TPS / (0.8 * 500 + 0.2 * 16_000)
    clock, finishes, rows = 0.0, [], []
    for _ in range(n):
        clock += rng.expovariate(rate)
        prompt = 16_000 if rng.random() < 0.2 else 500
        ahead = sum(1 for f in finishes[-50:] if f > clock)
        start = max(clock, finishes[-1] if finishes else 0.0)
        finishes.append(start + prompt / PREFILL_TPS)
        rows.append((ahead, (start - clock) * PREFILL_TPS, start - clock))
    return rows


def prefill_util(shape, rate=RATE):
    return rate * SHAPES[shape][0] / PREFILL_TPS


def decode_kv(shape, rate=RATE):
    """KV utilization by Little's law: concurrent sequences x mean context."""
    prompt, out = SHAPES[shape]
    return rate * (out / DECODE_TPS) * (prompt + out / 2) / KV_TOKENS


def rate_at(shape, kv):
    return kv / decode_kv(shape, 1.0)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = prefill_queue()
    waits = [w for _, _, w in rows]
    at2 = [w for a, _, w in rows if a == 2]
    return {
        "util": {s: (prefill_util(s), decode_kv(s)) for s in SHAPES},
        "r_requests": statistics.correlation([a for a, _, _ in rows], waits),
        "r_tokens": statistics.correlation([k for _, k, _ in rows], waits),
        "at2": (min(at2), max(at2)),
        "kv_up_rate": rate_at("reasoning", KV_UP),
        "queue_rate": rate_at("reasoning", 1.0),
        "ref_service": (ref.REQUEST_PREFILL_SEC, ref.REQUEST_DECODE_SEC),
        "ref_has_kv": any("kv" in name.lower() for name in dir(ref)),
    }


def verify(result):
    rag, rsn = result["util"]["rag"], result["util"]["reasoning"]
    lo, hi = result["at2"]
    return [
        practice.Check(
            "ANSWER: prefill on queued prompt tokens, decode on KV-cache utilization",
            all(
                [
                    round(rag[0], 2) == 0.75,
                    round(rag[1], 2) == 0.24,
                    round(rsn[0], 2) == 0.05,
                    round(rsn[1], 2) == 1.46,
                ]
            ),
            f"at {RATE} req/s: rag prefill {rag[0]:.0%} / KV {rag[1]:.0%}; "
            f"reasoning prefill {rsn[0]:.0%} / KV {rsn[1]:.0%}",
        ),
        practice.Check(
            "FINDING: queued requests do not predict TTFT when prompts vary; queued "
            "tokens do exactly",
            all(
                [
                    round(result["r_requests"], 2) == 0.80,
                    round(result["r_tokens"], 6) == 1.0,
                    round(lo, 2) == 0.03,
                    round(hi, 2) == 1.97,
                ]
            ),
            f"corr(wait, requests ahead) {result['r_requests']:.3f}, corr(wait, tokens "
            f"ahead) {result['r_tokens']:.3f}; with 2 ahead the wait is {lo:.2f}-{hi:.2f}s",
        ),
        practice.Check(
            "FINDING: on decode pods a queue signal fires only after KV is full",
            all(
                [
                    round(result["kv_up_rate"], 2) == 0.82,
                    round(result["queue_rate"], 2) == 1.02,
                    result["ref_service"] == (0.6, 1.8),
                    not result["ref_has_kv"],
                ]
            ),
            f"reasoning shape: KV crosses {KV_UP:.0%} at {result['kv_up_rate']:.2f} req/s, "
            f"a queue forms at {result['queue_rate']:.2f}; the reference charges a fixed "
            f"{result['ref_service']}s per request and models no KV",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
