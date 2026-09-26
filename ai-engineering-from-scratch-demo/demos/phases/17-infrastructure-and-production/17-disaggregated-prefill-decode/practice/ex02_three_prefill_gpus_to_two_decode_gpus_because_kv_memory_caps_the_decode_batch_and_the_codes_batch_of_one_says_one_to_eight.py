"""Exercise 2 — three prefill GPUs to two decode GPUs, because KV memory caps the decode batch; the code's batch of one says one to eight.

    Design the prefill pool and decode pool for a RAG service with P99
    prefix length 8K, output 300.

Reading of the exercise: size both pools for the P99 request (8192-token
prefix, 300 output tokens) at an assumed 10 requests/s, since the exercise
gives no rate. Prefill uses the lesson's own `PREFILL_TOK_PER_MS` and its KV
transfer via `ms_disaggregated`. The code has no batch, so decode is
modelled as a memory-bound step on an H200 (141 GB, 4.8 TB/s -- the "H200-like"
GPU the code names): step = (70 GB weights + batch x KV read) / bandwidth,
with the batch capped by how many requests' KV fits in the 71 GB beside the
weights. The prefill pool is held to 70% utilization for P99 headroom.

**ANSWER: 3 prefill GPUs and 2 decode GPUs per 10 req/s.** Each prefill takes
204.8 GPU-ms, so 10 req/s keeps 2.05 GPUs busy -- 3 at <= 70%. A decode GPU
holds 66 requests' KV (8492 tokens x 125,000 B = 1.06 GB each), a full-batch
step takes 28.9 ms, and 10 req/s needs 86.8 requests in flight: 2 GPUs.
P99 TTFT is 204.8 ms prefill + 10.24 ms RDMA transfer = 215 ms before
queueing; TPOT is 28.9 ms.

**FINDING: the code's batch-of-one decode sizes the pools 1 : 8 the other
way.** `ms_disaggregated` decodes at 0.18 tok/ms, 1666.7 GPU-ms per request
against 204.8 ms of prefill -- 8.14 decode GPUs per prefill GPU. Batched to
the KV cap, a request costs 131.5 decode GPU-ms, and prefill needs 1.56x the
decode pool's GPU time. For a RAG workload the prefill pool is the larger
one, which agrees with the skill's "2 prefill : 1 decode for RAG-heavy".

**FINDING: the decode batch is set by KV memory, and the lesson's KV size is
76% of Llama 3.3 70B's.** 80 layers x 8 KV heads x 128 x 2 (K and V) at FP8
is 163,840 B/token against the code's 125,000. At the real size a decode GPU
holds 51 requests, not 66, and the prefill:decode GPU-time ratio falls to
1.2 -- the pool still fits in 2 decode GPUs at this rate, with less headroom.
Batching also costs latency the code cannot show: full-batch TPOT is 28.9 ms
against the code's 5.56 ms.

Structure: `decode()` is the memory-bound decode step; `design()` sizes both
pools from it and the reference constants.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "17-disaggregated-prefill-decode"
PREFIX, OUTPUT, QPS, HEADROOM = 8192, 300, 10, 0.7
WEIGHTS, HBM, BW = 70e9, 141e9, 4.8e12  # 70B FP8 on an H200
LLAMA_KV = 80 * 8 * 128 * 2  # layers x KV heads x head dim x (K, V), 1 byte each


def decode(kv_per_token):
    """(max batch, step seconds at that batch, decode GPU-ms per request)."""
    batch = int((HBM - WEIGHTS) // ((PREFIX + OUTPUT) * kv_per_token))
    step = (WEIGHTS + batch * (PREFIX + OUTPUT / 2) * kv_per_token) / BW
    return batch, step, OUTPUT * step / batch * 1000


def design(ref, kv_per_token):
    batch, step, decode_ms = decode(kv_per_token)
    prefill_ms = PREFIX / ref.PREFILL_TOK_PER_MS
    in_flight = QPS * OUTPUT * step
    return {
        "prefill_gpus": math.ceil(QPS * prefill_ms / 1000 / HEADROOM),
        "decode_gpus": math.ceil(in_flight / batch), "batch": batch,
        "tpot_ms": round(step * 1000, 2), "in_flight": round(in_flight, 1),
        "ratio": round(prefill_ms / decode_ms, 2), "decode_ms": round(decode_ms, 1),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prefill_ms = PREFIX / ref.PREFILL_TOK_PER_MS
    return {
        "lesson": design(ref, ref.KV_BYTES_PER_TOKEN_70B_FP8), "llama": design(ref, LLAMA_KV),
        "prefill_ms": prefill_ms,
        "transfer_ms": round(ref.ms_disaggregated(PREFIX, 0) - prefill_ms, 2),
        "batch1_ms": OUTPUT / ref.DECODE_TOK_PER_MS_DECODE_GPU,
        "code_tpot": round(1 / ref.DECODE_TOK_PER_MS_DECODE_GPU, 2),
        "kv_share": round(ref.KV_BYTES_PER_TOKEN_70B_FP8 / LLAMA_KV, 2),
    }


def verify(result):
    lesson, llama = result["lesson"], result["llama"]
    batch1 = result["batch1_ms"] / result["prefill_ms"]
    return [
        practice.Check(
            "ANSWER: 3 prefill GPUs and 2 decode GPUs per 10 req/s",
            all([(lesson["prefill_gpus"], lesson["decode_gpus"]) == (3, 2),
                 lesson["batch"] == 66, lesson["tpot_ms"] == 28.92,
                 result["transfer_ms"] == 10.24]),
            f"{lesson}; TTFT {result['prefill_ms']} + {result['transfer_ms']} ms RDMA",
        ),
        practice.Check(
            "FINDING: the code's batch-of-one decode sizes the pools 1 : 8 the other way",
            round(batch1, 2) == 8.14 and lesson["ratio"] == 1.56,
            f"batch 1: {result['batch1_ms']:.1f} decode GPU-ms per {result['prefill_ms']} "
            f"prefill ms ({batch1:.2f} : 1); batched: {lesson['decode_ms']} ms, prefill "
            f"{lesson['ratio']}x the decode pool's time",
        ),
        practice.Check(
            "FINDING: the decode batch is set by KV memory, and the lesson's KV size is 76% "
            "of Llama 3.3 70B's",
            all([result["kv_share"] == 0.76, llama["batch"] == 51, llama["ratio"] == 1.2,
                 llama["decode_gpus"] == 2, lesson["tpot_ms"] > 5 * result["code_tpot"]]),
            f"{LLAMA_KV} B/token: batch {llama['batch']}, ratio {llama['ratio']}, "
            f"{llama['decode_gpus']} decode GPUs; full-batch TPOT {lesson['tpot_ms']} ms "
            f"against the code's {result['code_tpot']} ms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
