"""Exercise 5 — routing leaves the KV transfer alone and makes decode read 8x more weight per token than dense at batch 64.

    MoE expert routing changes KV access patterns. How does disaggregation
    behave with MoE that activates different experts per token?

Reading of the exercise: the MoE is DeepSeek-V3 as its config.json gives it
(fetched 2026-09-26: 61 layers, the first 3 dense, 256 routed experts, top-8,
1 shared, expert FFN 7168 x 2048, MLA KV of 512 + 64 per layer) with 671B
total / 37B active parameters, at 1 byte per value to match the lesson's FP8.
Routing is simulated with a seeded uniform top-8 per token per MoE layer, and
each phase is measured by what it has to read: the KV it hands over, and the
expert weights one forward pass touches at a given batch.

**ANSWER: the KV handoff does not change; what changes is decode, which
reads more expert weight as its batch grows.** KV is attention state, and
every token runs attention whatever experts it picks -- the experts are FFN
weights. So the transfer is set by the attention design: DeepSeek-V3's MLA
stores 61 x 576 = 35,136 B/token, and through the reference `ms_disaggregated`
a 4K prompt moves 143.9 MB in 1.44 ms over RDMA, 28% of the lesson's 70B
figure. Weight reads go the other way. A 4096-token prefill touches all 256
experts in every layer, 128 tokens each, so the prefill pool behaves like a
dense model on 37B active parameters and stays compute-bound. A decode step
touches 8 experts per layer at batch 1, 223.7 at batch 64 and 256.0 at 256:
37.0 -> 587.9 -> 670.4 GB per step, which is the whole model.

**FINDING: at batch 64 MoE decode reads 8.4x more weight per token than
dense 70B.** Per generated token MoE reads 37.0 GB at batch 1 against dense
70B's 70, but 9.19 GB at batch 64 against 1.09, because each expert then
serves 2.3 tokens instead of the prefill's 128. The decode pool
amortizes weights only with very large batches spread over many GPUs (wide
expert parallelism); 671 GB of FP8 weights is 4.8 H200s before any KV. The
two phases want opposite layouts, which is what disaggregation lets each pool
pick separately.

**FINDING: the lesson's simulator cannot express MoE at all.** It has one
decode rate per pool and no batch, expert or attention-type term. MoE enters
only by overriding `KV_BYTES_PER_TOKEN_70B_FP8`, and that constant is set by
the attention design, not by the routing.

Structure: `touched()` samples the router; `step_bytes()` turns touched
experts into bytes read.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "17-disaggregated-prefill-decode"
EXPERTS, TOP_K, LAYERS, DENSE_LAYERS = 256, 8, 61, 3
MOE_LAYERS = LAYERS - DENSE_LAYERS
EXPERT_BYTES = 3 * 7168 * 2048  # gate, up, down projections
ACTIVE, TOTAL, MLA_KV = 37e9, 671e9, LAYERS * (512 + 64)
NON_ROUTED = ACTIVE - TOP_K * MOE_LAYERS * EXPERT_BYTES
BATCHES, PROMPT, DENSE, H200 = (1, 8, 64, 256), 4096, 70e9, 141e9


def touched(rng, batch):
    """Distinct routed experts summed over the MoE layers for one forward pass."""
    total = 0
    for _ in range(MOE_LAYERS):
        hit = set()
        for _ in range(batch):
            hit.update(rng.sample(range(EXPERTS), TOP_K))
        total += len(hit)
    return total


def step_bytes(distinct):
    return NON_ROUTED + distinct * EXPERT_BYTES


def mla_transfer(ref):
    original = ref.KV_BYTES_PER_TOKEN_70B_FP8
    ref.KV_BYTES_PER_TOKEN_70B_FP8 = MLA_KV
    try:
        return round(ref.ms_disaggregated(PROMPT, 0) - PROMPT / ref.PREFILL_TOK_PER_MS, 2)
    finally:
        ref.KV_BYTES_PER_TOKEN_70B_FP8 = original


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(0)
    decode = {b: touched(rng, b) for b in BATCHES}
    return {
        "kv": MLA_KV, "kv_mb": round(PROMPT * MLA_KV / 1e6, 1), "rdma": mla_transfer(ref),
        "kv_share": round(MLA_KV / ref.KV_BYTES_PER_TOKEN_70B_FP8, 2),
        "prefill": touched(rng, PROMPT) / MOE_LAYERS,
        "per_layer": {b: round(d / MOE_LAYERS, 1) for b, d in decode.items()},
        "step_gb": {b: round(step_bytes(d) / 1e9, 1) for b, d in decode.items()},
        "per_token": {b: round(step_bytes(d) / b / 1e9, 2) for b, d in decode.items()},
        "dense": {b: round(DENSE / b / 1e9, 2) for b in BATCHES},
        "tokens_per_expert": round(64 * TOP_K * MOE_LAYERS / decode[64], 1),
        "h200s": round(TOTAL / H200, 1),
        "ref_terms": [n for n in dir(ref) if any(t in n.lower() for t in ("expert", "batch", "moe"))],
    }


def verify(result):
    per_token, dense, steps = result["per_token"], result["dense"], result["step_gb"]
    return [
        practice.Check(
            "ANSWER: the KV handoff does not change; decode reads more expert weight as its "
            "batch grows",
            all([result["kv"] == 35136, result["rdma"] == 1.44, result["kv_share"] == 0.28,
                 result["prefill"] == EXPERTS, steps[1] == 37.0, steps[256] == 670.4]),
            f"MLA KV {result['kv']} B/token, 4K prompt {result['kv_mb']} MB in "
            f"{result['rdma']} ms RDMA; prefill touches {result['prefill']:.0f} experts per "
            f"layer; decode {result['per_layer']} per layer, {steps} GB per step",
        ),
        practice.Check(
            "FINDING: at batch 64 MoE decode reads 8.4x more weight per token than dense 70B",
            per_token[1] < dense[1] and round(per_token[64] / dense[64], 1) == 8.4
            and result["tokens_per_expert"] == 2.3,
            f"GB per token MoE {per_token} vs dense {dense}; {result['tokens_per_expert']} "
            f"tokens per expert at batch 64; weights alone fill {result['h200s']} H200s",
        ),
        practice.Check(
            "FINDING: the lesson's simulator cannot express MoE at all",
            result["ref_terms"] == [],
            "the reference module has no expert, batch or MoE name; MoE enters only through "
            "the KV-bytes constant, which routing does not change",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
