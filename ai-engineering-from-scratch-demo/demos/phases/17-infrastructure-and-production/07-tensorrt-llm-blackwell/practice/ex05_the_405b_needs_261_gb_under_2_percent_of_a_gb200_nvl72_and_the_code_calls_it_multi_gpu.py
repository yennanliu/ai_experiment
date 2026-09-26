"""Exercise 5 — the 405B needs 261 GB, under 2% of a GB200 NVL72, and the code calls it multi-GPU.

    Compute the HBM needed for a 405B model at NVFP4 weights + FP8 KV cache at
    128k context. Does it fit on a single GB200 NVL72 node?

Reading of the exercise: one 128k-token sequence (131,072, Llama 3.1's
`max_position_embeddings`) is sized with the reference `hbm_footprint_gb` on
the module's GB200 NVL72 stack, which uses 4-bit weights and 8-bit KV. A "node"
is read as the whole 72-GPU NVLink rack. The same sequence is then sized with
Llama 3.1 405B's real shape: 126 layers, 8 KV heads, head_dim 128, from its
config.json. The rack's capacity is taken two ways: the module's 192 GB per GPU,
and NVIDIA's published 13.4 TB (372 GB per 2-GPU superchip).

**ANSWER: 260.9 GB, and it fits with room to spare: 1.9% of a 13.4 TB rack.**
The weights are 202.5 GB and the FP8 KV cache is 58.4 GB. The model does not
fit on one Blackwell GPU, since the weights alone exceed 192 GB. It does fit on
one GB200 superchip (372 GB). The rest of the rack holds 224 more 128k
sequences.

**FINDING: the module's layer count is 1.73x the real model's.** It invents
`layers = 64 * sqrt(active / 35)`, which gives 217.7 layers for 405B.
Llama 3.1 405B has 126. At the real shape the KV cache is 33.8 GB, the total is
236.3 GB, and the rack holds 390 sequences.

**FINDING: `print_stack` would label this "(multi-GPU)" on the GB200 NVL72
row.** The stack's `hbm_gb` is 192, a single GPU's HBM, not the rack's. The fit
test is against one GPU, so the row the lesson names as rack-scale fails it.

**FINDING: at 128k the KV cache is 22% of the bytes each decode step reads,
and the throughput model ignores it.** Every generated token reads the whole
cache as well as the weights, so the bytes per step are 1.29x what
`decode_throughput` charges. That function has no context-length input.

Structure: `real_kv_gb()` is the reference's own KV formula with the real
shape; `solve()` reads everything else from the reference.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "07-tensorrt-llm-blackwell"
PARAMS, SEQ = 405, 131_072
LAYERS, KV_HEADS, HEAD_DIM = 126, 8, 128  # Llama 3.1 405B config.json
RACK_GB, SUPERCHIP_GB, GPUS = 13_400, 372, 72  # NVIDIA GB200 NVL72 spec


def real_kv_gb(kv_bits):
    return LAYERS * 2 * KV_HEADS * HEAD_DIM * SEQ * kv_bits / 8 / 1e9


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    nvl = ref.STACKS[4]
    weights, kv = ref.hbm_footprint_gb(PARAMS, PARAMS, SEQ, nvl)
    real_kv = real_kv_gb(nvl.kv_bits)
    return {
        "weights": weights,
        "kv": round(kv, 1),
        "total": round(weights + kv, 1),
        "share": round((weights + kv) / RACK_GB, 3),
        "one_gpu": weights <= nvl.hbm_gb,
        "superchip": weights + kv <= SUPERCHIP_GB,
        "extra": math.floor((RACK_GB - weights - kv) / kv),
        "code_layers": round(64 * (PARAMS / 35) ** 0.5, 1),
        "real_kv": round(real_kv, 1),
        "real_total": round(weights + real_kv, 1),
        "real_seqs": math.floor((RACK_GB - weights) / real_kv),
        "stack_gb": nvl.hbm_gb,
        "code_rack_gb": nvl.hbm_gb * GPUS,
        "kv_share": round(kv / (weights + kv), 2),
        "kv_inputs": ref.decode_throughput.__code__.co_varnames[
            : ref.decode_throughput.__code__.co_argcount
        ],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: 260.9 GB, and it fits: 1.9% of a 13.4 TB rack",
            (r["weights"], r["kv"], r["total"], r["share"], r["extra"])
            == (202.5, 58.4, 260.9, 0.019, 224)
            and (r["one_gpu"], r["superchip"]) == (False, True),
            f"weights {r['weights']} + FP8 KV {r['kv']} = {r['total']} GB of {RACK_GB}; "
            f"no single GPU, one {SUPERCHIP_GB} GB superchip, {r['extra']} more sequences",
        ),
        practice.Check(
            "FINDING: the module's layer count is 1.73x the real model's",
            r["code_layers"] == 217.7
            and round(r["code_layers"] / LAYERS, 2) == 1.73
            and r["real_kv"] == 33.8
            and r["real_seqs"] == 390,
            f"{r['code_layers']} layers vs {LAYERS}; real KV {r['real_kv']} GB, total "
            f"{r['real_total']} GB, {r['real_seqs']} 128k sequences per rack",
        ),
        practice.Check(
            "FINDING: print_stack would label this multi-GPU on the GB200 NVL72 row",
            r["total"] > r["stack_gb"],
            f"hbm_gb = {r['stack_gb']} is one GPU; the rack is {r['code_rack_gb']} GB by the "
            f"module's own per-GPU number and {RACK_GB} GB by NVIDIA's",
        ),
        practice.Check(
            "FINDING: at 128k the KV cache is 22% of each decode step's reads, and is ignored",
            r["kv_share"] == 0.22 and r["kv_inputs"] == ("active_b", "stack"),
            f"KV is {r['kv_share']:.0%} of weights+KV; decode_throughput takes only "
            f"{r['kv_inputs']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
