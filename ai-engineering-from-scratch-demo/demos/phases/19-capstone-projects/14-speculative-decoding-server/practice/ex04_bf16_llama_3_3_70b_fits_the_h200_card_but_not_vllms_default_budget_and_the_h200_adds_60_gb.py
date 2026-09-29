"""Exercise 4 -- BF16 Llama 3.3 70B fits the H200 card but not vLLM's default budget, and the H200 buys 60 GB of KV.

    Extend to H200 (141 GB). Report the model-size-per-replica headroom gained and whether you can serve an unquantized Llama 3.3 70B.

Reading of the exercise: this is memory arithmetic, and no GPU is needed to
do it. The lesson's `code/main.py` has no memory model, so the solution
reads the lesson's claims from its doc and sizes them. The inputs, all read
on 2026-09-29, are:

- Llama-3.3-70B's shape from its config.json: 80 layers, hidden 8192, MLP
  28672, 64 query heads and 8 KV heads of 128, vocabulary 128256, untied
  embeddings
  (https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/raw/main/config.json,
  an ungated mirror, because Meta's repo returns 401).
- The cards' real capacities as nvidia-smi reports them: 81,559 MiB for the
  H100 and 143,771 MiB for the H200
  (https://thundergolfer.com/blog/nvidia-gpu-memory-capacity and
  https://github.com/NVIDIA/cuda-samples/issues/311).
- vLLM's `--gpu-memory-utilization` default of 0.92
  (https://docs.vllm.ai/en/latest/configuration/engine_args/).

"FP8" means FP8 linear layers with BF16 embeddings and norms, which is how
FP8 checkpoints ship. KV cache is BF16. Activations and CUDA graphs are left
out, so every pool below is an upper bound.

**ANSWER: the H200 adds 60.01 GB per replica, and unquantized Llama 3.3 70B
fits the card but not vLLM's default budget.** The model has 70,553,706,496
parameters: 141.11 GB in BF16 and 72.66 GB in FP8. vLLM's default budget is
78.68 GB on the H100 and 138.69 GB on the H200. With FP8 weights, the KV pool
grows from 6.02 GB (18,380 tokens at 327,680 B per token) to 66.04 GB
(201,532 tokens), which is 11x. In BF16 the weights exceed the H200's default
budget by 2.41 GB. They need a utilisation of at least 0.936 before any KV
cache is allocated. At 0.95 the pool is 2.11 GB (6,438 tokens), and at 0.98
it is 6.63 GB (20,240 tokens). So a single H200 can serve BF16 only with the
knob raised and room for about as many tokens as an FP8 H100 holds. At
serving batch sizes, BF16 still means two GPUs.

**FINDING: "141 GB" is closer to GiB, and read literally it gives the wrong
answer.** 143,771 MiB is 150.75 GB, or 140.4 GiB. Against a literal 141 GB,
the 141.11 GB of BF16 weights would not fit by 0.11 GB. On the real card
they fit with 9.64 GB to spare, before vLLM's budget takes 8% back.

**FINDING: the lesson's 1xH100 FP8 deployment cannot hold its own batch-32
report.** The 18,380-token pool gives 574 tokens per sequence at batch 32.
That is less than the 620-token response in the lesson's Use It block, and
it leaves nothing for the prompt. Batch 8 fits (8 x 620 = 4,960 tokens), but
the rubric requires p99 at batch 32.

Structure: `params()` counts from the config; `weights()` prices them per
format; `pool()` is what vLLM leaves for KV cache.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "14-speculative-decoding-server"
# Llama-3.3-70B-Instruct config.json
CFG = {"hidden": 8192, "inter": 28672, "layers": 80, "heads": 64, "kv_heads": 8, "head_dim": 128, "vocab": 128256}
MIB = 1 << 20
GPUS = {"H100": 81_559 * MIB, "H200": 143_771 * MIB}      # nvidia-smi totals
UTIL = 0.92                                               # vLLM --gpu-memory-utilization default


def params(c):
    """(linear, embed + lm_head, norm) parameter counts; embeddings are untied."""
    attn = 2 * c["hidden"] * c["heads"] * c["head_dim"] + 2 * c["hidden"] * c["kv_heads"] * c["head_dim"]
    linear = c["layers"] * (attn + 3 * c["hidden"] * c["inter"])
    return linear, 2 * c["vocab"] * c["hidden"], (2 * c["layers"] + 1) * c["hidden"]


def weights(c, linear_bytes):
    linear, embed, norm = params(c)
    return linear * linear_bytes + (embed + norm) * 2


def kv_per_token(c, nbytes=2):
    return 2 * c["layers"] * c["kv_heads"] * c["head_dim"] * nbytes


def pool(gpu, weight_bytes, util=UTIL):
    """KV-cache bytes and tokens left under vLLM's budget (activations and CUDA graphs ignored)."""
    free = util * GPUS[gpu] - weight_bytes
    return round(free / 1e9, 2), max(0, int(free // kv_per_token(CFG)))


def solve():
    doc = parity.doc_text(PHASE, LESSON)
    total = sum(params(CFG))
    bf16, fp8 = weights(CFG, 2), weights(CFG, 1)
    use_it = re.search(r"bs=(\d+).*?\((\d+) tokens\)", doc, re.S)
    h100_fp8 = pool("H100", fp8)
    return {
        "doc_h200_gb": int(re.search(r"H200 \((\d+) GB\)", doc).group(1)),
        "params": total, "bf16_gb": round(bf16 / 1e9, 2), "fp8_gb": round(fp8 / 1e9, 2),
        "card_gb": {g: round(b / 1e9, 2) for g, b in GPUS.items()},
        "card_gib": {g: round(b / (1 << 30), 1) for g, b in GPUS.items()},
        "budget_gb": {g: round(UTIL * b / 1e9, 2) for g, b in GPUS.items()},
        "fp8_pool": {g: pool(g, fp8) for g in GPUS},
        "bf16_pool": {u: pool("H200", bf16, u) for u in (UTIL, 0.95, 0.98)},
        "min_util_bf16": round(bf16 / GPUS["H200"], 3),
        "kv_token": kv_per_token(CFG),
        "use_it": tuple(map(int, use_it.groups())),
        "bs32_tokens": h100_fp8[1] // 32,
    }


def verify(result):
    r = result
    gained = round(r["budget_gb"]["H200"] - r["budget_gb"]["H100"], 2)
    return [
        practice.Check(
            "ANSWER: +60.01 GB per replica; BF16 fits the H200 card but not vLLM's 0.92 budget",
            (r["params"], r["bf16_gb"], r["fp8_gb"], r["budget_gb"], gained, r["fp8_pool"], r["kv_token"])
            == (70_553_706_496, 141.11, 72.66, {"H100": 78.68, "H200": 138.69}, 60.01,
                {"H100": (6.02, 18_380), "H200": (66.04, 201_532)}, 327_680)
            and r["bf16_pool"] == {0.92: (-2.41, 0), 0.95: (2.11, 6_438), 0.98: (6.63, 20_240)}
            and r["min_util_bf16"] == 0.936,
            f"{r['params']:,} params = {r['bf16_gb']} GB BF16 / {r['fp8_gb']} GB FP8; budgets {r['budget_gb']} "
            f"(+{gained} GB); FP8 KV pools {r['fp8_pool']}; BF16 on H200 by utilisation {r['bf16_pool']}, "
            f"needs >= {r['min_util_bf16']}",
        ),
        practice.Check(
            "FINDING: '141 GB' is closer to GiB, and read literally it gives the wrong answer",
            (r["doc_h200_gb"], r["card_gb"]["H200"], r["card_gib"]["H200"]) == (141, 150.75, 140.4)
            and r["bf16_gb"] > r["doc_h200_gb"] and round(r["card_gb"]["H200"] - r["bf16_gb"], 2) == 9.64,
            f"the exercise says {r['doc_h200_gb']} GB; nvidia-smi's 143,771 MiB = {r['card_gb']['H200']} GB = "
            f"{r['card_gib']['H200']} GiB; BF16 weights {r['bf16_gb']} GB",
        ),
        practice.Check(
            "FINDING: the lesson's 1xH100 FP8 deployment cannot hold its own batch-32 report",
            (r["use_it"], r["bs32_tokens"]) == ((8, 620), 574) and r["bs32_tokens"] < r["use_it"][1],
            f"Use It: bs={r['use_it'][0]}, {r['use_it'][1]}-token response; batch 32 gets {r['bs32_tokens']} "
            "tokens per sequence from the H100 FP8 pool",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
