"""Exercise 1 — INT4 gives exactly 4.000x, and leaves 41% of the card untouched.

    Modify the KV cache profiler to compare FP16 vs FP8 vs INT4 KV cache
    quantization. For Llama 3 70B at 4K context, compute the max concurrent
    users for each on 4xA100-80GB. KV quantization to INT4 should roughly 4x the
    user capacity.

Reading of the exercise: the profiler is the lesson's own `memory_budget`, which
already takes `kv_dtype_bytes`, so the three arms are one call each at 2, 1 and
0.5 bytes; the shape is `MODEL_CONFIGS["Llama-3-70B"]` and the card budget is
4 x 80 GB. The model-weight arm is run beside it, because the exercise's
quantiser is applied to one of the two terms in that budget and the other is
larger.

**ANSWER: 126, 252, 504 users -- exactly 2.000x and 4.000x.** Not "roughly".
`max_users_at_4k` is `available_for_kv / (2 x L x H_kv x d x bytes) / 4096`, and
`available_for_kv` does not contain `bytes` at all, so the capacity is exactly
proportional to `1 / kv_dtype_bytes`. The prediction is a restatement of the
formula, and a real server's fragmentation, paged-block rounding and activation
working set are not in it to spoil the ratio.

**FINDING: the exercise quantises the smaller of the two terms.** Of the 320 GB,
the fp16 weights take **130.4 GB** and the flat 10% overhead 32 GB, leaving
157.6 GB for KV. The weights are 41% of the card and the exercise never touches
them: quantising the *model* to INT4 instead takes users from 126 to **204**,
1.62x, and the two are independent -- both at once is 4x on top of 1.62x.

**FINDING: the 10% overhead scales with the card, not with the work.**
`memory_budget` books `gpu_memory_gb * 0.1`, so adding GPUs adds overhead in
proportion. At 4 x A100 that is 32 GB reserved for a quantity -- activations,
fragmentation, the CUDA context -- that does not grow with the card.

**MECHANISM: KV bytes per token is `2 x layers x kv_heads x head_dim x bytes`.**
80 layers, 8 KV heads and 128 head dims give 320 KB per token at fp16, 80 KB at
INT4. At 4K context that is 1.25 GB per user against 0.31 GB, which is what
turns 126 seats into 504.

Structure: `arm` is one `memory_budget` call at one pair of dtypes; `users` reads
the seat count the profiler reports at 4K.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "12-inference-optimization"
MODEL, GPU_GB, CONTEXT = "Llama-3-70B", 320, 4096
KV_DTYPES = (("fp16", 2.0), ("fp8", 1.0), ("int4", 0.5))


def arm(ref, config, kv_bytes, model_bytes=2.0):
    """One point of the profiler: the lesson's own budget at one pair of dtypes."""
    return ref.memory_budget(config, GPU_GB, model_dtype_bytes=model_bytes,
                             kv_dtype_bytes=kv_bytes)


def per_token_kb(config, kv_bytes):
    return 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_bytes / 1024


def field(arms, key):
    return {name: row[key] for name, row in arms.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    config = ref.MODEL_CONFIGS[MODEL]
    kv_arms = {name: arm(ref, config, b) for name, b in KV_DTYPES}
    weight_arms = {name: arm(ref, config, 2.0, model_bytes=b) for name, b in KV_DTYPES}
    per_token = {name: per_token_kb(config, b) for name, b in KV_DTYPES}
    baseline = kv_arms["fp16"]
    return {"users": field(kv_arms, "max_users_at_4k"),
            "weight_users": field(weight_arms, "max_users_at_4k"),
            "kv_gb": field(kv_arms, "available_for_kv_gb"),
            "per_token_kb": per_token,
            "model_gb": baseline["model_memory_gb"],
            "overhead_gb": baseline["overhead_gb"],
            "gb_per_user": {name: kb * CONTEXT / 1024 ** 2 for name, kb in per_token.items()}}


def row(values, fmt):
    return ", ".join(f"{name} {format(value, fmt)}" for name, value in values.items())


def verify(result):
    users, kv_gb = result["users"], result["kv_gb"]
    weights, per_token = result["weight_users"], result["per_token_kb"]
    ratios = {name: seats / users["fp16"] for name, seats in users.items()}
    return [
        practice.Check(
            "ANSWER: 126, 252, 504 users -- exactly 2.000x and 4.000x, not roughly",
            ratios["fp8"] == 2.0 and ratios["int4"] == 4.0,
            "at 4K context on 4xA100-80GB the profiler seats " + row(users, "d")
            + ", which is " + row(ratios, ".3f")
            + f". max_users_at_4k is available_for_kv / per_token / {CONTEXT} and "
            f"available_for_kv is {row(kv_gb, '.1f')} GB -- the same number in all three arms, "
            "because the KV dtype appears only in the denominator. The prediction restates the "
            "formula, and nothing a real server does is in it to spoil the ratio",
        ),
        practice.Check(
            "FINDING: the exercise quantises the smaller of the two terms",
            result["model_gb"] > max(kv_gb.values()) - 30 and weights["int4"] > users["fp16"],
            f"of the {GPU_GB} GB, the fp16 weights take {result['model_gb']:.1f} GB and the flat "
            f"10% overhead {result['overhead_gb']:.1f} GB, leaving "
            f"{kv_gb['fp16']:.1f} GB for KV. The weights are "
            f"{100 * result['model_gb'] / GPU_GB:.0f}% of the card and the exercise never touches "
            "them: quantising the model instead, at fp16 KV, seats " + row(weights, "d")
            + f", so INT4 weights alone buy {weights['int4'] / users['fp16']:.2f}x. The two are "
            "independent terms and the exercise sweeps one of them",
        ),
        practice.Check(
            "FINDING: the 10% overhead scales with the card, not with the work",
            abs(result["overhead_gb"] - 0.1 * GPU_GB) < 0.1,
            f"memory_budget books gpu_memory_gb * 0.1, so at {GPU_GB} GB that is "
            f"{result['overhead_gb']:.1f} GB reserved before anything is served, and adding GPUs "
            "adds overhead in proportion. Activations, fragmentation and the CUDA context do not "
            "grow with the card, so the one term standing in for all of them is the one term "
            "modelled as a fraction of it",
        ),
        practice.Check(
            "MECHANISM: 320 KB per token at fp16 is 1.25 GB per user at 4K",
            per_token["fp16"] == 320.0 and per_token["int4"] == 80.0,
            "KV bytes per token is 2 x layers x kv_heads x head_dim x bytes, so 80 layers, 8 KV "
            "heads and 128 dims give " + row(per_token, ".0f")
            + " KB per token, which at 4K context is "
            + row(result["gb_per_user"], ".2f")
            + " GB per user. That is what turns 126 seats into 504",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
