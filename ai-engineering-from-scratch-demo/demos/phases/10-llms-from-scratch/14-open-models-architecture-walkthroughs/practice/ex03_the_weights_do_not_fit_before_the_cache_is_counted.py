"""Exercise 3 — at BF16 the subtraction the exercise asks for goes negative by 112 GB.

    Compute the KV cache for Llama 3 405B at 128k context in FP8 and BF16. At
    FP8 it is half the BF16 number. How many parallel sequences can you serve on
    a single 8xH100 node (80GB each = 640GB total, minus weight memory)?

Reading of the exercise: `CONFIGS` stops at the 70B, so the 405B shape is added
-- hidden 16384, 126 layers, 128 query heads, 8 KV heads -- and run through the
lesson's own `analyze`, whose `kv_cache_bytes_bf16` already uses
`max_position_embeddings`, which is 131072. Both dtypes are then priced against
the node the exercise names, and the weights are priced at both dtypes too,
because "minus weight memory" does not say which dtype the weights are in.

**ANSWER: at BF16 weights the model does not fit, and at FP8 weights it serves 8
sequences.**

    weights bf16   752.0 GB of 640    ->  -112.0 GB left, before any cache
    weights fp8    376.0 GB of 640    ->   264.0 GB left  ->  8 seq at fp8 KV, 4 at bf16 KV

The KV cache is **63.0 GB per sequence** at BF16 and **31.5 GB** at FP8 -- and
"at FP8 it is half the BF16 number" is exact, because `2 * layers * kv_heads *
head_dim * seq * bytes` is linear in `bytes` and nothing else in it changes.

**FINDING: the question presumes a subtraction that goes negative.** "640GB
total, minus weight memory" reads as though the remainder were positive. A
405B model at BF16 is 752 GB before a single token is cached, so the first
decision on this node is quantising the *weights*, which the exercise does not
mention and Lesson 11 spent five exercises on.

**FINDING: one sequence at 128k costs 17% of what is left.** 63.0 GB of KV for
one user against 264 GB of headroom is a ratio no amount of batching improves:
the cache is per-sequence and linear in context, so the node serves 8 users at
128k or 1,024 at 1k, on the same memory.

**MECHANISM: 504 KB per token is a GQA number.** `2 x 126 layers x 8 KV heads x
128 dims x 2 bytes` is 504 KB. Without GQA -- 128 KV heads instead of 8 -- it
would be 8.06 MB per token and **1,008 GB** for a single 128k sequence, which is
more than the node. The 405B is servable at long context because of one config
field.

Structure: `shape` is the 405B config the lesson stops short of; `seats` prices
one weight dtype against one cache dtype on the node.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "14-open-models-architecture-walkthroughs"
NODE_GB, CONTEXT = 640, 131072
REPORTED_405B = 405.9e9
GIB = 1024 ** 3


def shape(ref):
    """The 405B config `CONFIGS` stops short of, on the 70B's shared knobs."""
    return dict(ref.CONFIGS["llama3-70b"], hidden_size=16384, intermediate_size=53248,
                num_hidden_layers=126, num_attention_heads=128, num_key_value_heads=8)


def kv_per_token(config, dtype_bytes):
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    return 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * dtype_bytes


def seats(params, config, weight_bytes, kv_bytes):
    """Sequences of `CONTEXT` tokens that fit after the weights, at one pair of dtypes."""
    weights_gb = params * weight_bytes / GIB
    left = NODE_GB - weights_gb
    per_seq = kv_per_token(config, kv_bytes) * CONTEXT / GIB
    return {"weights_gb": weights_gb, "left_gb": left, "per_seq_gb": per_seq,
            "seats": int(left / per_seq) if left > 0 else 0}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    config = shape(ref)
    counted = ref.analyze("llama3-405b", config)
    mha = dict(config, num_key_value_heads=config["num_attention_heads"])
    return {
        "params": counted.total_params,
        "reported": REPORTED_405B,
        "per_token": {"bf16": kv_per_token(config, 2), "fp8": kv_per_token(config, 1)},
        "arms": {"bf16 weights, bf16 KV": seats(counted.total_params, config, 2, 2),
                 "fp8 weights, bf16 KV": seats(counted.total_params, config, 1, 2),
                 "fp8 weights, fp8 KV": seats(counted.total_params, config, 1, 1)},
        "mha_per_seq_gb": kv_per_token(mha, 2) * CONTEXT / GIB,
        "scheme": counted.attention_scheme,
    }


def verify(result):
    arms, per_token = result["arms"], result["per_token"]
    dense, cheap, cheapest = (arms["bf16 weights, bf16 KV"], arms["fp8 weights, bf16 KV"],
                              arms["fp8 weights, fp8 KV"])
    return [
        practice.Check(
            "ANSWER: BF16 weights do not fit; FP8 weights seat 8 sequences at 128k",
            dense["left_gb"] < 0 and cheapest["seats"] == 8 and cheap["seats"] == 4,
            f"the 405B counts {result['params'] / 1e9:.1f}B parameters against a published "
            f"{result['reported'] / 1e9:.1f}B. At BF16 that is {dense['weights_gb']:.1f} GB on a "
            f"{NODE_GB} GB node, leaving {dense['left_gb']:+.1f} GB before a single token is "
            f"cached; at FP8 it is {cheap['weights_gb']:.1f} GB leaving {cheap['left_gb']:.1f}, "
            f"which seats {cheap['seats']} sequences with a BF16 cache and {cheapest['seats']} "
            "with an FP8 one",
        ),
        practice.Check(
            "ANSWER: FP8 is exactly half of BF16, because the formula is linear in bytes",
            per_token["bf16"] == 2 * per_token["fp8"],
            f"the cache is {per_token['bf16'] / 1024:.0f} KB per token at BF16 and "
            f"{per_token['fp8'] / 1024:.0f} KB at FP8, which is {cheap['per_seq_gb']:.1f} GB and "
            f"{cheapest['per_seq_gb']:.1f} GB for one {CONTEXT // 1024}k sequence. "
            "2 x layers x kv_heads x head_dim x seq x bytes is linear in bytes and nothing else "
            "in it changes, so 'at FP8 it is half the BF16 number' is exact rather than "
            "approximate",
        ),
        practice.Check(
            "FINDING: the question presumes a subtraction that goes negative",
            dense["weights_gb"] > NODE_GB and cheap["weights_gb"] < NODE_GB,
            f"'{NODE_GB}GB total, minus weight memory' reads as though the remainder were "
            f"positive. A 405B at BF16 is {dense['weights_gb']:.0f} GB before anything is cached, "
            f"so the first decision on this node is quantising the weights -- which the exercise "
            f"does not mention and which moves {dense['weights_gb'] - cheap['weights_gb']:.0f} "
            f"GB, {(dense['weights_gb'] - cheap['weights_gb']) / cheap['per_seq_gb']:.0f} "
            "sequences' worth of cache",
        ),
        practice.Check(
            "MECHANISM: 504 KB per token is a GQA number, and without it one sequence is 1,008 GB",
            result["mha_per_seq_gb"] > NODE_GB and result["scheme"].startswith("GQA"),
            f"the attention scheme is {result['scheme']}, so the cache is "
            f"2 x 126 layers x 8 KV heads x 128 dims x 2 bytes = "
            f"{per_token['bf16'] / 1024:.0f} KB per token. With 128 KV heads instead of 8 it "
            f"would be {result['mha_per_seq_gb']:.0f} GB for a single {CONTEXT // 1024}k "
            f"sequence, against a {NODE_GB} GB node. The 405B is servable at long context because "
            "of one config field",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
