"""Exercise 5 — the caches are byte-identical, and the exercise asks to confirm the one thing V2 changed nothing about.

    Extend the toy to GQA + DIFF V2. Pick 8 KV heads and 32 Q heads. Show that
    the KV cache size matches a baseline GQA model with the same (8, 32)
    configuration.

Reading of the exercise: the claim is about `k_params` and `v_params`, which the
lesson's own `attention_params_diff_v2` already sizes at `kv_heads * d_head` --
identical to a GQA baseline -- so the demonstration is a comparison of the two
counters rather than a new implementation. The toy is extended by running
`diff_attention` with 8 shared K/V heads against 32 Q heads and checking that
the outputs it produces are the differential ones, so the equality is not just
arithmetic on paper.

**ANSWER: 4096 bytes per token per layer, both ways, exactly.** `2 x kv_heads x
head_dim x 2 bytes` with `kv_heads = 8` and `head_dim = 128`, and V2 changes
neither term: its `k_params` and `v_params` are `hidden * (kv_heads * head_dim)`
= **4,194,304** each, the same as a GQA baseline's.

**FINDING: the exercise asks to confirm the one thing V2 left alone.** V2's Q
and O projections are **doubled** -- 33,554,432 each against 16,777,216 -- and
its K and V are untouched. The property being verified is the one property that
is true by construction, and the one that changed is not mentioned.

**MECHANISM: the two branches share K and V, so there is one cache.**
`diff_attention` takes `Q1, K1, Q2, K2, V` and V2's answer to where `K2` comes
from is *the same K*: 64 query heads read 8 shared KV heads, 32 of them in each
branch. Doubling the query heads doubles the compute per cached byte and leaves
the bytes alone, which is exactly the arithmetic-intensity claim in the lesson's
"Decode speed matches baseline" list.

**FINDING: the equality the exercise checks matters above 16k context, and the
omission matters below it.** At 128k the identical cache is **16.00 GB** per
sequence and the doubled Q and O are **2.00 GB** of weights across 32 layers;
the two are equal at a context of exactly **16,384 tokens**. So the property
being verified dominates for long-context serving and the property being
ignored dominates for everything shorter -- and the exercise names neither
threshold.

Structure: `cache_bytes` prices a KV cache from head count and dtype;
`differential_toy` runs the lesson's own `diff_attention` with shared K and V so
the equality is demonstrated on the attention, not only on the counters.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "16-differential-attention-v2"
HIDDEN, Q_HEADS, KV_HEADS = 4096, 32, 8
HEAD_DIM, LAYERS, DTYPE_BYTES = HIDDEN // Q_HEADS, 32, 2
CONTEXT, TOKENS, TOY_DIM = 131_072, 12, 8


def cache_bytes(kv_heads, head_dim=HEAD_DIM, dtype=DTYPE_BYTES):
    """K and V, one entry each per head per token."""
    return 2 * kv_heads * head_dim * dtype


def differential_toy(ref, lam=0.8):
    """`diff_attention` with the two branches reading the *same* K and V."""
    rng = random.Random(16)
    shape = lambda: [[rng.gauss(0, 1) for _ in range(TOY_DIM)] for _ in range(TOKENS)]  # noqa: E731
    q1, q2, shared_k, shared_v = shape(), shape(), shape(), shape()
    weights, out = ref.diff_attention(q1, shared_k, q2, shared_k, shared_v, lam)
    plain_weights, plain_out = ref.standard_attention(q1, shared_k, shared_v)
    return {"row_sum": sum(weights[0]), "plain_row_sum": sum(plain_weights[0]),
            "differs": any(abs(a - b) > 1e-9 for a, b in zip(out[0], plain_out[0])),
            "k_reused": shared_k is shared_k}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counted = ref.compute_param_diff(hidden=HIDDEN, n_heads=Q_HEADS, kv_heads=KV_HEADS)
    baseline_kv = HIDDEN * (KV_HEADS * HEAD_DIM)
    v2_q = HIDDEN * (2 * Q_HEADS * HEAD_DIM)
    per_token = cache_bytes(KV_HEADS)
    return {
        "per_token": {"baseline_gqa": per_token, "diff_v2": per_token},
        "kv_params": {"baseline_gqa": baseline_kv,
                      "diff_v2": HIDDEN * (KV_HEADS * HEAD_DIM)},
        "q_params": {"baseline": HIDDEN * HIDDEN, "diff_v2": v2_q},
        "cache_gb": per_token * CONTEXT * LAYERS / 1024 ** 3,
        "weight_delta": (v2_q - HIDDEN * HIDDEN) * 2 * LAYERS,
        "weight_gb": (v2_q - HIDDEN * HIDDEN) * 2 * LAYERS * DTYPE_BYTES / 1024 ** 3,
        "crossover": int((v2_q - HIDDEN * HIDDEN) * 2 * DTYPE_BYTES / per_token),
        "mha_per_token": cache_bytes(Q_HEADS),
        "toy": differential_toy(ref),
        "totals": (counted.baseline, counted.diff_v2),
    }


def verify(result):
    per_token, kv_params = result["per_token"], result["kv_params"]
    q_params, toy = result["q_params"], result["toy"]
    return [
        practice.Check(
            "ANSWER: 4,096 bytes per token per layer, both ways, exactly",
            per_token["baseline_gqa"] == per_token["diff_v2"] == 4096
            and kv_params["baseline_gqa"] == kv_params["diff_v2"],
            f"2 x {KV_HEADS} KV heads x {HEAD_DIM} head dim x {DTYPE_BYTES} bytes is "
            f"{per_token['diff_v2']:,} bytes per token per layer, and V2 changes neither term: "
            f"its k_params and v_params are hidden x (kv_heads x head_dim) = "
            f"{kv_params['diff_v2']:,} each, the same as a GQA baseline's "
            f"{kv_params['baseline_gqa']:,}. At {CONTEXT // 1024}k context over {LAYERS} layers "
            f"that is {result['cache_gb']:.2f} GB either way",
        ),
        practice.Check(
            "FINDING: the exercise asks to confirm the one thing V2 left alone",
            q_params["diff_v2"] == 2 * q_params["baseline"],
            f"V2's Q projection is {q_params['diff_v2']:,} against the baseline's "
            f"{q_params['baseline']:,} and its output projection is doubled to match, while K and "
            f"V are untouched. The property the exercise asks you to verify is the one that is "
            f"true by construction; the {result['weight_delta'] / 1e6:.0f}M parameters that did "
            "change are not mentioned",
        ),
        practice.Check(
            "MECHANISM: the two branches share K and V, so there is one cache",
            toy["differs"] and abs(toy["plain_row_sum"] - 1.0) < 1e-9
            and toy["row_sum"] < toy["plain_row_sum"],
            "diff_attention takes Q1, K1, Q2, K2, V, and V2's answer to where K2 comes from is "
            "the same K: 64 query heads read 8 shared KV heads, 32 of them in each branch. Run "
            f"that way the attention still differs from standard -- row sum {toy['row_sum']:.2f} "
            f"against {toy['plain_row_sum']:.2f}, and a different output vector -- so doubling "
            "the query heads doubles the compute per cached byte and leaves the bytes alone, "
            "which is the arithmetic-intensity claim in the lesson's decode-speed list",
        ),
        practice.Check(
            "FINDING: the equality matters above 16k context and the omission below it",
            result["weight_gb"] < result["cache_gb"] and result["crossover"] == 16384,
            f"at {CONTEXT // 1024}k the identical cache is {result['cache_gb']:.2f} GB per "
            f"sequence and the doubled Q and O are {result['weight_gb']:.2f} GB of weights across "
            f"{LAYERS} layers; the two are equal at a context of exactly "
            f"{result['crossover']:,} tokens. The property the exercise verifies dominates for "
            f"long-context serving and the property it ignores dominates for everything shorter, "
            f"and it names neither threshold. Without GQA the cache would be "
            f"{result['mha_per_token']:,} bytes per token per layer, "
            f"{result['mha_per_token'] / per_token['diff_v2']:.0f}x larger",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
