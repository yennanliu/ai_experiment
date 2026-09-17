"""Exercise 4 — absorbing the decompression costs 6.9x the parameters and saves 14x the cache.

    Read Section 2.1 of the DeepSeek-V3 technical report (arXiv:2412.19437) on
    MLA. Explain in three sentences why the K and V decompression matrices can be
    "absorbed" into the subsequent matmul for inference-time efficiency.

Reading of the exercise: the explanation is written as arithmetic on the
lesson's own `mla_attention_params`, because "efficiency" is a quantity and the
function that counts MLA's matrices is the place to count it. Both readings of
the claim are priced: the absorbed matrix as a *parameter* count, and what the
absorption lets the cache hold.

**ANSWER: the absorbed matrix is bigger, and that is not the point.** `W_q_up`
is `q_lora x (n_heads * head_dim)` = **11.01M** and `W_k_up` is
`kv_lora x (n_heads * head_dim)` = **3.67M**, 14.68M together. Folding them into
one `q_lora x kv_lora` matrix *per head* is `1536 x 512 x 128` = **100.66M** --
**6.9x more**. Absorption is not a parameter saving.

**MECHANISM: what it saves is the thing that is stored per token.** With the
product pre-multiplied, a query is projected straight into the latent's space and
scored against the cached latent, so the per-head keys are never materialised.
The cache holds `kv_lora = 512` numbers per token per layer instead of
`n_heads * head_dim = 7168` -- **14x** -- and that is the number
`compute_totals` reports as `kv_cache_bytes`.

**FINDING: the trade turns on one inequality, and DeepSeek is on the wrong side
of it for parameters.** Absorption is cheaper in weights only when
`q_lora * kv_lora < head_dim * (q_lora + kv_lora)`: here that is **786,432
against 114,688**, so it loses by 6.9x. The reason to do it anyway is that
parameters are paid once per model and the cache is paid once per token per
sequence in flight.

**FINDING: the absorbed form is never the one the parameter counter counts.**
`mla_attention_params` adds `q_up`, `k_up` and `v_up` as separate matrices, which
is the *training* form. An inference deployment that absorbs them holds different
weights, so the 84.4M per layer this function reports describes neither the
absorbed model nor its memory at serving time.

Structure: `matrices` prices each MLA projection from the config; `absorbed`
prices the folded per-head product against them.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "20-deepseek-v3-walkthrough"
CONTEXT, GIB = 131_072, 1024 ** 3


def matrices(cfg):
    """Each projection `mla_attention_params` sums, named."""
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    head_dim = hidden // heads
    kv_lora, q_lora = cfg["kv_lora_rank"], cfg["q_lora_rank"]
    return {"q_down": hidden * q_lora, "q_up": q_lora * heads * head_dim,
            "kv_down": hidden * kv_lora, "k_up": kv_lora * heads * head_dim,
            "v_up": kv_lora * heads * head_dim, "o_proj": heads * head_dim * hidden,
            "head_dim": head_dim, "heads": heads, "kv_lora": kv_lora, "q_lora": q_lora}


def absorbed(parts):
    """W_q_up^T W_k_up folded per head: a q_lora x kv_lora matrix for each."""
    return parts["q_lora"] * parts["kv_lora"] * parts["heads"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = dict(ref.DEEPSEEK_V3)
    parts = matrices(cfg)
    folded = absorbed(parts)
    materialised = parts["q_up"] + parts["k_up"]
    report = ref.compute_totals(cfg, CONTEXT)
    per_head = (parts["q_lora"] * parts["kv_lora"],
                parts["head_dim"] * (parts["q_lora"] + parts["kv_lora"]))
    return {
        "parts": parts,
        "folded": folded,
        "materialised": materialised,
        "param_ratio": folded / materialised,
        "cached": (parts["kv_lora"], parts["heads"] * parts["head_dim"]),
        "cache_ratio": parts["heads"] * parts["head_dim"] / parts["kv_lora"],
        "cache_gb": report.kv_cache_bytes / GIB,
        "uncompressed_gb": (2 * cfg["num_hidden_layers"] * parts["heads"] * parts["head_dim"]
                            * CONTEXT * 2 / GIB),
        "inequality": per_head,
        "counted": ref.mla_attention_params(cfg["hidden_size"], parts["heads"],
                                            parts["head_dim"], parts["kv_lora"],
                                            parts["q_lora"]),
    }


def verify(result):
    parts, cached = result["parts"], result["cached"]
    folded, lhs_rhs = result["folded"], result["inequality"]
    return [
        practice.Check(
            "ANSWER: the absorbed matrix is 6.9x bigger, and that is not the point",
            result["param_ratio"] > 5,
            f"W_q_up is q_lora x (n_heads x head_dim) = {parts['q_up'] / 1e6:.2f}M and W_k_up is "
            f"kv_lora x the same = {parts['k_up'] / 1e6:.2f}M, "
            f"{result['materialised'] / 1e6:.2f}M together. Folding them into one "
            f"q_lora x kv_lora matrix per head is {parts['q_lora']} x {parts['kv_lora']} x "
            f"{parts['heads']} = {folded / 1e6:.2f}M -- {result['param_ratio']:.1f}x more. "
            "Absorption is not a parameter saving",
        ),
        practice.Check(
            "MECHANISM: what it saves is the thing that is stored per token",
            result["cache_ratio"] == cached[1] / cached[0] > 10,
            f"with the product pre-multiplied, a query is projected straight into the latent's "
            f"space and scored against the cached latent, so the per-head keys are never "
            f"materialised. The cache holds kv_lora = {cached[0]} numbers per token per layer "
            f"instead of n_heads x head_dim = {cached[1]} -- {result['cache_ratio']:.0f}x -- "
            f"which at {CONTEXT // 1024}k context is {result['cache_gb']:.2f} GB against "
            f"{result['uncompressed_gb']:.2f}",
        ),
        practice.Check(
            "FINDING: the trade turns on one inequality and the parameters lose it",
            lhs_rhs[0] > lhs_rhs[1],
            f"absorption is cheaper in weights only when q_lora x kv_lora < head_dim x (q_lora + "
            f"kv_lora): here that is {lhs_rhs[0]:,} against {lhs_rhs[1]:,}, so it loses by "
            f"{lhs_rhs[0] / lhs_rhs[1]:.1f}x. The reason to do it anyway is that parameters are "
            "paid once per model and the cache is paid once per token per sequence in flight",
        ),
        practice.Check(
            "FINDING: the absorbed form is never the one the parameter counter counts",
            result["counted"] == sum(parts[key] for key in
                                     ("q_down", "q_up", "kv_down", "k_up", "v_up", "o_proj")),
            f"mla_attention_params adds q_down, q_up, kv_down, k_up, v_up and o_proj as separate "
            f"matrices, summing to exactly {result['counted'] / 1e6:.1f}M per layer -- the "
            f"training form. An inference deployment that absorbs the decompression holds "
            f"different weights, {folded / 1e6:.1f}M of them in place of "
            f"{result['materialised'] / 1e6:.1f}M, so this number describes neither the absorbed "
            "model nor its memory at serving time",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
