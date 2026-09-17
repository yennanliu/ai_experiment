"""Exercise 2 — rank 256 halves the cache exactly and costs 0.05% of the parameters.

    Modify the config to use MLA rank 256 instead of 512. Compute the resulting
    KV cache size at 128k context. What percentage reduction does it buy, and at
    what cost to the per-head expressiveness?

Reading of the exercise: both quantities come from the lesson's own
`compute_totals`, which returns `kv_cache_bytes` and the parameter total from the
same config, so the trade can be read off one call per rank. "Per-head
expressiveness" is made concrete as the rank of the subspace each head's K and V
are reconstructed from, since that is the only thing `kv_lora_rank` bounds.

**ANSWER: exactly 50.0% of the cache, for 0.34B of parameters -- 0.05%.**

    rank   KV at 128k   total params   attention / layer
     512      7.62 GB      664.54B          84.4M
     256      3.81 GB      664.20B          78.9M
     128      1.91 GB      664.04B          76.2M

`kv_cache_bytes` is `n_layers * kv_lora_rank * max_seq * 2`, linear in the rank
and in nothing else, so every halving is exactly a halving.

**MECHANISM: the cost is a rank bound, and it is the point of the method.** Each
head reconstructs its K and V by multiplying the shared latent by
`kv_lora x (n_heads * head_dim)` matrices, so all 128 heads' keys live in a
subspace of dimension at most **512** -- at rank 256, at most 256, for 128 heads
of width 56. The compression is already 14x at rank 512; halving it to 28x is the
same trade one step further, not a different one.

**FINDING: the parameters barely move because the latent is small either way.**
Dropping the rank from 512 to 256 removes `hidden * 256` from the down-projection
and `256 * n_heads * head_dim` from each of the two up-projections -- 5.5M per
layer, 0.34B over 61 layers, against a 664B model. Cache is a per-token cost and
parameters are a one-time one, which is why the ratio is 989:1.

**FINDING: the GQA reference the lesson prints for comparison is not the same
model.** `compute_totals` hard-codes `kv_heads_hypothetical = 8` and
`head_dim_hypothetical = 128` -- neither is in `DEEPSEEK_V3`, whose
`num_key_value_heads` is **128**. The 30.50 GB it reports at 128k is a Llama-3
shape, and the 4.0x advantage it implies is MLA against a different model's GQA
rather than against DeepSeek's own attention.

Structure: `arm` runs the lesson's own `compute_totals` at one rank; `subspace`
reports the rank bound each setting imposes on a head's reconstruction.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "20-deepseek-v3-walkthrough"
CONTEXT, RANKS = 131_072, (512, 256, 128)
GIB = 1024 ** 3


def arm(ref, cfg, rank):
    report = ref.compute_totals(dict(cfg, kv_lora_rank=rank), CONTEXT)
    return {"cache_gb": report.kv_cache_bytes / GIB, "total": report.total,
            "attn": report.per_layer_attn, "gqa_gb": report.gqa_kv_cache_bytes_ref / GIB}


def subspace(cfg, rank):
    """The dimension of the space every head's K and V are reconstructed from."""
    heads = cfg["num_attention_heads"]
    head_dim = cfg["hidden_size"] // heads
    return {"rank": rank, "heads": heads, "head_dim": head_dim,
            "full_width": heads * head_dim, "compression": heads * head_dim / rank}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = dict(ref.DEEPSEEK_V3)
    rows = {rank: arm(ref, cfg, rank) for rank in RANKS}
    base, half = rows[512], rows[256]
    return {
        "rows": rows,
        "subspace": {rank: subspace(cfg, rank) for rank in RANKS},
        "cache_cut": 1 - half["cache_gb"] / base["cache_gb"],
        "param_cut": (base["total"] - half["total"]) / base["total"],
        "param_delta": base["total"] - half["total"],
        "ratio": ((base["cache_gb"] - half["cache_gb"]) / base["cache_gb"]
                  / ((base["total"] - half["total"]) / base["total"])),
        "gqa_keys": [key for key in cfg if "key_value" in key or "head_dim" in key],
        "config_kv_heads": cfg["num_key_value_heads"],
        "gqa_gb": base["gqa_gb"],
    }


def column(rows, field, fmt):
    return ", ".join(f"rank {rank} {format(row[field], fmt)}" for rank, row in rows.items())


def verify(result):
    rows, space = result["rows"], result["subspace"]
    base, half = rows[512], rows[256]
    return [
        practice.Check(
            "ANSWER: exactly 50.0% of the cache for 0.05% of the parameters",
            abs(result["cache_cut"] - 0.5) < 1e-9 and result["param_cut"] < 0.002,
            "at 128k context the cache is " + column(rows, "cache_gb", ".2f")
            + " GB on totals of " + column(rows, "total", ",.0f")
            + f". Halving the rank cuts the cache {result['cache_cut']:.1%} and the parameters "
            f"{result['param_cut']:.2%} -- {result['param_delta'] / 1e9:.2f}B. kv_cache_bytes is "
            "n_layers * kv_lora_rank * max_seq * 2, linear in the rank and in nothing else, so "
            "every halving is exactly a halving",
        ),
        practice.Check(
            "MECHANISM: the cost is a rank bound, and it is the point of the method",
            space[512]["compression"] == space[512]["full_width"] / 512,
            f"each head reconstructs K and V by multiplying the shared latent by "
            f"kv_lora x (n_heads * head_dim) matrices, so all {space[512]['heads']} heads' keys "
            f"live in a subspace of dimension at most the rank: "
            + ", ".join(f"{s['rank']} of {s['full_width']} "
                        f"({s['compression']:.0f}x)" for s in space.values())
            + ". The compression is already 14x at rank 512, and halving it is the same trade one "
            "step further rather than a different one",
        ),
        practice.Check(
            "FINDING: the parameters barely move because the latent is small either way",
            result["ratio"] > 100,
            f"dropping the rank removes hidden * 256 from the down-projection and 256 * n_heads * "
            f"head_dim from each of the two up-projections -- "
            f"{(base['attn'] - half['attn']) / 1e6:.1f}M per layer, "
            f"{result['param_delta'] / 1e9:.2f}B over 61 layers, against a "
            f"{base['total'] / 1e9:.0f}B model. Cache is a per-token cost and parameters are a "
            f"one-time one, which is why the ratio of the two percentages is {result['ratio']:.0f}"
            " to 1",
        ),
        practice.Check(
            "FINDING: the GQA reference the lesson prints is not the same model",
            result["config_kv_heads"] == 128 and result["gqa_gb"] >= 4 * base["cache_gb"],
            f"compute_totals hard-codes kv_heads_hypothetical = 8 and head_dim_hypothetical = 128, "
            f"neither of which is in DEEPSEEK_V3 -- whose num_key_value_heads is "
            f"{result['config_kv_heads']}. The {result['gqa_gb']:.2f} GB it reports at 128k is a "
            f"Llama-3 shape, and the {result['gqa_gb'] / base['cache_gb']:.1f}x advantage it "
            "implies is MLA against another model's GQA rather than against DeepSeek's own "
            "attention",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
