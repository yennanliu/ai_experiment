"""Exercise 1 — the reduction is exactly 8.00x because 32 attention layers became 4.

    Run `code/main.py` to compute KV cache at 256k context for a 32-layer pure
    Transformer (hidden 4096, 32 heads) and for a Jamba-1 hybrid of the same
    shape. Verify the ~8x memory reduction the AI21 paper claims.

Reading of the exercise: both figures come from the lesson's own
`kv_cache_bytes` and `ssm_state_bytes` on its own `HybridConfig` shapes, and the
verification is then followed by asking what produced the 8 -- because
`kv_cache_bytes` is linear in `attn_layers` and the two configs differ in that
field and no other.

**ANSWER: exactly 8.00x on the KV cache, and the 8 is `32 / 4`.**

    pure Transformer 32L   KV 128.00 GB   SSM 0.00000 GB   total 128.00 GB
    Jamba 1:7 hybrid 32L   KV  16.00 GB   SSM 0.00342 GB   total  16.00 GB

Adding the SSM state takes the total ratio to 7.998. `kv_cache_bytes` is
`2 * attn_layers * n_kv_heads * head_dim * ctx * bytes`, and
the two configs share every field but `attn_layers` -- 32 against 4. The AI21
paper's "~8x" is the attention-layer fraction, and the calculator reproduces it
to fifteen decimal places because it cannot do anything else.

**FINDING: the SSM state is 0.02% of the hybrid's cache.** 3.42 MB against
16.00 GB. It is constant in context, so at 256k it rounds away entirely -- the
lesson's own summary line calls it "~4 MB" and that is the whole of it. The
hybrid's memory story at long context is which layers were removed, not what
replaced them.

**FINDING: the Jamba config uses MHA where Jamba uses GQA.** `n_kv_heads=32`
for every hybrid in the file, while Jamba's attention layers are grouped-query.
Setting `n_kv_heads=8`, as the lesson's own "pure Transformer 32L (GQA 8)"
config does, takes the 1:7 hybrid to **4.00 GB** and the reduction to **32x**.
The configuration understates the architecture it is modelling by 4x.

**MECHANISM: the two savings multiply, and the paper's 8x is one of them.**
Removing 7 of every 8 attention layers is 8x; sharing KV heads 4:1 is another 4x;
together 32x. The exercise asks you to verify the first and the file contains a
config that demonstrates the second, on a different row.

Structure: `shape` builds one `HybridConfig`; `memory` reports KV, SSM state and
their total through the lesson's own two functions.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "21-jamba-hybrid-ssm-transformer"
CONTEXT, BYTES, GIB = 262_144, 2, 1024 ** 3
LAYERS, HIDDEN, HEAD_DIM, STATE = 32, 4096, 128, 16


def shape(ref, name, attn_layers, kv_heads=32, state=STATE):
    return ref.HybridConfig(name=name, total_layers=LAYERS, attn_layers=attn_layers,
                            hidden=HIDDEN, n_q_heads=32, n_kv_heads=kv_heads,
                            head_dim=HEAD_DIM, ssm_state_size=state)


def memory(ref, cfg):
    kv = ref.kv_cache_bytes(cfg, CONTEXT, BYTES)
    state = ref.ssm_state_bytes(cfg, BYTES)
    return {"kv_gb": kv / GIB, "state_gb": state / GIB, "total_gb": (kv + state) / GIB,
            "state_share": state / kv if kv else float("inf"), "attn": cfg.attn_layers,
            "kv_heads": cfg.n_kv_heads}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {"pure": memory(ref, shape(ref, "pure", LAYERS, state=0)),
            "jamba 1:7": memory(ref, shape(ref, "1:7", 4)),
            "jamba 1:7 + GQA": memory(ref, shape(ref, "1:7 gqa", 4, kv_heads=8)),
            "pure + GQA": memory(ref, shape(ref, "pure gqa", LAYERS, kv_heads=8, state=0))}
    return {
        "rows": rows,
        "reduction": rows["pure"]["kv_gb"] / rows["jamba 1:7"]["kv_gb"],
        "with_state": rows["pure"]["total_gb"] / rows["jamba 1:7"]["total_gb"],
        "layer_ratio": LAYERS / 4,
        "gqa_reduction": rows["pure"]["kv_gb"] / rows["jamba 1:7 + GQA"]["kv_gb"],
        "head_ratio": 32 / 8,
        "state_mb": rows["jamba 1:7"]["state_gb"] * 1024,
    }


def column(rows, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in rows.items())


def verify(result):
    rows = result["rows"]
    pure, hybrid, gqa = rows["pure"], rows["jamba 1:7"], rows["jamba 1:7 + GQA"]
    return [
        practice.Check(
            "ANSWER: exactly 8.00x on the KV cache, and the 8 is 32 / 4",
            abs(result["reduction"] - result["layer_ratio"]) < 1e-9,
            f"at {CONTEXT // 1024}k the pure Transformer is {pure['kv_gb']:.2f} GB of KV cache "
            f"and the 1:7 hybrid {hybrid['kv_gb']:.2f} GB -- a ratio of "
            f"{result['reduction']:.2f} exactly, or {result['with_state']:.3f} once the SSM state "
            f"is added. kv_cache_bytes is 2 x attn_layers x n_kv_heads x "
            f"head_dim x ctx x bytes and the two configs share every field but attn_layers, "
            f"{pure['attn']} against {hybrid['attn']}, so the paper's '~8x' is the attention-layer "
            "fraction and the calculator reproduces it exactly because it cannot do anything else",
        ),
        practice.Check(
            "FINDING: the SSM state is 0.02% of the hybrid's cache",
            hybrid["state_share"] < 0.001,
            f"the SSM state is {result['state_mb']:.2f} MB against "
            f"{hybrid['kv_gb']:.2f} GB of KV -- {hybrid['state_share']:.4%}. It is constant in "
            f"context while the KV cache is linear in it, so at {CONTEXT // 1024}k it rounds away "
            "entirely: the hybrid's memory story at long context is which layers were removed, "
            "not what replaced them",
        ),
        practice.Check(
            "FINDING: the Jamba config uses MHA where Jamba uses GQA",
            hybrid["kv_heads"] == 32 and gqa["total_gb"] < hybrid["total_gb"] / 3,
            f"every hybrid config in the file sets n_kv_heads={hybrid['kv_heads']}, while Jamba's "
            f"attention layers are grouped-query. Setting it to {gqa['kv_heads']}, as the lesson's "
            f"own 'pure Transformer 32L (GQA 8)' row does, takes the 1:7 hybrid to "
            f"{gqa['total_gb']:.2f} GB and the reduction to {result['gqa_reduction']:.0f}x -- the "
            "configuration understates the architecture it is modelling",
        ),
        practice.Check(
            "MECHANISM: the two savings multiply, and the paper's 8x is one of them",
            abs(result["gqa_reduction"] - result["layer_ratio"] * result["head_ratio"]) < 1e-9,
            f"removing 7 of every 8 attention layers is {result['layer_ratio']:.0f}x; sharing KV "
            f"heads 4:1 is another {result['head_ratio']:.0f}x; together "
            f"{result['gqa_reduction']:.0f}x, exactly the product. The exercise asks you to verify "
            "the first, and the file contains a config demonstrating the second on a different row",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
