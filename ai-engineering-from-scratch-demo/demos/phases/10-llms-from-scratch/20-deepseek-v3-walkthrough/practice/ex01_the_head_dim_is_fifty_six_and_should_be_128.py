"""Exercise 1 — 664.5B against 671B, and the 6.5B is `hidden // n_heads`.

    Run `code/main.py`. Compare the calculator's total-parameter estimate to the
    published 671B and identify where the delta comes from. The paper's Section
    2 has the full itemization.

Reading of the exercise: the estimate is the lesson's own `compute_totals`, and
"identify where the delta comes from" is answered by naming each candidate term
and pricing it, rather than by describing one. The *active* count is checked
alongside the total, because DeepSeek publishes both and the calculator reports
both.

**ANSWER: 664.54B against 671B, and 30.36B active against 37B -- and one line
explains the second gap almost exactly.**

`compute_components` sets `head_dim = hidden_size // num_attention_heads` =
`7168 // 128` = **56**. DeepSeek V3's published head dimensions are **128** for
V and the NoPE half of QK, plus a 64-wide RoPE half, so the QK path is 192 wide.
Recomputing `mla_attention_params` with those gives **187.1M** per layer against
the calculator's 84.4M -- **2.22x** -- and over 61 layers that is a **6.26B**
shortfall.

    active as computed           30.36 B
    + attention at real head dims + 6.26 B
    + the untied LM head          + 0.93 B
                                  --------
                                   37.55 B   against a published 37 B

**FINDING: the total's delta is a different pair of terms.** The MTP module is
counted with a **dense** MLP -- `mtp_module_params`' own docstring says so and
says the published overhead is 14B -- giving 0.705B. Rebuilt with the MoE
structure DeepSeek actually uses, 257 experts at 44.0M each, it is **11.63B**.
Adding that and the untied head takes 664.54B to **676.39B**, overshooting 671B
by 5.4B, because the 14B the docstring cites is itself larger than the 11.63B
the MoE arithmetic gives.

**MECHANISM: `hidden // n_heads` is an MHA identity and MLA is not MHA.** In
multi-head attention the head dimensions must tile the hidden size; in MLA the
heads read from a latent and their width is a free hyperparameter. DeepSeek
chose 128, which is **2.29x** what tiling would give, and the calculator's one
line of arithmetic assumes the constraint that MLA exists to remove.

**FINDING: the embedding is counted once and the config does not say whether it
is tied.** `DEEPSEEK_V3` has no `tie_word_embeddings` key, and
`compute_components` adds `vocab_size * hidden_size` a single time -- 0.927B, or
0.14% of the total and 3.1% of the active count.

Structure: `published_dims` recomputes MLA with DeepSeek's real head widths;
`moe_mtp` prices the MTP block the way the paper builds it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "20-deepseek-v3-walkthrough"
PUBLISHED_TOTAL, PUBLISHED_ACTIVE = 671e9, 37e9
V_HEAD_DIM, QK_NOPE, QK_ROPE = 128, 128, 64
EXPERTS_IN_MTP = 257


def published_dims(ref, cfg):
    """MLA priced with DeepSeek's own head widths instead of hidden // n_heads."""
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    kv_lora, q_lora = cfg["kv_lora_rank"], cfg["q_lora_rank"]
    qk = QK_NOPE + QK_ROPE
    return (hidden * q_lora + q_lora * heads * qk + hidden * (kv_lora + QK_ROPE)
            + kv_lora * heads * V_HEAD_DIM + kv_lora * heads * V_HEAD_DIM
            + heads * V_HEAD_DIM * hidden)


def moe_mtp(ref, cfg):
    """The MTP block with the MoE MLP the paper uses, not the dense one the code uses."""
    hidden = cfg["hidden_size"]
    expert = ref.swiglu_mlp_params(hidden, cfg["moe_intermediate_size"])
    return (2 * hidden * hidden + 4 * hidden * hidden + expert * EXPERTS_IN_MTP
            + ref.router_params(hidden, cfg["num_experts"]) + 2 * ref.rmsnorm_params(hidden))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = dict(ref.DEEPSEEK_V3)
    report = ref.compute_totals(cfg)
    parts = ref.compute_components(cfg)
    layers = cfg["num_hidden_layers"]
    real_attn = published_dims(ref, cfg)
    gap = (real_attn - parts.attention_per_layer) * layers
    mtp = moe_mtp(ref, cfg)
    return {
        "total": report.total,
        "active": report.active,
        "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
        "attn": (parts.attention_per_layer, real_attn),
        "attn_gap": gap,
        "embedding": parts.embedding,
        "mtp": (parts.mtp_module, mtp),
        "explained_active": report.active + gap + parts.embedding,
        "explained_total": report.total + parts.embedding + mtp - parts.mtp_module,
        "tied_key": "tie_word_embeddings" in cfg,
        "layers": layers,
    }


def verify(result):
    counted_attn, real_attn = result["attn"]
    dense_mtp, moe = result["mtp"]
    return [
        practice.Check(
            "ANSWER: 664.5B against 671B, and the active gap is head_dim = 56",
            abs(result["explained_active"] - PUBLISHED_ACTIVE) < 0.03 * PUBLISHED_ACTIVE,
            f"compute_totals gives {result['total'] / 1e9:.2f}B total against a published "
            f"{PUBLISHED_TOTAL / 1e9:.0f}B and {result['active'] / 1e9:.2f}B active against "
            f"{PUBLISHED_ACTIVE / 1e9:.0f}B. compute_components sets head_dim = hidden_size // "
            f"num_attention_heads = {result['head_dim']}, where DeepSeek's published widths are "
            f"{V_HEAD_DIM} for V and {QK_NOPE}+{QK_ROPE} for QK: recomputing MLA with those gives "
            f"{real_attn / 1e6:.1f}M per layer against {counted_attn / 1e6:.1f}M, and over "
            f"{result['layers']} layers a {result['attn_gap'] / 1e9:.2f}B shortfall. Active plus "
            f"that plus the untied head is {result['explained_active'] / 1e9:.2f}B",
        ),
        practice.Check(
            "FINDING: the total's delta is a different pair of terms",
            moe > 10 * dense_mtp and result["explained_total"] > PUBLISHED_TOTAL,
            f"the MTP module is counted with a dense MLP -- mtp_module_params' own docstring says "
            f"so, and says the published overhead is 14B -- giving {dense_mtp / 1e9:.3f}B. Rebuilt "
            f"with the MoE structure DeepSeek uses, {EXPERTS_IN_MTP} experts at "
            f"{moe / EXPERTS_IN_MTP / 1e6:.0f}M each, it is {moe / 1e9:.2f}B. Adding that and the "
            f"untied head takes the total to {result['explained_total'] / 1e9:.2f}B, overshooting "
            f"{PUBLISHED_TOTAL / 1e9:.0f}B by "
            f"{(result['explained_total'] - PUBLISHED_TOTAL) / 1e9:.1f}B -- the 14B the docstring "
            "cites is itself larger than the MoE arithmetic gives",
        ),
        practice.Check(
            "MECHANISM: hidden // n_heads is an MHA identity and MLA is not MHA",
            real_attn > 2 * counted_attn,
            f"in multi-head attention the head dimensions must tile the hidden size; in MLA the "
            f"heads read from a latent and their width is a free hyperparameter. DeepSeek chose "
            f"{V_HEAD_DIM}, which is {V_HEAD_DIM / result['head_dim']:.2f}x what tiling gives, so "
            f"one line of arithmetic assumes the constraint MLA exists to remove -- and it "
            f"undercounts the attention block by {real_attn / counted_attn:.2f}x",
        ),
        practice.Check(
            "FINDING: the embedding is counted once and the config does not say whether it is tied",
            not result["tied_key"],
            f"DEEPSEEK_V3 has no tie_word_embeddings key and compute_components adds "
            f"vocab_size * hidden_size a single time -- {result['embedding'] / 1e9:.3f}B, or "
            f"{result['embedding'] / result['total']:.2%} of the total and "
            f"{result['embedding'] / result['active']:.1%} of the active count. Whether that is "
            "right is not recorded anywhere in the configuration it is read from",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
