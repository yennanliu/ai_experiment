"""Phase 07 / Lesson 03 -- pure-stdlib attention vs torch.nn.MultiheadAttention.

The lesson opens by declaring "No numpy, no torch. A tiny Matrix class carries
the ops we need." That is a good way to learn the mechanism, and it leaves one
question open: is the thing you just wrote actually what the library does?

This demo answers it. The lesson's own weights go into torch's module, the same
tokens go through both, and every number is compared -- outputs *and* the
per-head attention weight matrices, which is where a wrong head split or a
missing 1/sqrt(dk) would hide.

Weight layout: the lesson computes `Q = X @ Wq`; torch computes
`q = x @ in_proj_weight[:E].T`. So torch's packed projection is the transpose
of the lesson's, stacked [Wq, Wk, Wv], and out_proj.weight is Wo transposed.

Run:  uv run demo run phases/07-transformers-deep-dive/03-multi-head-attention
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from harness.explain import explain          # noqa: E402
from harness.parity import assert_close, compare, load_reference, report  # noqa: E402

LESSON = "phases/07-transformers-deep-dive/03-multi-head-attention/code/main.py"
SEED = 42
N_TOKENS = 6
D_MODEL = 8
N_HEADS = 2
TOKENS = ["the", "cat", "sat", "on", "the", "mat"]


def to_tensor(matrix, dtype):
    """The lesson's Matrix keeps a flat row-major list; torch takes it as-is."""
    import torch

    return torch.tensor(matrix.data, dtype=dtype).reshape(matrix.rows, matrix.cols)


def build_torch_mha(Wq, Wk, Wv, Wo, dtype):
    """A torch MultiheadAttention carrying exactly the lesson's weights."""
    import torch

    mha = torch.nn.MultiheadAttention(
        embed_dim=D_MODEL, num_heads=N_HEADS, bias=False, batch_first=True
    ).to(dtype)
    with torch.no_grad():
        # in_proj_weight is [3E, E] and is applied as x @ W.T, so each of the
        # lesson's [E, E] right-multiplied projections goes in transposed.
        mha.in_proj_weight.copy_(
            torch.cat([to_tensor(w, dtype).T for w in (Wq, Wk, Wv)], dim=0)
        )
        mha.out_proj.weight.copy_(to_tensor(Wo, dtype).T)
    return mha


def main() -> int:
    import torch
    import torch.nn.functional as F

    ref = load_reference(LESSON)
    rng = random.Random(SEED)

    # Exactly the setup in the lesson's own main().
    X = ref.randn_matrix(N_TOKENS, D_MODEL, rng, scale=1.0)
    Wq = ref.randn_matrix(D_MODEL, D_MODEL, rng)
    Wk = ref.randn_matrix(D_MODEL, D_MODEL, rng)
    Wv = ref.randn_matrix(D_MODEL, D_MODEL, rng)
    Wo = ref.randn_matrix(D_MODEL, D_MODEL, rng)

    mine_out, mine_weights = ref.multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=N_HEADS)

    checks = []

    # --- float64: same algorithm, or not ----------------------------------
    dtype = torch.float64
    mha = build_torch_mha(Wq, Wk, Wv, Wo, dtype)
    x = to_tensor(X, dtype).unsqueeze(0)
    with torch.no_grad():
        torch_out, torch_weights = mha(
            x, x, x, need_weights=True, average_attn_weights=False
        )

    checks.append(
        assert_close(mine_out, torch_out.squeeze(0),
                     label="MHA output vs torch (float64)", atol=1e-12)
    )
    # Per-head weights are the real test: a botched head split still produces a
    # plausible-looking output, but the wrong attention pattern.
    for head in range(N_HEADS):
        checks.append(
            assert_close(mine_weights[head], torch_weights[0, head],
                         label=f"head {head} attention weights", atol=1e-12)
        )

    # --- one head, straight against scaled_dot_product_attention ----------
    Q = ref.matmul(X, Wq)
    K = ref.matmul(X, Wk)
    V = ref.matmul(X, Wv)
    mine_head0, _ = ref.scaled_dot_product_attention(
        ref.split_heads(Q, N_HEADS)[0],
        ref.split_heads(K, N_HEADS)[0],
        ref.split_heads(V, N_HEADS)[0],
    )
    d_head = D_MODEL // N_HEADS
    q, k, v = (to_tensor(m, dtype)[:, :d_head] for m in (Q, K, V))
    checks.append(
        assert_close(mine_head0, F.scaled_dot_product_attention(q, k, v),
                     label="one head vs F.scaled_dot_product_attention", atol=1e-12)
    )

    # --- grouped-query attention ------------------------------------------
    n_q_heads, n_kv_heads = 4, 2
    d_gqa = D_MODEL // n_q_heads
    gWq = ref.randn_matrix(D_MODEL, D_MODEL, rng)
    gWk = ref.randn_matrix(D_MODEL, d_gqa * n_kv_heads, rng)
    gWv = ref.randn_matrix(D_MODEL, d_gqa * n_kv_heads, rng)
    gQ, gK, gV = (ref.matmul(X, w) for w in (gWq, gWk, gWv))

    # torch wants [batch, heads, seq, d_head]; the lesson splits along columns.
    def stack(matrix, n_heads):
        heads = ref.split_heads(matrix, n_heads)
        return torch.stack([to_tensor(h, dtype) for h in heads]).unsqueeze(0)

    mine_gqa_heads = []
    Qh = ref.split_heads(gQ, n_q_heads)
    Kh = ref.split_heads(gK, n_kv_heads)
    Vh = ref.split_heads(gV, n_kv_heads)
    repeat = n_q_heads // n_kv_heads
    for i in range(n_q_heads):
        out, _ = ref.scaled_dot_product_attention(Qh[i], Kh[i // repeat], Vh[i // repeat])
        mine_gqa_heads.append(out)

    torch_gqa = F.scaled_dot_product_attention(
        stack(gQ, n_q_heads), stack(gK, n_kv_heads), stack(gV, n_kv_heads),
        enable_gqa=True,
    )
    checks.append(
        assert_close(
            [h.data for h in mine_gqa_heads], torch_gqa.squeeze(0),
            label=f"GQA {n_q_heads}Q/{n_kv_heads}KV vs enable_gqa=True", atol=1e-12,
        )
    )

    report(checks, title="phase 07 / lesson 03: stdlib attention vs torch")

    # --- what float32 costs, measured not asserted ------------------------
    mha32 = build_torch_mha(Wq, Wk, Wv, Wo, torch.float32)
    x32 = to_tensor(X, torch.float32).unsqueeze(0)
    with torch.no_grad():
        out32, _ = mha32(x32, x32, x32, need_weights=False)
    drift = compare(mine_out, out32.squeeze(0),
                    label="same weights in float32", atol=1e-4)
    print(f"\nprecision, not algorithm:  float32 deviates by "
          f"{drift.max_abs_diff:.2e} absolute, {drift.max_rel_diff:.2e} relative")

    # --- the KV-cache claim GQA exists to make ----------------------------
    mha_cache = N_HEADS * N_TOKENS * (D_MODEL // N_HEADS) * 2
    gqa_cache = n_kv_heads * N_TOKENS * d_gqa * 2
    print(f"KV cache elements:  MHA {mha_cache}  vs  GQA {gqa_cache}  "
          f"({mha_cache / gqa_cache:.0f}x smaller)")
    print(f"\nattention over {' '.join(TOKENS)!r}: same numbers, both ways.")
    return 0


if __name__ == "__main__":
    if explain(__file__):
        raise SystemExit(0)
    raise SystemExit(main())
