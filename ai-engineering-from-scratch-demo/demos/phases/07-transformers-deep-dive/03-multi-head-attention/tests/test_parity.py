"""Assertions on what Phase 07 / Lesson 03 claims about multi-head attention."""

import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parents[3]))

from harness.parity import load_reference  # noqa: E402
from run import D_MODEL, LESSON, N_HEADS, N_TOKENS, build_torch_mha, to_tensor  # noqa: E402

ref = load_reference(LESSON)
F64 = torch.float64


def weights(seed=42):
    rng = random.Random(seed)
    X = ref.randn_matrix(N_TOKENS, D_MODEL, rng, scale=1.0)
    return X, [ref.randn_matrix(D_MODEL, D_MODEL, rng) for _ in range(4)]


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_output_matches_torch_at_every_head_count(n_heads):
    """The claim: splitting d_model into h heads is what torch does, for any h."""
    rng = random.Random(n_heads)
    X = ref.randn_matrix(N_TOKENS, D_MODEL, rng, scale=1.0)
    Wq, Wk, Wv, Wo = (ref.randn_matrix(D_MODEL, D_MODEL, rng) for _ in range(4))

    mine, _ = ref.multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=n_heads)
    mha = torch.nn.MultiheadAttention(D_MODEL, n_heads, bias=False, batch_first=True).to(F64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(
            torch.cat([to_tensor(w, F64).T for w in (Wq, Wk, Wv)], dim=0)
        )
        mha.out_proj.weight.copy_(to_tensor(Wo, F64).T)
        x = to_tensor(X, F64).unsqueeze(0)
        theirs, _ = mha(x, x, x, need_weights=False)

    torch.testing.assert_close(
        torch.tensor(mine.data, dtype=F64).reshape(mine.rows, mine.cols),
        theirs.squeeze(0), atol=1e-12, rtol=0,
    )


def test_attention_weights_match_per_head_not_just_on_average():
    """A wrong head split still yields a plausible output; the weights expose it."""
    X, (Wq, Wk, Wv, Wo) = weights()
    _, mine = ref.multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=N_HEADS)
    mha = build_torch_mha(Wq, Wk, Wv, Wo, F64)
    x = to_tensor(X, F64).unsqueeze(0)
    with torch.no_grad():
        _, theirs = mha(x, x, x, need_weights=True, average_attn_weights=False)

    for head in range(N_HEADS):
        torch.testing.assert_close(
            torch.tensor(mine[head].data, dtype=F64).reshape(N_TOKENS, N_TOKENS),
            theirs[0, head], atol=1e-12, rtol=0,
        )


def test_attention_weight_rows_are_a_probability_distribution():
    """softmax_rows must give each query a distribution over the keys."""
    X, (Wq, Wk, Wv, Wo) = weights()
    _, mine = ref.multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=N_HEADS)
    for head in mine:
        for i in range(head.rows):
            row = head.row(i)
            assert min(row) >= 0.0
            assert abs(sum(row) - 1.0) < 1e-12


def test_the_1_over_sqrt_dk_scaling_is_actually_applied():
    """Drop the scale and torch stops agreeing -- proof the check has teeth."""
    rng = random.Random(3)
    q = ref.randn_matrix(N_TOKENS, 4, rng, scale=3.0)
    k = ref.randn_matrix(N_TOKENS, 4, rng, scale=3.0)
    v = ref.randn_matrix(N_TOKENS, 4, rng, scale=3.0)
    scaled, _ = ref.scaled_dot_product_attention(q, k, v)
    tq, tk, tv = (to_tensor(m, F64) for m in (q, k, v))

    torch.testing.assert_close(
        torch.tensor(scaled.data, dtype=F64).reshape(scaled.rows, scaled.cols),
        F.scaled_dot_product_attention(tq, tk, tv), atol=1e-12, rtol=0,
    )
    unscaled = F.scaled_dot_product_attention(tq, tk, tv, scale=1.0)
    assert not torch.allclose(F.scaled_dot_product_attention(tq, tk, tv), unscaled)


def test_gqa_with_one_kv_head_per_query_head_is_plain_mha():
    """GQA's own claim: n_kv_heads == n_heads degenerates to multi-head attention."""
    rng = random.Random(11)
    X = ref.randn_matrix(N_TOKENS, D_MODEL, rng, scale=1.0)
    Wq, Wk, Wv, Wo = (ref.randn_matrix(D_MODEL, D_MODEL, rng) for _ in range(4))

    mha_out, _ = ref.multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=N_HEADS)
    gqa_out = ref.grouped_query_attention(
        X, Wq, Wk, Wv, Wo, n_heads=N_HEADS, n_kv_heads=N_HEADS
    )
    assert mha_out.data == pytest.approx(gqa_out.data, abs=1e-15)
