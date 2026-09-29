"""Exercise 3 — the mask matches SDPA and lesson 61, but nn.MultiheadAttention reads the same tril as look-ahead.

    Implement causal masking as an `attn_mask` argument so the same block can be reused as a decoder block. The mask shape is `(seq, seq)`, lower-triangular.

Reading of the exercise: the lesson's `MultiHeadSelfAttention.forward` and
`Block.forward` take no mask, so both get a replacement `forward` with an
`attn_mask=None` argument, bound onto the lesson's own modules. The math is
the lesson's, plus one `masked_fill`: `attn_mask[i, j]` True means query i
may see key j, the convention lesson 61's `causal_mask` uses. The test block
is width 64, 4 heads, on a (2, 10, 64) input, seed 0.

**ANSWER: `attn_mask` works, and without a mask the block is unchanged.**
With `attn_mask=None` the output equals the lesson's `Block` exactly (max
difference 0.0). With `torch.tril` of shape (10, 10) every weight above the
diagonal is exactly 0, every row still sums to 1, and changing tokens 5-9
leaves outputs 0-4 exactly unchanged. Without the mask the same change
moves output 0 by 0.38. The masked attention matches
`F.scaled_dot_product_attention(is_causal=True)` and lesson 61's
`CausalSelfAttention` with the same weights, both to within 1e-6.

**FINDING: nn.MultiheadAttention reads the same tril as the opposite
mask.** For a boolean `attn_mask` it treats True as "not allowed". Given
the (10, 10) tril, row 0 attends only to tokens 1-9, the first 9 rows put
all of their weight on future tokens, and row 9, whose mask is all True, is
NaN. The shape `(seq, seq)` and "lower-triangular" do not fix the meaning;
the True/False convention has to be stated too.

**FINDING: the lesson's pointer to lesson 61 is wrong.** It says "the
decoder-side cross-attention in lesson 61 will use a causal mask". Lesson
61's docstring says cross-attention uses no mask, and its `CrossAttention`
forward has no mask argument. The mask goes on the decoder's
self-attention. A `(seq, seq)` mask does not fit cross-attention scores:
with 10 text queries against 197 image keys they are (10, 197).

Structure: `attend()` and `block_forward()` are the masked forwards;
`solve()` runs them against the lesson's forward, SDPA, lesson 61 and
`nn.MultiheadAttention`.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON, L61 = "19-capstone-projects", "59-vit-transformer", "61-cross-attention-fusion"
D, H, N = 64, 4, 10


def attend(self, x, store_attn=False, attn_mask=None):
    """The lesson's attention math, plus attn_mask (True = may attend)."""
    b, n, d = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.cfg.heads, self.cfg.head_dim).permute(2, 0, 3, 1, 4)
    scores = (qkv[0] @ qkv[1].transpose(-2, -1)) * self.scale
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    self.last_attn = attn.detach() if store_attn else self.last_attn
    return self.drop(self.out((attn @ qkv[2]).transpose(1, 2).reshape(b, n, d)))


def block_forward(self, x, store_attn=False, attn_mask=None):
    x = x + attend(self.attn, self.ln1(x), store_attn, attn_mask)
    return x + self.ffn(self.ln2(x))


def lesson61_attention(r61, attn):
    torch.manual_seed(0)
    other = r61.CausalSelfAttention(r61.DecoderConfig(hidden=D, heads=H)).eval()
    other.load_state_dict(attn.state_dict())
    return other


def solve():
    ref, r61 = parity.load_reference(PHASE, LESSON, "main"), parity.load_reference(PHASE, L61, "main")
    torch.manual_seed(0)
    block = ref.Block(ref.ViTConfig(hidden=D, heads=H)).eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, N, D, generator=g)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(2, N - 5, D, generator=g)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    with torch.no_grad():
        attend(block.attn, x, store_attn=True, attn_mask=mask)
        w = block.attn.last_attn
        causal, causal2 = block_forward(block, x, attn_mask=mask), block_forward(block, x2, attn_mask=mask)
        qkv = block.attn.qkv(x).reshape(2, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        sdpa = block.attn.out(F.scaled_dot_product_attention(*qkv, is_causal=True).transpose(1, 2)
                              .reshape(2, N, D))
        mha = torch.nn.MultiheadAttention(D, H, batch_first=True)
        mw = mha(x, x, x, attn_mask=mask)[1][0]
        return {
            "none_vs_lesson": (block_forward(block, x) - block(x)).abs().max().item(),
            "upper": w.triu(1).abs().max().item(), "row_err": (w.sum(-1) - 1).abs().max().item(),
            "past_moved": (causal2[:, :5] - causal[:, :5]).abs().max().item(),
            "unmasked_moved": (block(x2)[:, 0] - block(x)[:, 0]).abs().max().item(),
            "vs_sdpa": (attend(block.attn, x, attn_mask=mask) - sdpa).abs().max().item(),
            "vs_l61": (attend(block.attn, x, attn_mask=mask)
                       - lesson61_attention(r61, block.attn)(x, r61.causal_mask(N))).abs().max().item(),
            "mha_row0": mw[0].nonzero().flatten().tolist(),
            "all_future": bool(((mw[:9].triu(1).sum(-1) - 1).abs() < 1e-6).all()),
            "mha_future": mw[:9].triu(1).sum(-1).round(decimals=3).tolist(),
            "mha_nan_rows": mw.isnan().all(-1).sum().item(),
            "l61_doc": "cross-attention uses no mask" in inspect.getdoc(r61),
            "cross_params": list(inspect.signature(r61.CrossAttention.forward).parameters),
            "cross_mask": any("mask" in p for p in inspect.signature(r61.CrossAttention.forward).parameters),
            "doc_claim": "cross-attention in lesson 61 will use a causal mask"
            in parity.doc_text(PHASE, LESSON),
        }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: attn_mask works, and without a mask the block is unchanged",
            all([r["none_vs_lesson"] == 0.0, r["upper"] == 0.0, r["row_err"] < 1e-6, r["past_moved"] == 0.0,
                 r["unmasked_moved"] > 0.1, r["vs_sdpa"] < 1e-6, r["vs_l61"] < 1e-6]),
            f"no mask vs lesson {r['none_vs_lesson']}; weights above diagonal {r['upper']}; "
            f"row-sum error {r['row_err']:.1e}; outputs 0-4 moved {r['past_moved']} masked, "
            f"output 0 moved {r['unmasked_moved']:.2f} unmasked; vs SDPA {r['vs_sdpa']:.1e}, "
            f"vs lesson 61 {r['vs_l61']:.1e}",
        ),
        practice.Check(
            "FINDING: nn.MultiheadAttention reads the same tril as the opposite mask",
            all([r["mha_row0"] == list(range(1, N)), r["all_future"], r["mha_nan_rows"] == 1]),
            f"row 0 attends to {r['mha_row0']}; weight on future tokens, rows 0-8: "
            f"{r['mha_future']}; all-NaN rows {r['mha_nan_rows']}",
        ),
        practice.Check(
            "FINDING: the lesson's pointer to lesson 61 is wrong",
            all([r["doc_claim"], r["l61_doc"], not r["cross_mask"]]),
            f"doc says lesson 61 cross-attention is causal: {r['doc_claim']}; lesson 61 says "
            f"'cross-attention uses no mask': {r['l61_doc']}; CrossAttention.forward{r['cross_params']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
