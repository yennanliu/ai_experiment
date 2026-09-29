"""Exercise 3 — the upper triangle is exactly 0.0, but with dropout on V sees rows that do not sum to 1.

    Add a flag that returns the attention weights for the first head as a `(B, T, T)` tensor. Plot the upper triangle to confirm it is zero after softmax.

Reading of the exercise: the flag is `return_attn=True` on a wrapper around
the lesson's own `MultiHeadAttention.forward`, not an edit to the lesson's
code. The softmax output is exactly the input of `attn_dropout`, so a forward
pre-hook on that module captures it with no attention math copied. Head 0 is
`[:, 0]`. The "plot" is a text heatmap, because the solution runs headless:
`#` for a nonzero weight and `.` for an exact zero. It runs on the lesson's
test shape: d_model 64, 4 heads, B = 2, T = 16, seed 0.

**ANSWER: shape (2, 16, 16), and all 240 upper-triangle entries are exactly
0.0.** They are exact zeros, not small numbers: `masked_fill(-inf)` makes
softmax return 0.0 exactly. Every row sums to 1 to within 1e-6, and token 0
puts weight 1.0 on itself. Batch 0, head 0:

    #...............
    ##..............
    ###.............
    ####............
    (... the staircase continues to row 16)

Turning the flag on leaves the block's output unchanged (difference 0.0).

**FINDING: with dropout on, the weights V is multiplied by are not the
softmax.** The default `BlockConfig` has `attn_dropout=0.1`, and a module
starts in train mode. In that mode the tensor `attn @ v` actually uses has 32
of the 272 causal entries in head 0 zeroed, and its row sums range from 0.0
to 1.11 instead of 1. A row sum of 0.0 means a token read nothing at all. The
exact counts depend on the platform's dropout RNG, so the check asserts
10-60 zeroed entries and row sums on both sides of 1. The upper triangle stays zero, because dropout cannot
turn a zero into anything else. So the mask check passes either way, but only
the pre-dropout capture returns softmax weights. Two train-mode calls also
return different outputs.

Structure: `attention()` is the flagged wrapper; `plot()` draws the heatmap;
`solve()` runs eval mode and train mode on the same input.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "34-transformer-block"
B, T, D_MODEL, HEADS = 2, 16, 64, 4


def attention(mha, x, return_attn=False):
    """`mha(x)`, plus (softmax weights, weights after dropout) for head 0 when flagged."""
    if not return_attn:
        return mha(x)
    seen = {}

    def pre(_module, args):
        seen["softmax"] = args[0][:, 0].detach().clone()

    def post(_module, _args, output):
        seen["dropped"] = output[:, 0].detach().clone()

    hooks = [mha.attn_dropout.register_forward_pre_hook(pre),
             mha.attn_dropout.register_forward_hook(post)]
    try:
        out = mha(x)
    finally:
        for hook in hooks:
            hook.remove()
    return out, seen["softmax"], seen["dropped"]


def plot(weights):
    return "\n".join("".join("." if w == 0.0 else "#" for w in row) for row in weights.tolist())


def upper(weights):
    return weights[:, torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    mha = ref.MultiHeadAttention(ref.BlockConfig(d_model=D_MODEL, num_heads=HEADS,
                                                 context_length=32)).eval()
    x = torch.randn(B, T, D_MODEL, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        plain = attention(mha, x)
        out, weights, _ = attention(mha, x, return_attn=True)
        mha.train()
        torch.manual_seed(1)
        train_out, _, dropped = attention(mha, x, return_attn=True)
        train_again = mha(x)
    lower = ~torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    return {
        "shape": tuple(weights.shape), "upper_n": upper(weights).numel(),
        "upper_max": upper(weights).abs().max().item(),
        "row_err": (weights.sum(-1) - 1).abs().max().item(),
        "self0": weights[0, 0, 0].item(), "plot": plot(weights[0]),
        "flag_changes_out": (out - plain).abs().max().item(),
        "dropped_zeros": int((dropped[:, lower] == 0).sum()),
        "dropped_upper_max": upper(dropped).abs().max().item(),
        "dropped_rows": (round(dropped.sum(-1).min().item(), 2),
                         round(dropped.sum(-1).max().item(), 2)),
        "train_calls_differ": not torch.equal(train_out, train_again),
    }


def verify(result):
    r = result
    stair = "\n".join("#" * (i + 1) + "." * (T - i - 1) for i in range(T))
    return [
        practice.Check(
            "ANSWER: shape (2, 16, 16), and all 240 upper-triangle entries are exactly 0.0",
            all([
                r["shape"] == (B, T, T),
                r["upper_n"] == 240,
                r["upper_max"] == 0.0,
                r["row_err"] < 1e-6,
                r["self0"] == 1.0,
                r["plot"] == stair,
                r["flag_changes_out"] == 0.0,
            ]),
            f"shape {r['shape']}; max of {r['upper_n']} upper entries {r['upper_max']}; row-sum "
            f"error {r['row_err']:.1e}; w[0,0] {r['self0']}; plot is the staircase: "
            f"{r['plot'] == stair}; flag changes output by {r['flag_changes_out']}",
        ),
        practice.Check(
            "FINDING: with dropout on, the weights V is multiplied by are not the softmax",
            all([
                10 <= r["dropped_zeros"] <= 60,
                r["dropped_upper_max"] == 0.0,
                r["dropped_rows"][0] < 0.95,
                r["dropped_rows"][1] > 1.05,
                r["train_calls_differ"],
            ]),
            f"train mode, head 0: {r['dropped_zeros']} of 272 causal weights zeroed; row sums {r['dropped_rows']}; upper max "
            f"{r['dropped_upper_max']}; two train calls differ: {r['train_calls_differ']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
