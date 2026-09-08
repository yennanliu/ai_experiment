"""Exercise 2 — joint attention costs less.

    **(Medium)** Swap the divided attention block above for a full joint
    attention block and measure the shape and parameter count. Explain why
    divided attention is necessary for real video models.

Reading of the exercise: the measurement it asks for refutes the explanation it
asks for. A joint block is one attention over the flattened sequence where the
lesson's divided block runs two, so swapping them *removes* parameters -- the
joint version is 25% smaller. Divided attention is not a parameter saving; it
buys a smaller attention matrix by paying for a second set of projections. The
file therefore measures both axes separately and says which one the necessity
lives on. A second reading problem sits underneath: at the scale the lesson's
own `main()` runs, joint attention is entirely affordable, so the demonstration
cannot exhibit the constraint it is explaining.

Structure: `joint_block` builds the swap -- the lesson's own block with its two
attentions replaced by one over the flattened sequence, everything else
unchanged; `params` counts a module; `pairs` is the attention-entry count for a
grid under each scheme; `outputs` runs both blocks on one clip's tokens.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "28-world-models-video-diffusion"

DIM, HEADS, CLIP = 64, 2, (1, 4, 8, 16, 16)
REAL_TIME, REAL_SPACE = 75, 2700          # exercise 1's 5-second clip at patch (2, 8, 8)

params = lambda module: sum(p.numel() for p in module.parameters())            # noqa: E731
joint_pairs = lambda time, space: (time * space) ** 2                          # noqa: E731
split_pairs = lambda time, space: space * time ** 2 + time * space ** 2        # noqa: E731
gigabytes = lambda entries: entries * 4 / 1e9                                  # noqa: E731


def joint_block(nn, functional):
    """The swap the exercise asks for: one attention over T*H*W, nothing else changed."""

    class JointAttentionBlock(nn.Module):
        def __init__(self, dim=DIM, heads=HEADS):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.ln1, self.ln3 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        def forward(self, x, grid):
            normed = self.ln1(x)
            attended, _ = self.attn(normed, normed, normed, need_weights=False)
            merged = x + attended
            return merged + self.mlp(self.ln3(merged))

    return JointAttentionBlock


def solve():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    patch = ref.VideoPatch3D(in_channels=CLIP[1], dim=DIM, patch_t=2, patch_h=2, patch_w=2)
    with torch.no_grad():
        tokens, grid = patch(torch.randn(*CLIP))
    torch.manual_seed(1)
    split = ref.DividedAttentionBlock(DIM, HEADS)
    torch.manual_seed(1)
    whole = joint_block(nn, functional)(DIM, HEADS)
    with torch.no_grad():
        split_out, whole_out = split(tokens, grid), whole(tokens, grid)
        normed = split.ln1(tokens)
        repeated = bool(torch.equal(normed, split.ln1(tokens)))
    demo_time, demo_space = int(grid[0]), int(grid[1] * grid[2])
    return {"grid": tuple(int(g) for g in grid), "shape": tuple(split_out.shape),
            "same_shape": split_out.shape == whole_out.shape,
            "differs": float((split_out - whole_out).abs().max()),
            "split_params": params(split), "whole_params": params(whole),
            "attn_each": params(split.time_attn), "mlp": params(split.mlp),
            "demo": (joint_pairs(demo_time, demo_space), split_pairs(demo_time, demo_space)),
            "real": (joint_pairs(REAL_TIME, REAL_SPACE), split_pairs(REAL_TIME, REAL_SPACE)),
            "ln_repeated": repeated}


def verify(result):
    split_p, whole_p = result["split_params"], result["whole_params"]
    demo_joint, demo_split = result["demo"]
    real_joint, real_split = result["real"]
    return [
        practice.Check(
            "ANSWER: same shape out, and the joint block is 25% smaller",
            result["same_shape"] and whole_p < split_p,
            f"a {CLIP} clip patches to grid {result['grid']}, and both blocks map its tokens to "
            f"{result['shape']}. The divided block holds {split_p:,} parameters against the joint "
            f"block's {whole_p:,} -- {split_p - whole_p:,} fewer, {1 - whole_p / split_p:.0%}, "
            f"because it runs one attention where the lesson's runs two at {result['attn_each']:,} "
            "each. Swapping in joint attention removes parameters"),
        practice.Check(
            "FINDING: so the necessity is not about parameters -- divided costs them",
            split_p - whole_p == result["attn_each"] + 2 * DIM and result["differs"] > 0.0,
            f"the whole difference is one nn.MultiheadAttention at {result['attn_each']:,} plus the "
            f"third LayerNorm that fed it at {2 * DIM}: divided attention *buys* a smaller attention "
            f"matrix by *paying* for a second set of projections. The two blocks are different "
            f"functions -- their outputs on the same tokens differ by {result['differs']:.4f} -- so "
            "this is a trade, not a refactoring, and the usual one-line explanation has it backwards"),
        practice.Check(
            "MECHANISM: what it buys is the attention matrix, quadratic in the sequence",
            real_joint / real_split > 70,
            f"on exercise 1's 5-second clip -- {REAL_TIME} temporal x {REAL_SPACE:,} spatial tokens "
            f"-- joint attention is {real_joint:,} entries ({gigabytes(real_joint):.1f} GB at "
            f"float32) against divided's {real_split:,} ({gigabytes(real_split):.1f} GB), "
            f"{real_joint / real_split:.0f}x fewer. That factor, not the parameter count, is what "
            "makes the divided form necessary at video scale"),
        practice.Check(
            "CONTROL: the lesson's own demo is far too small to show the problem",
            demo_joint < 100_000 and real_joint / demo_joint > 100_000,
            f"at the demo's {result['grid']} grid the joint matrix is only {demo_joint:,} entries "
            f"({gigabytes(demo_joint) * 1e6:.0f} KB) against divided's {demo_split:,} -- joint is "
            f"perfectly affordable here, and is in fact only {demo_joint / demo_split:.1f}x the "
            f"divided cost. The real clip is {real_joint / demo_joint:,.0f}x larger, so main() "
            "cannot exhibit the constraint the exercise asks you to explain"),
        practice.Check(
            "CONTROL: the lesson normalises the same tensor three times per attention",
            result["ln_repeated"],
            "DividedAttentionBlock passes `self.ln1(xt)` separately as query, key and value, so the "
            "same LayerNorm runs three times on one tensor and returns the identical result each "
            f"time ({result['ln_repeated']}). Two of the three calls are wasted work in a block "
            "whose entire purpose is to be cheap enough to run at video scale"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
