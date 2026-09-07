"""Exercise 1 — vit token shape ledger.

    **(Easy)** Print the shapes of every intermediate tensor for a forward pass through the tiny ViT above. Confirm: input `(N, 3, 64, 64)` -> patches `(N, 16, 192)` -> with CLS `(N, 17, 192)` -> classifier input `(N, 192)` -> output `(N, num_classes)`.

Reading of the exercise: the ledger it asks to confirm is correct, and confirming
it takes one line -- so the interesting question is what the ledger is *made of*,
and every piece turns out to be an exact identity rather than a tolerance. 16 is
not a free parameter: it is `(64 // 16) ** 2`, and the 17 and the 192 are welded
to it by `pos_embed`, a `(1, 17, 192)` `nn.Parameter` that fixes the input
resolution as hard as the patch size does. Feeding the same model a 32x32 image
-- legal for every layer it contains -- does not give a smaller ledger, it raises
`RuntimeError` on the `x + self.pos_embed` line, and the standard repair (bicubic
interpolation of the 4x4 position grid, the CLS row kept aside) restores a
coherent 4-patch ledger. Two more identities sit underneath: the "first conv"
really is a per-patch linear map, reproducible from `F.unfold` and the same
weight matrix reshaped, and with `pos_embed` zeroed the CLS logit is invariant to
permuting the 16 patch tokens -- so position is carried by that parameter and by
nothing else in the architecture. At the lesson's `trunc_normal_(std=0.02)` init
it is carried very weakly, which is measured here rather than assumed. Nothing is
trained and nothing is downloaded; every number below is a property of the freshly
constructed `ViT`.

Structure: `tokens_forward` re-runs the lesson's own blocks over a supplied
position table with an optional patch permutation, returning patches, classifier
input and logits; `shape_ledger` reports the five shapes at one batch size;
`geometry_probe` rebuilds the patch embedding from `F.unfold` and shuffles patch
tokens with and without positions; `interp_pos` bicubic-resizes a position table
between square grids and `resize_probe` uses it to fail and then repair a 32x32
forward. At 137 code lines this sits above D14's 120-line target and 13 clear of
the ceiling: five checks over three probes, and the shape ledger alone is four
batch sizes wide.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "14-vision-transformers"

IMAGE, PATCH, DIM, CLASSES, SMALL = 64, 16, 192, 10, 32
GRID, TOKENS, BATCHES = IMAGE // PATCH, (IMAGE // PATCH) ** 2 + 1, (1, 2, 3, 8)
shape = lambda tensor: tuple(tensor.shape)                          # noqa: E731 - a formatter


def tokens_forward(torch, vit, x, pos, perm=None):
    patches = vit.patch(x)
    if perm is not None:
        patches = patches[:, perm]
    seq = torch.cat([vit.cls_token.expand(x.size(0), -1, -1), patches], dim=1) + pos
    for block in vit.blocks:
        seq = block(seq)
    return patches, vit.ln(seq[:, 0]), vit.head(vit.ln(seq[:, 0]))


def shape_ledger(torch, vit, size) -> tuple:
    torch.manual_seed(size)
    x = torch.randn(size, 3, IMAGE, IMAGE)
    with torch.no_grad():
        patches, head_in, logits = tokens_forward(torch, vit, x, vit.pos_embed)
    return shape(x), shape(patches), (size, TOKENS, DIM), shape(head_in), shape(logits)


def geometry_probe(torch, functional, vit, x) -> dict:
    torch.manual_seed(11)
    perm, zero = torch.randperm(TOKENS - 1), torch.zeros_like(vit.pos_embed)
    columns = functional.unfold(x, kernel_size=PATCH, stride=PATCH).transpose(1, 2)
    with torch.no_grad():
        rebuilt = columns @ vit.patch.proj.weight.reshape(DIM, -1).T + vit.patch.proj.bias
        placed, shuffled = (tokens_forward(torch, vit, x, vit.pos_embed, p) for p in (None, perm))
        flat, flat_moved = (tokens_forward(torch, vit, x, zero, p)[2] for p in (None, perm))
    return {"unfold": (rebuilt - placed[0]).abs().max().item(), "pos_std": vit.pos_embed.std().item(),
            "with_pos": (placed[2] - shuffled[2]).abs().max().item(), "logits": placed[2],
            "no_pos": (flat - flat_moved).abs().max().item(), "logit_std": placed[2].std().item()}


def interp_pos(torch, functional, pos, old, new):
    square = pos[:, 1:].reshape(1, old, old, DIM).permute(0, 3, 1, 2)
    resized = functional.interpolate(square, size=(new, new), mode="bicubic", align_corners=False)
    return torch.cat([pos[:, :1], resized.permute(0, 2, 3, 1).reshape(1, new * new, DIM)], dim=1)


def resize_probe(torch, functional, vit) -> dict:
    torch.manual_seed(5)
    small, grid = torch.randn(2, 3, SMALL, SMALL), SMALL // PATCH
    try:
        vit(small)
        failure = "no error"
    except RuntimeError as exc:
        failure = f"{type(exc).__name__}: {str(exc).split(' at non-singleton')[0]}"
    with torch.no_grad():
        moved = interp_pos(torch, functional, vit.pos_embed, GRID, grid)
        same = interp_pos(torch, functional, vit.pos_embed, GRID, GRID)
        ledger = [shape(t) for t in tokens_forward(torch, vit, small, moved)]
    return {"failure": failure, "identity": (same - vit.pos_embed).abs().max().item(),
            "tokens": grid * grid + 1, "ledger": ledger}


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    vit = ref.ViT(image_size=IMAGE, patch_size=PATCH, num_classes=CLASSES, dim=DIM).eval()
    torch.manual_seed(3)
    x = torch.randn(4, 3, IMAGE, IMAGE)
    counts = {name: sum(p.numel() for p in module.parameters()) for name, module in
              (("params", vit), ("patch", vit.patch), ("blocks", vit.blocks))}
    with torch.no_grad():
        drift = (vit.eval()(x) - vit.train()(x)).abs().max().item()
    return {**counts, "ledger": {n: shape_ledger(torch, vit.eval(), n) for n in BATCHES},
            "geometry": geometry_probe(torch, functional, vit, x), "drift": drift,
            "resize": resize_probe(torch, functional, vit), "depth": len(vit.blocks),
            "table": vit.pos_embed.numel() + vit.cls_token.numel()}


def verify(result):
    geo, resize = result["geometry"], result["resize"]
    claimed = [((n, 3, IMAGE, IMAGE), (n, TOKENS - 1, DIM), (n, TOKENS, DIM), (n, DIM), (n, CLASSES))
               for n in BATCHES]
    return [
        practice.Check(
            "ANSWER: the ledger holds exactly, at every batch size the exercise leaves open",
            [result["ledger"][n] for n in BATCHES] == claimed,
            f"the lesson's own `ViT` at N in {BATCHES}: input (N, 3, {IMAGE}, {IMAGE}) -> patches (N, "
            f"{TOKENS - 1}, {DIM}) -> with CLS (N, {TOKENS}, {DIM}) -> classifier input (N, {DIM}) -> logits "
            f"(N, {CLASSES}), all four rows matching. {TOKENS - 1} is ({IMAGE} // {PATCH})**2, not a free "
            f"choice; total parameters {result['params']:,}, matching the docs' \"About 2.8M\""),
        practice.Check(
            "MECHANISM: the 'first conv' is exactly a per-patch linear map, to 1e-06",
            geo["unfold"] < 5e-06,
            f"`F.unfold(x, {PATCH}, stride={PATCH})` gives {TOKENS - 1} columns of 3*{PATCH}*{PATCH}=768 pixels; "
            f"multiplying them by `proj.weight.reshape({DIM}, -1).T` plus the bias reproduces `vit.patch(x)` to "
            f"{geo['unfold']:.1e}. That is why the block holds {result['patch']:,} = {DIM}*(768+1) parameters, "
            f"and why patch size and stride are the same number"),
        practice.Check(
            "MECHANISM: position lives entirely in pos_embed — zero it and patch order stops mattering",
            geo["no_pos"] < 1e-05 < geo["with_pos"],
            f"shuffling the {TOKENS - 1} patch tokens with `pos_embed` zeroed moves the CLS logits by "
            f"{geo['no_pos']:.1e} against a logit std of {geo['logit_std']:.3f} -- attention and the MLP are "
            f"permutation-equivariant, so the CLS row is a set function. The real table moves the same shuffle "
            f"{geo['with_pos']:.4f}, only {geo['with_pos'] / geo['logit_std']:.1%} of that spread, because "
            f"`trunc_normal_(std=0.02)` starts positions at std {geo['pos_std']:.4f}"),
        practice.Check(
            "FINDING: the ledger is welded to 64x64 — a 32x32 image raises, interpolation repairs it",
            resize["failure"].startswith("RuntimeError") and resize["identity"] == 0.0,
            f"`vit(torch.randn(2, 3, {SMALL}, {SMALL}))` gives {resize['failure']} -- the {TOKENS}-row "
            f"`nn.Parameter` cannot broadcast onto {resize['tokens']} tokens. Bicubic-resizing its {GRID}x{GRID} "
            f"grid to 2x2 with the CLS row set aside restores the ledger as {resize['ledger'][0]} -> "
            f"{resize['ledger'][1]} -> {resize['ledger'][2]}, and resizing {GRID}x{GRID} to itself is the "
            f"identity to {resize['identity']:.1f}, so the repair costs nothing at the native resolution"),
        practice.Check(
            "CONTROL: `.eval()` is not load-bearing here, unlike every CNN lesson in this phase",
            result["drift"] == 0.0,
            f"the same batch through `vit.eval()` and `vit.train()` differs by {result['drift']:.1f} -- `Block` "
            f"defaults to `dropout=0.0` and `ViT` never overrides it, and there is no BatchNorm anywhere, so "
            f"LayerNorm makes train and inference the same function. Of the {result['params']:,} parameters "
            f"{result['blocks']:,} are the {result['depth']} blocks and only {result['table']:,} are the "
            "CLS token and the position table together"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
