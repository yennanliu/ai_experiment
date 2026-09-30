"""Exercise 3 -- sinusoidal_2d already handles a 7x28 grid; the config and PatchEmbed are what reject 32x16 patches.

    Add support for non-square patch sizes (e.g. 32x16 for wide-aspect inputs) and verify the position table handles non-square grids.

Reading of the exercise: "32x16" is read in torch's (height, width) kernel
order, applied to a wide 224x448 input, which gives a 7x28 grid: 196 tokens,
the lesson's count, in a non-square layout. Support is added without forking
the lesson. The lesson's `VisionFrontEnd` is kept, its `PatchEmbed.forward` is
pointed at a `Conv2d` with kernel and stride (32, 16), and its `pos_embed`
buffer is rebuilt with the lesson's own `sinusoidal_2d(7, 28, 768)`. The
lesson's forward (CLS prepend, position add) runs unchanged. "Handles" is
checked four ways: shape, row and column halves, uniqueness, and whether
token i lines up with patch (i // 28, i % 28).

**ANSWER: the table handles it, and the front end then does too.** The wide
input returns (1, 197, 768). `sinusoidal_2d(7, 28, 768)` is (196, 768); its
row half is constant along each row and its column half is constant along
each column (both to 0.0). All 196 rows are distinct, and the nearest pair is
3.23 apart. A bright pixel block at patch (3, 20) moves token 105 = 1 + 3 * 28 +
20 most, so the table's row-major order matches the Conv2d's `flatten(2)`.
Conv2d and `nn.Unfold((32, 16))` + matmul agree to 1.1e-6.

**FINDING: the lesson rejects non-square input in three places, but not in
the table.** `FrontEndConfig(patch_size=(32, 16)).num_patches` raises
TypeError (`int % tuple`). `PatchEmbed.forward` raises "spatial mismatch" on a
224x448 image. The lesson's `unfold_then_linear` raises TypeError on a tuple
patch. `sinusoidal_2d` takes `grid_h` and `grid_w` separately and needed no
change.

**FINDING: "uniform within a row" is true of every position.** The doc says
the position norms "are uniform within a row, which is the sinusoidal
signature". Each half is sin^2 + cos^2 summed over 192 frequencies, so every
one of the 196 positions has norm sqrt(384) = 19.596, spread 4e-6, in any
grid shape. The norm says nothing about position.

**FINDING: a wide grid is a crop of a square one.** The 7x28 table equals the
first 7 rows of the 28x28 table exactly, and position (r, c) is (c, r) with
its halves swapped. The table encodes absolute indices, not the grid's shape.

Structure: `wide()` adapts the lesson's front end; `probe()` finds which token
a patch lands in; `solve()` runs the checks.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "58-vision-encoder-patches"
IMAGE, PATCH, D = (224, 448), (32, 16), 768
GRID = (IMAGE[0] // PATCH[0], IMAGE[1] // PATCH[1])


def wide(ref):
    torch.manual_seed(0)
    model = ref.VisionFrontEnd(ref.FrontEndConfig(hidden=D)).eval()
    conv = nn.Conv2d(3, D, kernel_size=PATCH, stride=PATCH)
    model.patch.forward = lambda x: conv(x).flatten(2).transpose(1, 2)
    pos = ref.sinusoidal_2d(*GRID, D)
    model.pos_embed = torch.cat([torch.zeros(1, D), pos]).unsqueeze(0)
    return model, conv


def probe(model, row, col):
    img = torch.zeros(1, 3, *IMAGE)
    img[..., row * PATCH[0] : (row + 1) * PATCH[0], col * PATCH[1] : (col + 1) * PATCH[1]] = 1
    with torch.no_grad():
        return int((model(img) - model(torch.zeros_like(img))).norm(dim=-1).argmax())


def errors(ref, img):
    out = []
    for call in (lambda: ref.FrontEndConfig(patch_size=PATCH).num_patches,
                 lambda: ref.VisionFrontEnd(ref.FrontEndConfig(hidden=64))(img),
                 lambda: ref.unfold_then_linear(img, torch.zeros(8, 3, *PATCH), torch.zeros(8), PATCH)):
        try:
            call()
            out.append(None)
        except (TypeError, ValueError) as exc:
            out.append(type(exc).__name__)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model, conv = wide(ref)
    img = torch.rand(1, 3, *IMAGE, generator=torch.Generator().manual_seed(0))
    table = ref.sinusoidal_2d(*GRID, D)
    g = table.reshape(*GRID, D)
    with torch.no_grad():
        spelled = nn.Unfold(PATCH, stride=PATCH)(img).transpose(1, 2) @ conv.weight.reshape(D, -1).T
        unfold_gap = (conv(img).flatten(2).transpose(1, 2) - spelled - conv.bias).abs().max().item()
        shape = tuple(model(img).shape)
    square = ref.sinusoidal_2d(28, 28, D).reshape(28, 28, D)
    norms = table.norm(dim=1)
    return {
        "shape": shape, "table": tuple(table.shape), "unfold_gap": unfold_gap,
        "row_half": (g[..., : D // 2] - g[:, :1, : D // 2]).abs().max().item(),
        "col_half": (g[..., D // 2 :] - g[:1, :, D // 2 :]).abs().max().item(),
        "distinct": len({tuple(r.tolist()) for r in table}),
        "nearest": round(torch.cdist(table, table).add(torch.eye(len(table)) * 1e9).min().item(), 2),
        "probe": probe(model, 3, 20), "errors": errors(ref, img),
        "norm": round(norms.mean().item(), 3), "spread": (norms.max() - norms.min()).item(),
        "doc_uniform": "uniform within a row" in parity.doc_text(PHASE, LESSON),
        "crop": torch.equal(square[: GRID[0]], g),
        "swap": torch.equal(square[2, 5, : D // 2], square[5, 2, D // 2 :]),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the table handles it, and the front end then does too",
            all([
                r["shape"] == (1, 197, D),
                r["table"] == (196, D),
                r["row_half"] == r["col_half"] == 0.0,
                r["distinct"] == 196,
                r["nearest"] == 3.23,
                r["probe"] == 1 + 3 * GRID[1] + 20,
                r["unfold_gap"] < 5e-6,
            ]),
            f"output {r['shape']}, table {r['table']}, row/col half drift "
            f"{r['row_half']}/{r['col_half']}, {r['distinct']} distinct rows, nearest pair "
            f"{r['nearest']}, patch (3, 20) -> token {r['probe']}, Conv2d vs Unfold {r['unfold_gap']:.2g}",
        ),
        practice.Check(
            "FINDING: the lesson rejects non-square input in three places, but not in the table",
            r["errors"] == ["TypeError", "ValueError", "TypeError"],
            f"FrontEndConfig / PatchEmbed.forward / unfold_then_linear: {r['errors']}",
        ),
        practice.Check(
            "FINDING: 'uniform within a row' is true of every position",
            all([r["doc_uniform"], r["norm"] == round(math.sqrt(D / 2), 3), r["spread"] < 1e-4]),
            f"all 196 position norms {r['norm']} (sqrt({D // 2})), spread {r['spread']:.1g}",
        ),
        practice.Check(
            "FINDING: a wide grid is a crop of a square one",
            all([r["crop"], r["swap"]]),
            f"7x28 == first 7 rows of 28x28: {r['crop']}; (2, 5) and (5, 2) swap halves: {r['swap']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
