"""Exercise 1 — patchify builds a grid, then returns a different one.

    **Easy.** Run `code/main.py`. Verify the number of patches equals
    `(H/P) * (W/P)` and the flat patch dimension equals `P*P*C`.

Reading of the exercise: both identities are checked against the lesson's own
`patchify` at `main()`'s settings, and then the function's source is read to see
what it does to arrive at them.

**ANSWER: both hold.** 24x24x3 at patch 6 gives **16** patches of **108** each,
against `(24/6) * (24/6) = 16` and `6*6*3 = 108`.

**FINDING: `patchify` assembles a grid of `(row, col)` pairs and discards it.**
`grid` and `grid_row` are built cell by cell through both loops and then never
reach the `return`, which recomputes the shape as
`(H // patch_size, W // patch_size)`. `C = len(image[0][0])` is assigned and
never read at all -- the channel count the exercise asks you to verify is
computed by the function and then dropped.

**FINDING: the `[CLS]` token gets no positional encoding.** `cls_and_pos` writes
`out = [list(cls)]` and adds `pe` only inside the patch loop, so token 0 carries
position information from nowhere. Real ViT gives `[CLS]` its own learned
position, and `param_count_vit` charges for one -- `(num_patches + 1) * d_model`
-- but `pos_2d` never produces it.

**FINDING: the parameter model counts learnable positions the code does not
have.** `param_count_vit` bills `(num_patches + 1) * d_model` of position
embeddings -- 151,296 at ViT-Base/16 -- while `cls_and_pos` uses a sinusoidal
code with **zero** parameters. The two halves of the lesson describe different
models.

**CONTROL: the parameter model is otherwise accurate to 0.1%.** 86.5M against a
published 86.6M for ViT-Base/16, 304.1M against 304.0M for ViT-Large/16, 631.7M
against 632.0M for ViT-Huge/14. It is right about the real ViT, which is how the
position-embedding line got there.

Structure: `dead` reads the source for names assigned and never loaded;
`discarded` asks which locals the return value can reach.
"""

from __future__ import annotations

import ast
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "09-vision-transformers"
SIDE, CHANNELS, PATCH, D_MODEL = 24, 3, 6, 48
PUBLISHED = {"ViT-Base/16": (768, 12, 16, 86.6), "ViT-Large/16": (1024, 24, 16, 304.0),
             "ViT-Huge/14": (1280, 32, 14, 632.0)}


def names(tree, context):
    """Every identifier used in this AST under the given Load/Store context."""
    return {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, context)}


def dead(function):
    """Locals assigned by this function and never read back."""
    tree = ast.parse(inspect.getsource(function))
    return sorted(names(tree, ast.Store) - names(tree, ast.Load))


def discarded(function):
    """Named containers the function fills that the return value cannot reach."""
    tree = ast.parse(inspect.getsource(function))
    returned = {node.value for node in ast.walk(tree) if isinstance(node, ast.Return)}
    reached = set().union(*(names(value, ast.Load) for value in returned))
    built = {node.targets[0].id for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.List)}
    return sorted(built - reached)


def sizes(ref):
    """Modelled vs published parameter counts, in millions."""
    out = {}
    for name, (width, layers, patch, published) in PUBLISHED.items():
        grid = (224 // patch) ** 2
        modelled = ref.param_count_vit(width, layers, 12, 4, grid, 1000)
        out[name] = ((modelled + patch * patch * 3 * width) / 1e6, published)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    image = ref.make_image(SIDE, SIDE, CHANNELS, seed=0)
    patches, grid = ref.patchify(image, PATCH)
    tokens, _ = ref.linear_project(patches, D_MODEL, random.Random(42))
    sequence = ref.cls_and_pos(tokens, grid[0], grid[1], random.Random(7))
    code = ref.pos_2d(grid[0], grid[1], D_MODEL)
    width, _, patch, _ = PUBLISHED["ViT-Base/16"]
    return {
        "patches": len(patches), "expected": (SIDE // PATCH) ** 2,
        "flat": len(patches[0]), "flat_expected": PATCH * PATCH * CHANNELS,
        "grid": grid, "sequence": len(sequence), "dead": dead(ref.patchify),
        "discarded": discarded(ref.patchify),
        "cls_plain": sequence[1] != tokens[0] and all(
            abs(sequence[1][k] - tokens[0][k] - code[0][0][k]) < 1e-15 for k in range(D_MODEL)),
        "cls_unmoved": sequence[0] != [t + p for t, p in zip(sequence[0], code[0][0])],
        "billed": ((224 // patch) ** 2 + 1) * width, "sizes": sizes(ref),
    }


def verify(result):
    sizes_seen = result["sizes"]
    worst = max(abs(m - p) / p for m, p in sizes_seen.values())
    return [
        practice.Check(
            "ANSWER: 16 patches of 108, matching (H/P)*(W/P) and P*P*C",
            result["patches"] == result["expected"] and result["flat"] == result["flat_expected"],
            f"{SIDE}x{SIDE}x{CHANNELS} at patch {PATCH}: {result['patches']} patches against "
            f"({SIDE}/{PATCH})^2 = {result['expected']}, each {result['flat']} long against "
            f"{PATCH}*{PATCH}*{CHANNELS} = {result['flat_expected']}. The returned grid is "
            f"{result['grid']} and the sequence is {result['sequence']} with [CLS]",
        ),
        practice.Check(
            "FINDING: patchify assembles a grid of (row, col) pairs and discards it",
            "grid" in result["discarded"] and "grid_row" in result["discarded"],
            f"lists the function fills that the return value cannot reach: "
            f"{result['discarded']}. `grid` is filled cell "
            "by cell through both loops and then never reaches the return, which recomputes the "
            "shape as (H // patch_size, W // patch_size) from scratch",
        ),
        practice.Check(
            "FINDING: C is assigned and never read",
            result["dead"] == ["C"],
            f"names assigned and never loaded: {result['dead']}. The channel count the exercise "
            "asks you to verify is computed by the function on line 3 and dropped -- the assert "
            "below it checks H and W only, so nothing would change if the line were deleted",
        ),
        practice.Check(
            "FINDING: the [CLS] token gets no positional encoding",
            result["cls_plain"] and result["cls_unmoved"],
            "cls_and_pos writes out = [list(cls)] and adds pe only inside the patch loop, so "
            "token 0 carries position from nowhere while token 1 is exactly its patch embedding "
            f"plus pe[0][0]. Real ViT gives [CLS] a learned position and param_count_vit charges "
            f"for one: {result['billed']:,} at ViT-Base/16 for (num_patches + 1) * d_model",
        ),
        practice.Check(
            "CONTROL: the parameter model is accurate to 0.1% even so",
            worst < 0.002,
            "modelled vs published, in millions: " + ", ".join(
                f"{name} {m:.1f}/{p:.1f}" for name, (m, p) in sizes_seen.items())
            + f" -- worst {worst:.2%}. It is right about the real ViT, which uses learnable "
              "positions, and that is how the line the code contradicts got there",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
