"""Exercise 2 — on the lesson's own grid, not one frequency band completes a cycle.

    **Medium.** Implement 2D sinusoidal positional embeddings -- two independent
    sinusoidal codes for `row` and `col` of each patch, concatenated. Feed them
    into a tiny PyTorch ViT and compare accuracy vs learnable positional
    embeddings on CIFAR-10.

Reading of the exercise: `pos_2d` already is that code -- `d_model // 2` for the
row, `d_model // 2` for the column, concatenated -- so "implement" is read as
*audit it*, and the PyTorch comparison is unbuildable (`torch` returns None from
`find_spec`, there is no CIFAR-10 and no network). What the comparison would be
measuring is measured directly instead: how much positional resolution the two
schemes have on the grids these models actually use.

**ANSWER: the two codes are independent and the concatenation is exactly
separable.** `pe[i][j] . pe[i'][j']` depends only on `(i' - i, j' - j)` -- worst
spread **3.6e-15** across 49 distinct offsets on the lesson's 4x4 grid -- because
the row half and the column half share no dimensions, so their dot products add.
Every position also has the same norm, `sqrt(d_model/2)` to twelve places.

**FINDING: not one band completes a cycle on the lesson's own grid.** A band's
wavelength is `2*pi*10000^(2k/half)`, so the fastest is 6.28 positions and the
grid is **4**. At `d_model=48` that is **0 of 12** bands -- the entire positional
code is a monotone ramp in each axis, with no periodicity anywhere in it.

**FINDING: a real ViT is barely better.** ViT-Base/16 on 224x224 has a 14x14
grid and **17 of 192** bands complete a cycle -- 8.9%. The other 91% of the
positional dimensions traverse less than one period across the whole image, so
they carry a slope and not a position. Sinusoidal coding was designed for
sequences of hundreds; a patch grid is 14 wide.

**FINDING: the comparison the exercise sets up is 0.17% of the parameters.**
Learnable positions cost `(num_patches + 1) * d_model` = 151,296 at ViT-Base/16
against a total of 86.5M. Sinusoidal costs zero. That is the entire budget being
traded, which is why the ViT paper's own ablation finds the two within noise.

**CONTROL: `d_model % 4 == 0` is asserted and it is the binding constraint.**
`half // 2` pairs per axis need `half` even, so `d_model` must be a multiple of
4; at 48 the code fills dimensions 0-23 from the row and 24-47 from the column
with nothing left over.

Structure: `wavelength` is the closed form; `covered` counts bands that fit in a
grid; `separable` measures the offset-only property.
"""

from __future__ import annotations

import importlib.util
import math

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "09-vision-transformers"
GRID, D_MODEL, BASE = 4, 48, 10000.0
REAL_GRID, REAL_WIDTH, REAL_TOTAL = 14, 768, 86.5e6


def wavelength(band, width, base=BASE):
    """Positions per cycle for band k of a `width`-wide sinusoidal code."""
    return 2 * math.pi * base ** (2 * band / width)


def covered(grid, width):
    """(bands completing a cycle across `grid`, bands available)."""
    half = width // 2
    return sum(1 for k in range(half // 2) if wavelength(k, half) <= grid), half // 2


def separable(code, grid=GRID):
    """Worst spread of pe[i][j] . pe[i2][j2] within one (drow, dcol) offset."""
    offsets = {}
    for i in range(grid):
        for j in range(grid):
            for i2 in range(grid):
                for j2 in range(grid):
                    dot = sum(a * b for a, b in zip(code[i][j], code[i2][j2]))
                    offsets.setdefault((i2 - i, j2 - j), []).append(dot)
    return max(max(v) - min(v) for v in offsets.values()), len(offsets)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    code = ref.pos_2d(GRID, GRID, D_MODEL)
    norms = [math.sqrt(sum(v * v for v in code[i][j])) for i in range(GRID) for j in range(GRID)]
    spread, offsets = separable(code)
    try:
        ref.pos_2d(GRID, GRID, D_MODEL - 2)
        odd = True
    except AssertionError:
        odd = False
    return {
        "spread": spread, "offsets": offsets, "norms": (min(norms), max(norms)),
        "target": math.sqrt(D_MODEL / 2), "fastest": wavelength(0, D_MODEL // 2),
        "toy": covered(GRID, D_MODEL), "real": covered(REAL_GRID, REAL_WIDTH),
        "learnable": (REAL_GRID ** 2 + 1) * REAL_WIDTH, "total": REAL_TOTAL,
        "torch": importlib.util.find_spec("torch") is None, "odd": odd,
        "halves": (len(code[0][0]), D_MODEL // 2),
    }


def verify(result):
    low, high = result["norms"]
    toy, real = result["toy"], result["real"]
    return [
        practice.Check(
            "ANSWER: the two codes are independent, so the dot product is offset-only",
            result["spread"] < 1e-13 and low == high == result["target"],
            f"pe[i][j] . pe[i2][j2] varies by at most {result['spread']:.1e} within a "
            f"(drow, dcol) offset, across {result['offsets']} offsets on the {GRID}x{GRID} grid: "
            "the row half and the column half share no dimensions, so their dot products add. "
            f"Every position also has norm {low:.12f} = sqrt(d_model/2)",
        ),
        practice.Check(
            "FINDING: not one band completes a cycle on the lesson's own grid",
            toy[0] == 0,
            f"the fastest band has a wavelength of {result['fastest']:.2f} positions and the grid "
            f"is {GRID}: {toy[0]} of {toy[1]} bands complete a cycle at d_model={D_MODEL}. The "
            "whole positional code is a monotone ramp in each axis, with no periodicity in it",
        ),
        practice.Check(
            "FINDING: a real ViT is barely better -- 17 of 192",
            real[0] / real[1] < 0.1,
            f"ViT-Base/16 on 224x224 has a {REAL_GRID}x{REAL_GRID} grid: {real[0]} of {real[1]} "
            f"bands complete a cycle, {real[0] / real[1]:.1%}. The other "
            f"{1 - real[0] / real[1]:.0%} traverse less than one period across the whole image, "
            "so they carry a slope and not a position. Sinusoidal coding was built for sequences "
            "of hundreds; a patch grid is 14 wide",
        ),
        practice.Check(
            "FINDING: the comparison is over 0.17% of the parameters",
            result["learnable"] / result["total"] < 0.005,
            f"learnable positions cost (num_patches + 1) * d_model = {result['learnable']:,} at "
            f"ViT-Base/16, against {result['total'] / 1e6:.1f}M total -- "
            f"{result['learnable'] / result['total']:.2%}. Sinusoidal costs zero. That is the "
            "entire budget being traded, which is why the ViT paper's own ablation finds the two "
            "within noise of each other",
        ),
        practice.Check(
            "CONTROL: the PyTorch half is unbuildable, and d_model % 4 is the binding constraint",
            result["torch"] and not result["odd"] and result["halves"][0] == D_MODEL,
            f"find_spec('torch') is None, there is no CIFAR-10 and no network. pos_2d asserts "
            f"d_model % 4 == 0 and rejects {D_MODEL - 2}: half // 2 pairs per axis need half "
            f"even, and at {D_MODEL} the code fills 0-{D_MODEL // 2 - 1} from the row and the "
            "rest from the column with nothing left over",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
