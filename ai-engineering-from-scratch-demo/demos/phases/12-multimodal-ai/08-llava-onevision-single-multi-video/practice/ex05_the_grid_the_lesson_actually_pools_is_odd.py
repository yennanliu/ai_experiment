"""Exercise 5 — the grid the lesson actually pools is odd.

    Bilinear pooling of 24x24 patches to 12x12 is a 4x reduction per dim.
    Implement the pooling in stdlib Python and verify that the mean over each
    2x2 block matches the bilinear output.

Reading of the exercise: the equality is implemented and checked at the 24x24
the exercise names, and then the same check is run at the grid the lesson's own
planner actually produces -- which is 27x27, is odd, and is where the equality
stops holding. The bilinear convention is the half-pixel one (`align_corners=
False`): output cell k samples input coordinate `(k + 0.5) * scale - 0.5`.

**ANSWER: at 24 -> 12 the two agree to 0.0 on all 144 outputs.** At an integer
factor of 2 the bilinear sample point of each output cell falls exactly on the
centre of its 2x2 block, and the four corner weights come out 0.25 each. It is
the same arithmetic written twice.

**FINDING: "4x reduction per dim" is 2x per dim.** 24 -> 12 is a factor of 2 on
each axis and **4x** in token count -- 576 -> 144. The exercise's phrase names
the area reduction and attaches it to the wrong noun, which matters because the
lesson's other pooling factor, 3, is 9x in tokens and not 3x.

**FINDING: the lesson's own default grid is 27, and 2x2 blocks do not tile it.**
`per_tile_tokens(384, 14, 2)` floors 384/14 to 27 and then 27/2 to 13, so the
covered area is 26x26 = 676 of 729 patches: **53 patches, 7.3%, are dropped**
and the reduction is **4.31x**, not 4x. The clean 24x24 case the exercise names
is `(336, 14, 2)`, which the planner lists third and never reaches.

**FINDING: at 27 -> 13 the two methods stop agreeing.** Block-mean-after-crop
and bilinear disagree by up to **0.439** on a unit-scale grid, because bilinear
samples at non-integer positions and interpolates across block boundaries while
the crop simply discards the last row and column. The verification the exercise
asks for passes on a grid the lesson does not use and fails on the one it does.

Structure: `block_mean` is the pooling, `bilinear` is the half-pixel resample,
`compare` runs both over one grid, and `RAMP` is the labelled deterministic
fixture.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "08-llava-onevision-single-multi-video"
NAMED, POOL, SEED = 24, 2, 4321
LESSON_RES, PATCH = 384, 14


def make_fixture(side, seed=SEED):
    """A labelled grid: a smooth diagonal ramp plus unit-scale noise, in [0, 2]."""
    rng = random.Random(seed)
    return [[(row + column) / (2 * side) + rng.random() for column in range(side)]
            for row in range(side)]


def block_mean(grid, factor=POOL):
    """Mean over each factor x factor block, discarding any partial last block."""
    out = len(grid) // factor
    return [[sum(grid[r * factor + dr][c * factor + dc]
                 for dr in range(factor) for dc in range(factor)) / (factor * factor)
             for c in range(out)] for r in range(out)]


def _sample(grid, y, x):
    side = len(grid)
    y0, x0 = math.floor(y), math.floor(x)
    fy, fx = y - y0, x - x0
    corners = [(min(max(y0 + dy, 0), side - 1), min(max(x0 + dx, 0), side - 1))
               for dy in (0, 1) for dx in (0, 1)]
    weights = [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx]
    return sum(weight * grid[r][c] for weight, (r, c) in zip(weights, corners))


def bilinear(grid, out):
    """Half-pixel resample: output cell k samples (k + 0.5) * scale - 0.5."""
    scale = len(grid) / out
    return [[_sample(grid, (r + 0.5) * scale - 0.5, (c + 0.5) * scale - 0.5)
             for c in range(out)] for r in range(out)]


def compare(side, factor=POOL):
    grid = make_fixture(side)
    pooled = block_mean(grid, factor)
    resampled = bilinear(grid, len(pooled))
    worst = max(abs(a - b) for rows in zip(pooled, resampled) for a, b in zip(*rows))
    return {"side": side, "out": len(pooled), "cells": len(pooled) ** 2, "worst": worst}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lesson_side = LESSON_RES // PATCH
    covered = (lesson_side // POOL * POOL) ** 2
    return {
        "named": compare(NAMED), "lesson": compare(lesson_side),
        "lesson_side": lesson_side,
        "lesson_tokens": ref.per_tile_tokens(LESSON_RES, PATCH, POOL),
        "named_tokens": ref.per_tile_tokens(336, PATCH, POOL),
        "patches": lesson_side ** 2, "covered": covered,
        "dropped": lesson_side ** 2 - covered,
        "dropped_pct": round((lesson_side ** 2 - covered) / lesson_side ** 2 * 100, 1),
        "reduction": round(lesson_side ** 2 / (lesson_side // POOL) ** 2, 2),
        "named_reduction": round(NAMED ** 2 / (NAMED // POOL) ** 2, 2),
        "per_dim": NAMED // (NAMED // POOL),
        "pool3_tokens": ref.per_tile_tokens(LESSON_RES, PATCH, 3),
    }


def verify(result):
    named, lesson = result["named"], result["lesson"]
    return [
        practice.Check(
            "ANSWER: at 24 -> 12 the two agree to 0.0 on all 144 outputs",
            all([named["out"] == 12, named["cells"] == 144, named["worst"] == 0.0,
                 result["named_tokens"] == 144]),
            f"a {NAMED}x{NAMED} grid pools to {named['out']}x{named['out']} = "
            f"{named['cells']} cells and the block mean matches the half-pixel bilinear "
            f"resample to {named['worst']}. At an integer factor of {POOL} each output's "
            "sample point lands on its block's centre and the four corner weights are 0.25 "
            "each -- the same arithmetic written twice",
        ),
        practice.Check(
            "FINDING: '4x reduction per dim' is 2x per dim",
            all([result["per_dim"] == 2, result["named_reduction"] == 4.0,
                 result["pool3_tokens"] == 81]),
            f"{NAMED} -> {named['out']} is a factor of {result['per_dim']} on each axis and "
            f"{result['named_reduction']}x in token count. The phrase names the area "
            f"reduction and attaches it to the wrong noun, which matters because the "
            f"lesson's other pooling factor, 3, gives {result['pool3_tokens']} tokens from "
            f"{result['patches']} patches -- 9x, not 3x",
        ),
        practice.Check(
            "FINDING: the lesson's own default grid is 27, and 2x2 blocks do not tile it",
            all([result["lesson_side"] == 27, result["patches"] == 729,
                 result["lesson_tokens"] == 169, result["dropped"] == 53,
                 result["dropped_pct"] == 7.3, result["reduction"] == 4.31]),
            f"per_tile_tokens({LESSON_RES}, {PATCH}, {POOL}) floors {LESSON_RES}/{PATCH} to "
            f"{result['lesson_side']} and then {result['lesson_side']}/{POOL} to "
            f"{lesson['out']}, so {result['covered']} of {result['patches']} patches are "
            f"covered and {result['dropped']} -- {result['dropped_pct']}% -- are dropped. "
            f"The reduction is {result['reduction']}x, not 4x, and the clean case the "
            "exercise names is the (336, 14, 2) entry the planner lists third",
        ),
        practice.Check(
            "FINDING: at 27 -> 13 the two methods stop agreeing",
            all([lesson["out"] == 13, lesson["worst"] > 0.2, named["worst"] < lesson["worst"]]),
            f"block-mean-after-crop and bilinear disagree by up to {lesson['worst']:.3f} on a "
            f"unit-scale {result['lesson_side']}x{result['lesson_side']} grid, against "
            f"{named['worst']} at {NAMED}. Bilinear samples at non-integer positions and "
            "interpolates across block boundaries; the crop discards the last row and "
            "column. The verification the exercise asks for passes on a grid the lesson does "
            "not use and fails on the one it does",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
