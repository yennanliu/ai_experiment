"""Exercise 1 — both dimensions the exercise names raise.

    Compute the patch-token sequence length for Qwen2.5-VL at native 1280x720
    input with patch size 14. How does that compare to a CLS-only
    representation?

Reading of the exercise: the arithmetic is done first and the lesson's own
`grid_shape` is then asked for the same number, because the interesting part of
this exercise is that it cannot answer. "Compare to a CLS-only representation"
is read as both the token count and the number of floats, since one token of
1280 dimensions is not one number.

**ANSWER: 91 x 51 = 4,641 patch tokens against 1 for CLS-only.** In floats that
is 5,940,480 against 1,280 -- a ratio of 4,641 either way, because both sides
carry the same hidden size. The CLS vector is 0.02% of what the patch grid
holds.

**FINDING: the lesson cannot express the input the exercise names.**
`grid_shape` raises `ValueError` on 1280 and again on 720, because neither is
divisible by 14, and `ViTConfig` has a single square `image_size` field -- so
1280x720 is not a configuration this code can hold, let alone measure. The
number above comes from floor division done by hand.

**FINDING: floor and ceil disagree by 143 tokens.** 1280/14 is 91.43 and 720/14
is 51.43, so the grid is 91x51 or 92x52 depending on whether the image is
cropped or padded: 4,641 against 4,784, a 3.1% difference in sequence length
and therefore in attention cost. The exercise names neither convention.

**FINDING: Qwen2.5-VL's own answer is 1,125, not 4,641.** The model merges 2x2
neighbourhoods after the encoder, so the count the language model sees is a
quarter of the patch count -- and the merge requires both sides to be multiples
of 28, not 14, so 1280x720 is first resized to 1260x700. That is 4,500 patches
becoming 1,125 tokens, and 20 pixels of width and height discarded before any
of it starts.

**FINDING: the lesson's own Qwen entry is a square 896 crop.**
`ZOO[-1]` is `Qwen2.5-VL ViT @ 896x896`, whose sequence length is 4,097 -- 64x64
patches plus a CLS token. So the entry named after the native-resolution model
is the one configuration that model does not use, and it happens to land within
12% of the native-resolution answer by coincidence.

Structure: `NATIVE` is the resolution the exercise names, `tokens` is the
patch-grid count under a stated rounding rule, `merged` applies the 2x2 spatial
merge with its multiple-of-28 constraint, and `refused` collects what the
lesson's own `grid_shape` does with each dimension.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "01-vision-transformer-patch-tokens"
NATIVE = (1280, 720)
PATCH, MERGE = 14, 2
HIDDEN = 1280


def tokens(size, patch=PATCH, rounding=math.floor):
    """The patch-grid token count for a (width, height) at one rounding rule."""
    grid = [int(rounding(side / patch)) for side in size]
    return grid, grid[0] * grid[1]


def merged(size, patch=PATCH, merge=MERGE):
    """Qwen's 2x2 spatial merge: both sides must be multiples of patch * merge."""
    step = patch * merge
    usable = tuple(side - side % step for side in size)
    grid, patches = tokens(usable, patch)
    return {"resized": usable, "lost": tuple(a - b for a, b in zip(size, usable)),
            "patches": patches, "tokens": patches // (merge * merge),
            "grid": [side // step for side in usable]}


def refused(ref, size, patch=PATCH):
    """What the lesson's own grid_shape does with each dimension."""
    errors = {}
    for side in size:
        try:
            ref.grid_shape(side, patch)
        except ValueError as problem:
            errors[side] = str(problem)
    return errors


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    floor_grid, floor_count = tokens(NATIVE)
    ceil_grid, ceil_count = tokens(NATIVE, rounding=math.ceil)
    shipped = ref.ZOO[-1]
    return {
        "native": NATIVE, "patch": PATCH,
        "grid": floor_grid, "count": floor_count,
        "ceil_grid": ceil_grid, "ceil_count": ceil_count,
        "rounding_gap_pct": round((ceil_count / floor_count - 1) * 100, 1),
        "floats": floor_count * HIDDEN, "cls_floats": HIDDEN,
        "refused": refused(ref, NATIVE), "divisible": [side % PATCH for side in NATIVE],
        **{f"merge_{k}": v for k, v in merged(NATIVE).items()},
        "shipped_name": shipped.name, "shipped_seq": ref.seq_length(shipped),
        "shipped_grid": list(ref.grid_shape(shipped.image_size, shipped.patch_size)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 91 x 51 = 4,641 patch tokens against 1 for CLS-only",
            all([result["grid"] == [91, 51], result["count"] == 4641,
                 result["floats"] == 5940480]),
            f"{result['native'][0]}x{result['native'][1]} at patch {result['patch']} is a "
            f"{result['grid'][0]}x{result['grid'][1]} grid -- {result['count']:,} tokens "
            f"against 1 -- and {result['floats']:,} floats against {result['cls_floats']:,}. "
            f"The ratio is {result['count']:,} either way, because both carry the same "
            f"hidden size, so the CLS vector is "
            f"{result['cls_floats'] / result['floats'] * 100:.2f}% of the grid",
        ),
        practice.Check(
            "FINDING: the lesson cannot express the input the exercise names",
            all([len(result["refused"]) == 2,
                 all(r % result["patch"] for r in result["divisible"] or [1]),
                 all("divisible" in message for message in result["refused"].values())]),
            f"`grid_shape` raises ValueError on both {sorted(result['refused'])}, because "
            f"{result['native'][0]} % {result['patch']} = {result['divisible'][0]} and "
            f"{result['native'][1]} % {result['patch']} = {result['divisible'][1]}; and "
            "ViTConfig has a single square image_size field, so a 16:9 input is not a "
            "configuration this code can hold. The answer above is floor division by hand",
        ),
        practice.Check(
            "FINDING: floor and ceil disagree by 143 tokens",
            all([result["ceil_count"] == 4784,
                 result["ceil_count"] - result["count"] == 143,
                 result["rounding_gap_pct"] == 3.1]),
            f"1280/14 is 91.43 and 720/14 is 51.43, so the grid is "
            f"{result['grid']} cropped or {result['ceil_grid']} padded: "
            f"{result['count']:,} against {result['ceil_count']:,}, "
            f"{result['rounding_gap_pct']}% of sequence length and therefore of attention "
            "cost. The exercise names neither convention",
        ),
        practice.Check(
            "FINDING: Qwen2.5-VL's own answer is 1,125, not 4,641",
            all([result["merge_tokens"] == 1125, result["merge_patches"] == 4500,
                 result["merge_resized"] == (1260, 700), result["merge_lost"] == (20, 20)]),
            f"the 2x2 spatial merge quarters the count the language model sees, and it needs "
            f"both sides to be multiples of {PATCH * MERGE}: {result['native']} is resized "
            f"to {result['merge_resized']}, losing {result['merge_lost']} pixels, giving "
            f"{result['merge_patches']:,} patches and {result['merge_tokens']:,} tokens -- "
            f"{result['count'] / result['merge_tokens']:.1f}x fewer than the raw grid",
        ),
        practice.Check(
            "FINDING: the lesson's own Qwen entry is a square 896 crop",
            all([result["shipped_seq"] == 4097, result["shipped_grid"] == [64, 64],
                 "896x896" in result["shipped_name"]]),
            f"ZOO[-1] is {result['shipped_name']!r}: a "
            f"{result['shipped_grid'][0]}x{result['shipped_grid'][1]} grid plus a CLS token, "
            f"{result['shipped_seq']:,} tokens. The entry named after the native-resolution "
            "model is the one configuration that model does not use, and it lands within "
            f"{abs(result['shipped_seq'] / result['count'] - 1) * 100:.0f}% of the native "
            "answer by coincidence",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
