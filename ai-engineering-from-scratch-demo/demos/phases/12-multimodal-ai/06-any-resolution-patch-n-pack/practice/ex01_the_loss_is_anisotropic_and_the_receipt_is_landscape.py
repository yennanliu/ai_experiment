"""Exercise 1 — the loss is anisotropic, and the receipt is landscape.

    A receipt is 600x1500 (1:2.5). At patch size 14, how many
    native-resolution tokens? How many after square-resize to 336? Which loses
    more OCR accuracy in practice?

Reading of the exercise: the counts come from the lesson's own `Image.seq` and
`square_cost`, and "which loses more OCR accuracy" is answered with the scale
factors rather than with an opinion -- because square-resize does two separate
things to a 1:2.5 page and only one of them is visible in the token count.

**ANSWER: 4,494 native against 576 square, a 7.80x reduction.** 600 and 1500
are not multiples of 14, so the lesson crops to 588 x 1498 first and the grid is
42 x 107.

**FINDING: the token ratio and the area ratio are the same number, and neither
is the reason OCR breaks.** One square token covers **7.80x** the original area
of one native token -- but the two axes are scaled by **0.5714** and **0.2243**,
a **2.55x** anisotropy. A glyph that was taller than wide comes out wider than
tall. Square-resize loses more OCR accuracy, and the mechanism is the
distortion, not the budget.

**FINDING: the lesson's own receipt is landscape.** `Image(name, h, w)` takes
height first, so `Image("receipt 600x1500", 600, 1500)` is 600 tall and 1500
wide. The token count does not notice -- a grid product is symmetric -- but
`anyres_cost` does: it tiles the same receipt **1 x 2** as given and **2 x 1**
when the sides are swapped. The strategy that depends on orientation is the one
reading it backwards.

**FINDING: the crop is not free either.** 600 -> 588 discards **2.0%** of the
short side and 1500 -> 1498 **0.13%** of the long one, before any resize
strategy is chosen. On a receipt the short side is the one carrying character
width.

Structure: `receipt` builds the lesson's own image, `scales` is the per-axis
resize factor, and `tiling` runs the lesson's `anyres_cost` on both
orientations.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "06-any-resolution-patch-n-pack"
RAW_H, RAW_W, PATCH, SQUARE = 600, 1500, 14, 336


def cropped(ref, height=RAW_H, width=RAW_W, patch=PATCH):
    """The lesson's own crop-to-multiple, as main() applies it to the workload."""
    return ref.Image("receipt", height - height % patch, width - width % patch)


def scales(image, side=SQUARE):
    return round(side / image.h, 4), round(side / image.w, 4)


def tiling(ref, image):
    return ref.anyres_cost(image)["grid"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    image = cropped(ref)
    native, square = image.seq(PATCH), ref.square_cost(image, SQUARE, PATCH)
    vertical, horizontal = scales(image)
    return {
        "raw": (RAW_H, RAW_W), "cropped": (image.h, image.w),
        "grid": list(image.grid(PATCH)), "native": native, "square": square,
        "ratio": round(native / square, 2),
        "area_ratio": round(image.h * image.w / (SQUARE * SQUARE), 2),
        "vertical": vertical, "horizontal": horizontal,
        "anisotropy": round(image.w / image.h, 3),
        "as_given": tiling(ref, image),
        "swapped": tiling(ref, ref.Image("receipt", image.w, image.h)),
        "crop_loss": [round((RAW_H - image.h) / RAW_H * 100, 2),
                      round((RAW_W - image.w) / RAW_W * 100, 2)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 4,494 native against 576 square, a 7.80x reduction",
            all([result["native"] == 4494, result["square"] == 576,
                 result["ratio"] == 7.8, result["grid"] == [42, 107],
                 result["cropped"] == (588, 1498)]),
            f"{RAW_H}x{RAW_W} crops to {result['cropped'][0]}x{result['cropped'][1]} at patch "
            f"{PATCH} -- a {result['grid'][0]} x {result['grid'][1]} grid, "
            f"{result['native']:,} tokens -- against {result['square']} after square-resize "
            f"to {SQUARE}: {result['ratio']}x",
        ),
        practice.Check(
            "FINDING: the token ratio and the area ratio are the same number",
            all([result["area_ratio"] == result["ratio"],
                 result["vertical"] == 0.5714, result["horizontal"] == 0.2243,
                 result["anisotropy"] == 2.548]),
            f"one square token covers {result['area_ratio']}x the original area of one native "
            f"token -- the same {result['ratio']}x -- but the two axes are scaled by "
            f"{result['vertical']} and {result['horizontal']}, an anisotropy of "
            f"{result['anisotropy']}x. A glyph taller than wide comes out wider than tall, "
            "and that is the OCR loss, not the budget",
        ),
        practice.Check(
            "FINDING: the lesson's own receipt is landscape",
            all([result["as_given"] == (1, 2), result["swapped"] == (2, 1),
                 result["cropped"][0] < result["cropped"][1]]),
            f"Image(name, h, w) takes height first, so the receipt is "
            f"{result['cropped'][0]} tall and {result['cropped'][1]} wide. The token count "
            f"cannot notice -- a grid product is symmetric -- but anyres_cost tiles it "
            f"{result['as_given']} as given and {result['swapped']} with the sides swapped. "
            "The one strategy that depends on orientation is reading it backwards",
        ),
        practice.Check(
            "FINDING: the crop is not free either",
            result["crop_loss"] == [2.0, 0.13],
            f"{RAW_H} -> {result['cropped'][0]} discards {result['crop_loss'][0]}% of the "
            f"short side and {RAW_W} -> {result['cropped'][1]} {result['crop_loss'][1]}% of "
            "the long one, before any resize strategy is chosen. On a receipt the short side "
            "is the one carrying character width",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
