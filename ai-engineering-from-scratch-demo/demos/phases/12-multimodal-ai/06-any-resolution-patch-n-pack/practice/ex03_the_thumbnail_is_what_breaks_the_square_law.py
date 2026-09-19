"""Exercise 3 — the thumbnail is what breaks the square law.

    For a 1792x896 image at patch 14, compare: (a) square-resize to 336 then
    encode, (b) AnyRes 2x1 + thumbnail, (c) M-RoPE at native. Which uses fewest
    tokens? Which preserves most detail?

Reading of the exercise: the three token counts come from the lesson's own
`square_cost`, `anyres_cost` and `Image.seq`, and "preserves most detail" is
answered with each strategy's effective linear resolution -- pixels of the
original per pixel of what the encoder sees -- because token count alone ranks
the three in the opposite order to quality and says nothing about which of them
distorts.

**ANSWER: square-resize is fewest at 576; native is most detail at 8,192.**
AnyRes sits between at **1,728** -- and the lesson's own scorer independently
picks the **2 x 1** grid the exercise names. The span is **14.2x**.

**FINDING: excluding the thumbnail, tokens are exactly the square of linear
resolution.** AnyRes's two tiles are 1,152 tokens against native's 8,192 --
**7.111x** -- and each tile downscales a 896x896 region to 336, a linear ratio
of 2.6667 whose square is **7.111**. The relationship is exact, and the
thumbnail is the only thing that breaks it: with it included the ratio is
4.74 and means nothing.

**FINDING: square-resize is the only one of the three that distorts.** At
1792x896 the two axes scale by **0.1875** and **0.375**, a 2x anisotropy;
AnyRes's tiles are **0.375** on both axes and native is 1.0 on both. So the
three differ on two axes, not one, and the cheapest is cheap partly by changing
the shape of the content.

**FINDING: native is 8,192 tokens for one image.** That is exactly a
8,192-token context, **4x** a 2,048 window, and **14.2x** the 576 that base
LLaVA spends. "Fewest tokens" and "most detail" are the two ends of the same
14.2x, which is the whole of this lesson.

Structure: `strategies` prices all three through the lesson's own functions,
`linear` is the effective per-axis resolution of each, and `tiles_only` removes
the thumbnail to expose the square law.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "06-any-resolution-patch-n-pack"
HEIGHT, WIDTH, PATCH, SQUARE = 1792, 896, 14, 336
BASE_LLAVA, SMALL_WINDOW = 576, 2048


def strategies(ref, image):
    """(a) square-resize, (b) AnyRes with thumbnail, (c) native -- all from the lesson."""
    anyres = ref.anyres_cost(image, SQUARE, SQUARE)
    return ({"square": ref.square_cost(image, SQUARE, PATCH),
             "anyres": anyres["total"], "native": image.seq(PATCH)}, anyres)


def linear(image, anyres_grid, side=SQUARE):
    """Effective linear resolution per axis: encoder pixels per original pixel."""
    rows, columns = anyres_grid
    return {"square": (round(side / image.h, 4), round(side / image.w, 4)),
            "anyres": (round(side / (image.h / rows), 4),
                       round(side / (image.w / columns), 4)),
            "native": (1.0, 1.0)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    image = ref.Image("panorama", HEIGHT, WIDTH)
    counts, anyres = strategies(ref, image)
    resolutions = linear(image, anyres["grid"])
    tiles_only = anyres["tile_tokens"]
    tile_side = image.h / anyres["grid"][0]
    return {
        "counts": counts, "grid": anyres["grid"],
        "fewest": min(counts, key=counts.get), "most": max(counts, key=counts.get),
        "span": round(counts["native"] / counts["square"], 1),
        "tiles_only": tiles_only, "thumbnail": anyres["thumb_tokens"],
        "tile_ratio": round(counts["native"] / tiles_only, 3),
        "linear_ratio": round(tile_side / SQUARE, 4),
        "square_law": round((tile_side / SQUARE) ** 2, 3),
        "with_thumbnail": round(counts["native"] / counts["anyres"], 2),
        "resolutions": resolutions,
        "anisotropy": {name: round(max(pair) / min(pair), 2)
                       for name, pair in resolutions.items()},
        "vs_llava": round(counts["native"] / BASE_LLAVA, 1),
        "vs_window": round(counts["native"] / SMALL_WINDOW, 1),
    }


def verify(result):
    counts, resolutions = result["counts"], result["resolutions"]
    return [
        practice.Check(
            "ANSWER: square-resize is fewest at 576; native is most detail at 8,192",
            all([counts == {"square": 576, "anyres": 1728, "native": 8192},
                 result["grid"] == (2, 1), result["fewest"] == "square",
                 result["most"] == "native", result["span"] == 14.2]),
            f"the three cost {counts} and the lesson's own scorer picks the "
            f"{result['grid'][0]} x {result['grid'][1]} grid the exercise names. Fewest is "
            f"{result['fewest']}, most detail is {result['most']}, and the span is "
            f"{result['span']}x",
        ),
        practice.Check(
            "FINDING: excluding the thumbnail, tokens are the square of linear resolution",
            all([result["tiles_only"] == 1152, result["thumbnail"] == 576,
                 result["tile_ratio"] == 7.111, result["square_law"] == 7.111,
                 result["linear_ratio"] == 2.6667, result["with_thumbnail"] == 4.74]),
            f"AnyRes's tiles are {result['tiles_only']:,} tokens against native's "
            f"{counts['native']:,} -- {result['tile_ratio']}x -- and each tile downscales a "
            f"896x896 region to {SQUARE}, a linear ratio of {result['linear_ratio']} whose "
            f"square is {result['square_law']}. Exact. Add the {result['thumbnail']}-token "
            f"thumbnail and the ratio becomes {result['with_thumbnail']}, which means nothing",
        ),
        practice.Check(
            "FINDING: square-resize is the only one of the three that distorts",
            all([resolutions["square"] == (0.1875, 0.375),
                 resolutions["anyres"] == (0.375, 0.375),
                 result["anisotropy"] == {"square": 2.0, "anyres": 1.0, "native": 1.0}]),
            f"per-axis effective resolution is {resolutions} -- anisotropy "
            f"{result['anisotropy']}. AnyRes and native are square; square-resize is 2x "
            "distorted, so the three differ on two axes and the cheapest is cheap partly by "
            "changing the shape of the content",
        ),
        practice.Check(
            "FINDING: native is 8,192 tokens for one image",
            all([result["vs_llava"] == 14.2, result["vs_window"] == 4.0,
                 result["span"] == result["vs_llava"]]),
            f"{counts['native']:,} tokens is exactly an 8,192-token context, "
            f"{result['vs_window']}x a {SMALL_WINDOW:,} window and {result['vs_llava']}x the "
            f"{BASE_LLAVA} base LLaVA spends. 'Fewest tokens' and 'most detail' are the two "
            f"ends of the same {result['span']}x",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
