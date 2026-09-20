"""Exercise 3 — the only non-linearity in AnyRes is the thumbnail.

    Read the AnyRes section of the LLaVA-NeXT blog. Compute the visual token
    count for a 1344x672 image at AnyRes. Compare to base 576 tokens at 336x336.

Reading of the exercise: the count is taken from the lesson's own `demo_anyres`
table, which already carries the 1344x672 row, and then the whole table is
checked against the pixel counts it is supposed to track -- because "compare to
base 576" invites one ratio and the table contains five, which is enough to see
the shape of the scheme.

**ANSWER: 5,184 tokens -- 8 tiles plus 1 thumbnail at 576 each -- against 576
at base.** A **9x** increase for an 8x increase in pixels.

**FINDING: the thumbnail is the only non-linearity in the scheme.** Across all
five configurations the token count is exactly `(pixel ratio + 1) x 576`: 1, 3,
5, 9, 17 for pixel ratios 1, 2, 4, 8, 16. AnyRes is linear in pixels plus one
constant tile, so its share falls from **33.3%** of the bill at two tiles to
**5.9%** at sixteen.

**FINDING: the lesson's own takeaway is 3.4x below its own table.** The
takeaway prints "AnyRes: up to 2880 tokens"; 2,880 is the 672x672 row, three
rows above the last, and the table's own maximum is **9,792**.

**FINDING: `visualize_context` reports a 253% overrun as "remain 0".** At 5,184
visual tokens the lesson's own budget table prints `image 253.1%` alongside
`remain 0 tokens` for the 2,048 window, because the remainder is clamped with
`max(remain, 0)`. A configuration that cannot fit prints as one that exactly
fills.

Structure: `CONFIGS` is the lesson's own table with the pixel counts added,
`tokens` prices one row, and `budget_line` captures what `visualize_context`
actually prints for the 2,048 window.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "05-llava-visual-instruction-tuning"
TILE, BASE_SIDE = 576, 336
TAKEAWAY_MAX, SMALL_WINDOW = 2880, 2048
TEXT_TOKENS = 30
CONFIGS = (("336x336", 1, 0, 336, 336), ("672x336", 2, 1, 672, 336),
           ("672x672", 4, 1, 672, 672), ("1344x672", 8, 1, 1344, 672),
           ("1344x1344", 16, 1, 1344, 1344))
TARGET = "1344x672"


def tokens(tiles, thumbnail, tile=TILE):
    return (tiles + thumbnail) * tile


def pixel_ratio(width, height, side=BASE_SIDE):
    return width * height / (side * side)


def budget_line(ref, visual, window=SMALL_WINDOW, text=TEXT_TOKENS):
    """The row visualize_context prints for one window, captured from the lesson."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.visualize_context(visual, text)
    return next(line for line in buffer.getvalue().splitlines()
                if f"window {window:>6d}" in line)


def build_table():
    """The lesson's own AnyRes rows, priced and set against their pixel ratios."""
    return {name: {"tokens": tokens(tiles, thumb),
                   "ratio": tokens(tiles, thumb) / TILE,
                   "pixels": pixel_ratio(width, height),
                   "thumb_share": round(thumb * TILE / tokens(tiles, thumb) * 100, 1)}
            for name, tiles, thumb, width, height in CONFIGS}


def is_linear(table):
    """Every row with a thumbnail costs exactly (pixel ratio + 1) tiles."""
    return all(row["ratio"] == row["pixels"] + (0 if name == "336x336" else 1)
               for name, row in table.items())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = build_table()
    target = table[TARGET]
    biggest = max(row["tokens"] for row in table.values())
    return {
        "target": target["tokens"], "base": TILE,
        "ratio": target["ratio"], "pixels": target["pixels"],
        "linear": is_linear(table),
        "ratios": [table[name]["ratio"] for name, *_ in CONFIGS],
        "pixel_ratios": [table[name]["pixels"] for name, *_ in CONFIGS],
        "thumb_shares": [table[name]["thumb_share"] for name, *_ in CONFIGS],
        "table_max": biggest, "takeaway": TAKEAWAY_MAX,
        "takeaway_gap": round(biggest / TAKEAWAY_MAX, 1),
        "takeaway_row": [name for name, row in table.items()
                         if row["tokens"] == TAKEAWAY_MAX],
        "budget_line": budget_line(ref, target["tokens"]).strip(),
        "overrun_pct": round(target["tokens"] / SMALL_WINDOW * 100, 1),
    }


def verify(result):
    # Indexed eagerly when these Checks are built, so guard rather than raise.
    shares, takeaway_row = result["thumb_shares"], result["takeaway_row"]
    first_share = shares[1] if len(shares) > 1 else None
    last_share = shares[-1] if shares else None
    takeaway_label = takeaway_row[0] if takeaway_row else None
    return [
        practice.Check(
            "ANSWER: 5,184 tokens -- 8 tiles plus 1 thumbnail -- against 576 at base",
            all([result["target"] == 5184, result["base"] == 576,
                 result["ratio"] == 9.0, result["pixels"] == 8.0]),
            f"{TARGET} is 8 tiles of {TILE} plus one {TILE}-token thumbnail = "
            f"{result['target']:,} tokens against the base {result['base']} -- "
            f"{result['ratio']:.0f}x the tokens for {result['pixels']:.0f}x the pixels",
        ),
        practice.Check(
            "FINDING: the thumbnail is the only non-linearity in the scheme",
            all([result["linear"], result["ratios"] == [1.0, 3.0, 5.0, 9.0, 17.0],
                 result["pixel_ratios"] == [1.0, 2.0, 4.0, 8.0, 16.0],
                 result["thumb_shares"] == [0.0, 33.3, 20.0, 11.1, 5.9]]),
            f"token ratios {result['ratios']} against pixel ratios "
            f"{result['pixel_ratios']} -- exactly (pixels + 1) x {TILE} in every row with a "
            f"thumbnail. Its share of the bill falls {first_share}% -> "
            f"{last_share}% as tiles grow, so AnyRes is linear in pixels "
            "plus one constant tile",
        ),
        practice.Check(
            "FINDING: the lesson's own takeaway is 3.4x below its own table",
            all([result["table_max"] == 9792, result["takeaway_gap"] == 3.4,
                 result["takeaway_row"] == ["672x672"]]),
            f"the takeaway prints 'up to {result['takeaway']:,} tokens'; "
            f"{result['takeaway']:,} is the {takeaway_label} row, three above the "
            f"last, and the table's own maximum is {result['table_max']:,} -- "
            f"{result['takeaway_gap']}x higher",
        ),
        practice.Check(
            "FINDING: visualize_context reports a 253% overrun as 'remain 0'",
            all([result["overrun_pct"] == 253.1, "253.1%" in result["budget_line"],
                 result["budget_line"].endswith("remain      0 tokens")]),
            f"at {result['target']:,} visual tokens the lesson prints "
            f"{result['budget_line']!r} for the {SMALL_WINDOW:,} window -- the remainder is "
            "clamped with max(remain, 0), so a configuration that overruns by "
            f"{result['overrun_pct'] - 100:.1f}% prints as one that exactly fills",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
