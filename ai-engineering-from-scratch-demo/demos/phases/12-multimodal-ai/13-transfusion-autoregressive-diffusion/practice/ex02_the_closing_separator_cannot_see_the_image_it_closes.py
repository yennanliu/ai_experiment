"""Exercise 2 — the closing separator cannot see the image it closes.

    Implement the block-triangular mask for a sequence: `[T, T, <image>, P, P,
    P, P, </image>, T]`. Mark each entry 0 or 1.

Reading of the exercise: the mask is produced by the lesson's own `build_mask`
on exactly the nine-token sequence the exercise writes down, and then every row
is read rather than the pattern being eyeballed -- because the two separator
rows turn out not to follow the rule the mask is named for.

**ANSWER: 40 of 81 entries are 1.** Row sums are
**1, 2, 2, 6, 6, 6, 6, 2, 9**: causal over text, fully bidirectional within the
four patches, and everything visible to the trailing text token.

```
     0 1 2 3 4 5 6 7 8
 0 | 1 . . . . . . . .    T
 1 | 1 1 . . . . . . .    T
 2 | 1 1 . . . . . . .    <image>
 3 | 1 1 . 1 1 1 1 . .    P
 4 | 1 1 . 1 1 1 1 . .    P
 5 | 1 1 . 1 1 1 1 . .    P
 6 | 1 1 . 1 1 1 1 . .    P
 7 | 1 1 . . . . . . .    </image>
 8 | 1 1 1 1 1 1 1 1 1    T
```

**FINDING: `</image>` attends to the two text tokens and to none of the four
patches it delimits.** `in_text` excludes both separators by name and `same_img`
covers only the open interval between them, so row 7 falls through every branch
except "text at or before me". The token whose job is to close an image block
cannot read the block.

**FINDING: the patches cannot see their own opening separator either.** Column 2
is zero for rows 3-6, for the mirror-image reason -- `<image>` is neither text
nor inside the image range. The delimiters are structurally invisible to the
content they delimit, in both directions.

**FINDING: the trailing text token is the only row that is complete.** Row 8 is
all nine entries, because a text query takes preceding text by one branch and
everything non-text by another, with no exclusion for the separators. The
sequence has one position that can see the whole of it.

Structure: `SEQUENCE` is the exercise's own nine tokens, `rows` extracts the
mask's row sums, and `column` reads one column across the image block.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "13-transfusion-autoregressive-diffusion"
TEXT_BEFORE, PATCHES = 2, 4
OPEN_AT = TEXT_BEFORE
PATCH_RANGE = range(OPEN_AT + 1, OPEN_AT + 1 + PATCHES)
CLOSE_AT = OPEN_AT + 1 + PATCHES


def sequence(ref):
    return ([10, 11, ref.SEP_OPEN] + [f"p{i}" for i in range(PATCHES)]
            + [ref.SEP_CLOSE, 12])


def column(mask, index):
    return [row[index] for row in mask]


def patch_block(mask):
    return [[mask[i][j] for j in PATCH_RANGE] for i in PATCH_RANGE]


def complete_rows(mask):
    return [i for i, row in enumerate(mask) if sum(row) == len(mask)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tokens = sequence(ref)
    mask = ref.build_mask(tokens)
    return {
        "size": len(mask), "cells": len(mask) ** 2,
        "ones": sum(sum(row) for row in mask),
        "row_sums": [sum(row) for row in mask],
        "close_row": mask[CLOSE_AT],
        "close_sees_patches": sum(mask[CLOSE_AT][j] for j in PATCH_RANGE),
        "open_column": [column(mask, OPEN_AT)[j] for j in PATCH_RANGE],
        "patch_block": patch_block(mask),
        "last_row": mask[-1],
        "complete_rows": complete_rows(mask),
        "density_pct": round(sum(sum(row) for row in mask) / len(mask) ** 2 * 100, 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 40 of 81 entries are 1, with row sums 1, 2, 2, 6, 6, 6, 6, 2, 9",
            all([result["size"] == 9, result["cells"] == 81, result["ones"] == 40,
                 result["row_sums"] == [1, 2, 2, 6, 6, 6, 6, 2, 9],
                 result["patch_block"] == [[1] * PATCHES] * PATCHES,
                 result["density_pct"] == 49.4]),
            f"the mask is {result['size']}x{result['size']} with {result['ones']} ones, "
            f"{result['density_pct']}% dense, and row sums {result['row_sums']}. The "
            f"{PATCHES}x{PATCHES} patch block is fully bidirectional and the text rows are "
            "causal, which is the rule the mask is named for",
        ),
        practice.Check(
            "FINDING: </image> attends to the two text tokens and to none of its patches",
            all([result["close_sees_patches"] == 0,
                 sum(result["close_row"]) == TEXT_BEFORE,
                 result["close_row"][:TEXT_BEFORE] == [1] * TEXT_BEFORE]),
            f"row {CLOSE_AT} has {sum(result['close_row'])} ones, all in the leading text, "
            f"and {result['close_sees_patches']} in the patch columns. in_text excludes both "
            "separators by name and same_img covers only the open interval between them, so "
            "the token whose job is to close the block cannot read it",
        ),
        practice.Check(
            "FINDING: the patches cannot see their own opening separator either",
            all([result["open_column"] == [0] * PATCHES,
                 result["row_sums"][OPEN_AT] == TEXT_BEFORE]),
            f"column {OPEN_AT} is {result['open_column']} across the patch rows, for the "
            f"mirror-image reason -- <image> is neither text nor inside the image range. The "
            "delimiters are structurally invisible to the content they delimit, in both "
            "directions",
        ),
        practice.Check(
            "FINDING: the trailing text token is the only row that is complete",
            all([result["complete_rows"] == [8], result["last_row"] == [1] * 9]),
            f"row 8 is all {len(result['last_row'])} entries -- a text query takes preceding "
            f"text by one branch and everything non-text by another, with no exclusion for "
            f"the separators -- and it is the only such row: {result['complete_rows']}. The "
            "sequence has exactly one position that can see the whole of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
