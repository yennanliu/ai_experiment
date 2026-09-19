"""Exercise 5 — packing without the mask costs more than padding.

    Modify `code/main.py` to support patch-n'-pack: given a list of images of
    different resolutions, produce a single packed sequence and the
    block-diagonal attention mask. Verify against Lesson 12.06 when you reach
    it.

Reading of the exercise: the packed sequence and the mask are built here rather
than patched into `code/main.py`, because the lesson's module is imported and
not forked (D5); what is taken from it is the geometry -- `grid_shape` for each
image and `pos_embed_params` for the table the packed sequence has to index
into. The mask is verified by its row sums rather than materialised at full
size, and a toy batch is materialised in full to show the two agree.

**ANSWER: four images pack to 2,537 tokens against 5,040 padded.** The batch is
256 + 768 + 1,260 + 253 patch tokens; padding every row to the largest wastes
**49.7%** of the sequence on tokens that carry nothing.

**FINDING: the mask is not an optimisation of packing, it is the whole of it.**
Attention over the packed sequence touches 2,537^2 = 6,436,369 query-key pairs,
which is **1.4% more** than the padded batch's 6,350,400. Masked to the block
diagonal it is 2,306,969 -- **35.8%** of the packed square and **63.7% below**
the padded batch. Packing without the mask is a regression, not a speed-up.

**FINDING: the packed sequence has no positions to be embedded with.** The
lesson's `pos_embed_params` is a table of exactly `seq_length(cfg)` rows, and
2,537 exceeds four of the five `ZOO` entries -- 197, 261, 577, 733. Only the
Qwen entry's 4,097 rows are enough, and even then rows are addressed by a
single index while the packed batch needs four independent 2D origins. This is
the reason patch-n'-pack ships with 2D-RoPE and not a learned table.

**FINDING: the block structure is exactly the segment identity.** For all
2,537 rows the mask's row sum equals its own image's token count, and on a
toy batch of 11 tokens the materialised 11x11 mask is identical to
`seg[i] == seg[j]` in all 121 cells.

Structure: `BATCH` is the labelled four-image batch, `pack` returns the token
counts and segment ids, `pairs` counts attended query-key pairs under each
regime, and `dense_mask` materialises the toy case the row-sum check is
cross-examined against.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "01-vision-transformer-patch-tokens"
PATCH = 14
BATCH = (("poster 224x224", 224, 224), ("slide 336x448", 336, 448),
         ("photo 630x392", 630, 392), ("icon 154x322", 154, 322))
TOY = ((2, 3), (1, 2), (3, 1))
TWIN = ((2, 2), (2, 2), (1, 3))   # two images of equal token count, on purpose


def pack(images, patch=PATCH):
    """Token count per image and the segment id of every packed position."""
    counts = [(width // patch) * (height // patch) for _, width, height in images]
    segments = [image for image, count in enumerate(counts) for _ in range(count)]
    return counts, segments


def pairs(counts):
    """Query-key pairs attended under packing+mask, packing alone, and padding."""
    total, widest = sum(counts), max(counts)
    return {"masked": sum(count * count for count in counts),
            "unmasked": total * total, "padded": len(counts) * widest * widest}


def row_sums(segments):
    """The same sums without materialising: a row sums to its own segment's size."""
    occupancy = Counter(segments)
    return [occupancy[segment] for segment in segments]


def twin_evidence(twin=TWIN):
    """Row sums where two images share a token count -- a set would collapse them."""
    mask, segments = dense_mask(twin)
    counts = [rows * cols for rows, cols in twin]
    return {"twin_counts": counts, "twin_tally": sorted(Counter(row_sums(segments)).items()),
            "twin_formula_holds": [sum(row) for row in mask] == row_sums(segments)}


def dense_mask(grids):
    """The toy case materialised in full, as (mask, segments)."""
    counts = [rows * cols for rows, cols in grids]
    segments = [image for image, count in enumerate(counts) for _ in range(count)]
    mask = [[left == right for right in segments] for left in segments]
    return mask, segments


def grids(ref, images, patch=PATCH):
    """Each image's patch grid, taken one side at a time from the lesson's grid_shape."""
    return [[ref.grid_shape(width, patch)[0], ref.grid_shape(height, patch)[0]]
            for _, width, height in images]


def agrees(mask, segments):
    """The materialised toy mask against the segment identity it should encode."""
    span = range(len(mask))
    return all(mask[left][right] == (segments[left] == segments[right])
               for left in span for right in span)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts, segments = pack(BATCH)
    cost = pairs(counts)
    mask, toy_segments = dense_mask(TOY)
    toy_sums = [sum(row) for row in mask]
    tables = sorted(ref.seq_length(cfg) for cfg in ref.ZOO)
    shapes = grids(ref, BATCH)
    packed, widest = sum(counts), max(counts)
    return {
        "labels": [name for name, _, _ in BATCH], "counts": counts,
        "grids": shapes, "from_grids": [rows * cols for rows, cols in shapes],
        "packed": packed, "padded_seq": widest * len(counts),
        "waste_pct": round((1 - packed / (widest * len(counts))) * 100, 1),
        **cost,
        "density_pct": round(cost["masked"] / cost["unmasked"] * 100, 2),
        "mask_vs_padded_pct": round((1 - cost["masked"] / cost["padded"]) * 100, 1),
        "unmasked_vs_padded_pct": round((cost["unmasked"] / cost["padded"] - 1) * 100, 1),
        "row_sum_tally": sorted(Counter(row_sums(segments)).items()),
        "toy_dense_sums": toy_sums, "formula_holds": toy_sums == row_sums(toy_segments),
        **twin_evidence(),
        "tables": tables, "too_small": [seq for seq in tables if seq < packed],
        "qwen_table": ref.pos_embed_params(ref.ZOO[-1]),
        "toy_cells": len(mask) * len(mask),
        "toy_agrees": agrees(mask, toy_segments),
        "toy_true": sum(sum(row) for row in mask),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four images pack to 2,537 tokens against 5,040 padded",
            all([result["counts"] == [256, 768, 1260, 253], result["packed"] == 2537,
                 result["padded_seq"] == 5040, result["waste_pct"] == 49.7,
                 result["from_grids"] == result["counts"],
                 result["grids"] == [[16, 16], [24, 32], [45, 28], [11, 23]]]),
            f"{dict(zip(result['labels'], result['counts']))} packs to "
            f"{result['packed']:,} tokens; padding every row to the largest "
            f"({max(result['counts']):,}) makes {result['padded_seq']:,}, so "
            f"{result['waste_pct']}% of the padded sequence carries nothing",
        ),
        practice.Check(
            "FINDING: the mask is not an optimisation of packing, it is the whole of it",
            all([result["unmasked"] == 6_436_369, result["padded"] == 6_350_400,
                 result["masked"] == 2_306_969, result["density_pct"] == 35.84,
                 result["mask_vs_padded_pct"] == 63.7,
                 result["unmasked_vs_padded_pct"] == 1.4]),
            f"the packed square is {result['unmasked']:,} query-key pairs, "
            f"{result['unmasked_vs_padded_pct']}% *more* than the padded batch's "
            f"{result['padded']:,}; masked to the block diagonal it is "
            f"{result['masked']:,} -- {result['density_pct']}% of the square and "
            f"{result['mask_vs_padded_pct']}% below padding. Unmasked packing is a regression",
        ),
        practice.Check(
            "FINDING: the packed sequence has no positions to be embedded with",
            all([result["too_small"] == [197, 261, 577, 733], len(result["tables"]) == 5,
                 result["tables"][-1] == 4097]),
            f"pos_embed_params is a table of exactly seq_length rows, and {result['packed']:,} "
            f"exceeds {len(result['too_small'])} of the {len(result['tables'])} ZOO entries "
            f"({result['too_small']}). Only Qwen's {result['tables'][-1]:,} rows "
            f"({result['qwen_table']:,} params) are enough -- and they are addressed by one "
            "index, while this batch needs four independent 2D origins",
        ),
        practice.Check(
            "FINDING: the block structure is exactly the segment identity",
            all([result["row_sum_tally"] ==
                 [(253, 253), (256, 256), (768, 768), (1260, 1260)],
                 result["formula_holds"], result["toy_agrees"],
                 result["toy_cells"] == 121, result["toy_true"] == 49,
                 result["twin_formula_holds"],
                 result["twin_tally"] == [(3, 3), (4, 8)]]),
            f"summing the toy's {result['toy_cells']}-cell mask row by row gives "
            f"{result['toy_dense_sums']} -- its segment occupancy exactly, so a row sums to "
            f"its own segment's size. The batch tally is {result['row_sum_tally']}: one "
            f"block per image, each contributing as many rows as it has tokens. Tallying "
            f"rather than setting is what makes it a measurement -- two equally sized "
            f"images ({result['twin_counts']}) still tally to {result['twin_tally']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
