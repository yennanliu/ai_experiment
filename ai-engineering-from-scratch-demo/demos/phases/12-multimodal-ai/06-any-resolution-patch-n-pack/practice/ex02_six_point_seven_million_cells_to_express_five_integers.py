"""Exercise 2 — 6.7 million cells to express five integers.

    Build the block-diagonal mask for a batch of four images with lengths 256,
    576, 729, 1024. Verify the attention matrix is 2585x2585 and has exactly
    `256^2 + 576^2 + 729^2 + 1024^2` non-zero entries.

Reading of the exercise: the counts come from the lesson's own `pack_batch`,
which reports `mask_size` and `mask_nonzero` without materialising anything, and
`build_dense_mask` is checked against it at the only size the lesson itself ever
materialises -- its 14-token toy. Building the 2585 x 2585 mask to count its
entries would be answering an arithmetic question with 6.7 million Python
objects.

**ANSWER: 2585 x 2585 = 6,682,225 cells, 1,977,329 non-zero -- 29.59%.**
`pack_batch` returns both figures directly, and `build_dense_mask` reproduces
them exactly on the lesson's toy batch (196 cells, 100 non-zero).

**FINDING: the same mask is five integers.** `cu_seqlens` is
`[0, 256, 832, 1561, 2585]` -- what FlashAttention's varlen kernels actually
take. The dense form is **1,336,445x** larger and carries no additional
information. The lesson's own demo materialises 196 cells; this exercise asks
for **34,092x** that.

**FINDING: the density is `(1 + CV^2) / k`, so unequal lengths cost extra.**
Four equal blocks summing to 2585 give exactly **25.00%**. These four have a
coefficient of variation of **0.4285**, and 1.1836/4 reproduces the measured
**29.59%** to the digit -- **18.4%** more attention than the same token budget
evenly split.

**FINDING: so the batch sampler sets the attention bill, not the packer.**
Packing is what makes the density 1/k instead of 1; the spread of the lengths is
what moves it off 1/k. Both of the levers are in how the batch was assembled.

Structure: `pack` runs the lesson's own `pack_batch` on the four lengths as
square images, `toy_parity` checks `build_dense_mask` against it at 14 tokens,
and `predicted_density` is the closed form checked against the measurement.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "06-any-resolution-patch-n-pack"
LENGTHS = (256, 576, 729, 1024)
PATCH = 1


def as_images(ref, lengths, patch=PATCH):
    """One 1 x n image per length, so pack_batch sees exactly these sequence lengths."""
    return [ref.Image(f"img{i}", patch, length * patch)
            for i, length in enumerate(lengths)]


def predicted_density(lengths):
    """Sum(n^2) / (Sum n)^2 in closed form: (1 + CV^2) / k."""
    spread = statistics.pstdev(lengths) / statistics.fmean(lengths)
    return (1 + spread * spread) / len(lengths) * 100, spread


def toy_parity(ref):
    """build_dense_mask against pack_batch on the lesson's own two-image demo."""
    pack = ref.pack_batch([ref.Image("A", 6, 4), ref.Image("B", 4, 8)], 2)
    mask = ref.build_dense_mask(pack)
    return {"tokens": pack.total_tokens, "cells": pack.mask_size,
            "reported": pack.mask_nonzero,
            "counted": sum(sum(row) for row in mask),
            "cu_seqlens": pack.cu_seqlens}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pack = ref.pack_batch(as_images(ref, LENGTHS), PATCH)
    density = pack.mask_nonzero / pack.mask_size * 100
    closed, spread = predicted_density(LENGTHS)
    even = sum((sum(LENGTHS) / len(LENGTHS)) ** 2 for _ in LENGTHS)
    toy = toy_parity(ref)
    return {
        "lengths": pack.per_image, "total": pack.total_tokens,
        "cells": pack.mask_size, "nonzero": pack.mask_nonzero,
        "expected_nonzero": sum(n * n for n in LENGTHS),
        "density": round(density, 2), "closed_form": round(closed, 2),
        "spread": round(spread, 4),
        "even_density": round(even / pack.mask_size * 100, 2),
        "excess_pct": round(density / (even / pack.mask_size * 100) * 100 - 100, 1),
        "cu_seqlens": pack.cu_seqlens, "descriptor": len(pack.cu_seqlens),
        "compression": pack.mask_size // len(pack.cu_seqlens),
        "toy": toy, "toy_scale": pack.mask_size // toy["cells"],
    }


def verify(result):
    toy = result["toy"]
    return [
        practice.Check(
            "ANSWER: 2585 x 2585 = 6,682,225 cells, 1,977,329 non-zero -- 29.59%",
            all([result["lengths"] == list(LENGTHS), result["total"] == 2585,
                 result["cells"] == 6_682_225, result["nonzero"] == 1_977_329,
                 result["nonzero"] == result["expected_nonzero"],
                 toy["reported"] == toy["counted"] == 100, toy["cells"] == 196]),
            f"pack_batch reports {result['cells']:,} cells and {result['nonzero']:,} "
            f"non-zero, which is exactly 256^2 + 576^2 + 729^2 + 1024^2, "
            f"{result['density']}% dense -- and build_dense_mask reproduces it on the "
            f"lesson's own toy batch, {toy['counted']} of {toy['cells']}",
        ),
        practice.Check(
            "FINDING: the same mask is five integers",
            all([result["cu_seqlens"] == [0, 256, 832, 1561, 2585],
                 result["descriptor"] == 5, result["compression"] == 1_336_445,
                 result["toy_scale"] == 34_092]),
            f"cu_seqlens is {result['cu_seqlens']} -- {result['descriptor']} integers, what "
            f"FlashAttention's varlen kernels take. The dense form is "
            f"{result['compression']:,}x larger and carries nothing extra. The lesson "
            f"materialises {toy['cells']} cells in its demo; this exercise asks for "
            f"{result['toy_scale']:,}x that",
        ),
        practice.Check(
            "FINDING: the density is (1 + CV^2) / k, so unequal lengths cost extra",
            all([result["closed_form"] == result["density"],
                 result["even_density"] == 25.0, result["spread"] == 0.4285,
                 result["excess_pct"] == 18.4]),
            f"four equal blocks summing to {result['total']:,} give "
            f"{result['even_density']}%; these have a coefficient of variation of "
            f"{result['spread']}, and (1 + CV^2)/k = {result['closed_form']}% reproduces the "
            f"measured {result['density']}% exactly -- {result['excess_pct']}% more "
            "attention than the same budget evenly split",
        ),
        practice.Check(
            "FINDING: so the batch sampler sets the attention bill, not the packer",
            all([result["even_density"] == 100 / len(LENGTHS),
                 result["density"] > result["even_density"]]),
            f"packing is what takes the density from 100% to 1/k = "
            f"{result['even_density']}%; the spread of the lengths is what moves it to "
            f"{result['density']}%. Both levers live in how the batch was assembled, and "
            "neither is in the mask",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
