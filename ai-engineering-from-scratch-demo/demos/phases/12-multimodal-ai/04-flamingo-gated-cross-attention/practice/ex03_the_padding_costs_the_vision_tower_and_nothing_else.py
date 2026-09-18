"""Exercise 3 — the padding costs the vision tower and nothing else.

    Read OpenFlamingo Section 3.2 (arXiv:2308.01390) on how they handle
    multiple images in a batch when each prompt has a different image count.
    Describe the padding strategy.

Reading of the exercise: "describe" is answered by implementing the strategy and
pricing it, because the strategy itself is one sentence -- pad the image axis to
the batch maximum and carry a mask -- and everything interesting about it is a
number. The mask is the lesson's own `interleaved_mask`, run over a padded
sequence, so the claim that the pads are unattended is checked rather than
asserted.

**ANSWER: pad the image axis to the batch maximum, mask the pads.** For image
counts (1, 3, 5, 2) that is a (4, 5) image tensor -- **20** slots holding 11
images, **45%** of them padding.

**FINDING: the waste is exactly `1 - mean/max`, so it depends on the spread and
not on the size.** (1, 3, 5, 2) and (10, 30, 50, 20) waste the identical 45%,
and a batch whose prompts all carry the same number of images wastes 0 however
many that is.

**FINDING: none of the waste is in the attention.** Run over a padded sequence,
`interleaved_mask` gives every padding column **0** attended positions, because
a text token only ever reaches a *preceding* image and the pads are appended
after all the text. The cost is 9 image encodes and, at Flamingo's 64 latents
per image, **576** resampler latents computed and then discarded -- 45% of the
1,280 the batch produces.

**FINDING: sorting the batch halves it.** Bucketing the same four prompts by
image count into pairs gives (1, 2) padded to 2 and (3, 5) padded to 5: **14**
slots for 11 images, **21.4%**. The padding strategy is free to fix; the batch
sampler is where the cost actually lives.

Structure: `pad` is the strategy, `waste` its cost, `bucketed` the sorted
variant, and `pad_columns` runs the lesson's mask over a padded sequence.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "04-flamingo-gated-cross-attention"
COUNTS = (1, 3, 5, 2)
SCALED = tuple(count * 10 for count in COUNTS)
LATENTS, BUCKET = 64, 2


def pad(counts):
    """OpenFlamingo's strategy: one (batch, max_images) tensor, pads masked out."""
    return max(counts) * len(counts)


def waste(counts, slots=None):
    slots = pad(counts) if slots is None else slots
    return round((slots - sum(counts)) / slots * 100, 1)


def bucketed(counts, size=BUCKET):
    """Sort by image count, then pad within each bucket instead of across the batch."""
    order = sorted(counts)
    return sum(pad(order[i:i + size]) for i in range(0, len(order), size))


def pad_columns(ref, images, pads, text_per_image=2):
    """Attended positions per image column, with the pads appended after all text."""
    sequence = [label for i in range(images)
                for label in [f"IMG{i}"] + [f"t{i}{j}" for j in range(text_per_image)]]
    sequence += [f"IMG_PAD{i}" for i in range(pads)]
    mask = ref.interleaved_mask(sequence)
    return [sum(row[column] for row in mask) for column in range(images + pads)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    slots, real = pad(COUNTS), sum(COUNTS)
    columns = pad_columns(ref, min(COUNTS), max(COUNTS) - min(COUNTS))
    buckets = bucketed(COUNTS)
    return {
        "counts": list(COUNTS), "shape": (len(COUNTS), max(COUNTS)),
        "slots": slots, "real": real, "waste": waste(COUNTS),
        "scaled_waste": waste(SCALED),
        "uniform_waste": waste((5, 5, 5, 5)),
        "mean_over_max": round(real / len(COUNTS) / max(COUNTS), 3),
        "real_columns": columns[:min(COUNTS)], "pad_columns": columns[min(COUNTS):],
        "wasted_latents": (slots - real) * LATENTS, "total_latents": slots * LATENTS,
        "buckets": [sorted(COUNTS)[i:i + BUCKET] for i in range(0, len(COUNTS), BUCKET)],
        "bucket_slots": buckets, "bucket_waste": waste(COUNTS, buckets),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: pad the image axis to the batch maximum, mask the pads -- 45% padding",
            all([result["shape"] == (4, 5), result["slots"] == 20, result["real"] == 11,
                 result["waste"] == 45.0]),
            f"image counts {result['counts']} give a {result['shape'][0]} x "
            f"{result['shape'][1]} image tensor -- {result['slots']} slots holding "
            f"{result['real']} images, {result['waste']}% padding, carried alongside a mask "
            "that says which slots are real",
        ),
        practice.Check(
            "FINDING: the waste is exactly 1 - mean/max, so it depends on spread not size",
            all([result["scaled_waste"] == result["waste"],
                 result["uniform_waste"] == 0.0,
                 round((1 - result["mean_over_max"]) * 100, 1) == result["waste"]]),
            f"{result['counts']} and {list(SCALED)} both waste {result['waste']}%, because "
            f"the ratio is 1 - mean/max = 1 - {result['mean_over_max']}. A batch whose "
            f"prompts all carry the same count wastes {result['uniform_waste']}%, however "
            "many that is",
        ),
        practice.Check(
            "FINDING: none of the waste is in the attention",
            all([result["pad_columns"] == [0, 0, 0, 0], all(result["real_columns"]),
                 result["wasted_latents"] == 576, result["total_latents"] == 1280]),
            f"run over a padded sequence, interleaved_mask attends {result['pad_columns']} "
            f"positions in the padding columns against {result['real_columns']} in the real "
            f"one, because a text token only reaches a preceding image. The cost is "
            f"{result['slots'] - result['real']} image encodes and "
            f"{result['wasted_latents']} of {result['total_latents']} resampler latents at "
            f"{LATENTS} per image -- computed, then discarded",
        ),
        practice.Check(
            "FINDING: sorting the batch halves it",
            all([result["buckets"] == [[1, 2], [3, 5]], result["bucket_slots"] == 14,
                 result["bucket_waste"] == 21.4]),
            f"bucketing the same four prompts by image count into pairs gives "
            f"{result['buckets']} -- {result['bucket_slots']} slots for {result['real']} "
            f"images, {result['bucket_waste']}% against {result['waste']}%. The padding "
            "strategy is free to fix; the batch sampler is where the cost lives",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
