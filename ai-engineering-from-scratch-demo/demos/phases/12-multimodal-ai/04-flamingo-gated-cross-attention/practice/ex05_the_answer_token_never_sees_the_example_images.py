"""Exercise 5 — the answer token never sees the example images.

    In-context few-shot: construct a prompt with 4 examples of "image → color of
    main object" for a new Flamingo variant. Describe the expected accuracy
    pattern as you vary the number of examples from 0 to 8.

Reading of the exercise: the accuracy pattern cannot be measured without the
model, so what is measured instead is the thing that produces it -- the prompt
is constructed for every shot count from 0 to 8 and handed to the lesson's own
`interleaved_mask`, which decides what each answer token is allowed to see. The
expected pattern is then argued from that, rather than recited.

**ANSWER: the answer token sees exactly one image -- its own -- at every shot
count from 0 to 8.** Under Flamingo's most-recent-image mask the k example
images are invisible to the token that produces the answer. Whatever few-shot
learning happens here is happening over the example *text*.

**ANSWER: so the expected pattern is a step, not a curve.** Nearly all of the
gain lands between 0 and 1 shot, where the model first sees the output format --
a bare colour word rather than a sentence -- and 1 to 8 adds little, because
each further shot contributes another (invisible image, colour word) pair and
the colour words are the part that carries. Past a few shots the curve is flat
or drifts down as the label distribution in the prompt starts biasing the
answer.

**FINDING: the mask is 11.1% dense at 8 shots and never denser than one per
row.** 17 text positions by 9 images is 153 cells with **17** True -- every row
sums to exactly 1, at every k. The cross-attention cost is linear in text
length and independent of image count, which is the property the mask is for.

**FINDING: 8 of the 9 images are encoded and then hidden.** At 64 latents an
image that is **576** resampler latents computed for images no answer token can
reach -- the same 45% shape as the padding in exercise 3, arrived at from the
other direction.

Structure: `prompt` builds the interleaved sequence for k shots, `mask_stats`
runs the lesson's `interleaved_mask` over it, and `SHOTS` is the 0-to-8 sweep.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "04-flamingo-gated-cross-attention"
SHOTS = tuple(range(9))
TEXT_PER_SHOT, LATENTS = 2, 64


def prompt(shots):
    """k demonstrations of image -> colour, then the query image and its answer slot."""
    sequence = [label for i in range(shots)
                for label in (f"IMG{i}", f"ex{i}_prompt", f"ex{i}_colour")]
    return sequence + [f"IMG{shots}", "answer"]


def mask_stats(ref, shots):
    mask = ref.interleaved_mask(prompt(shots))
    rows, columns = len(mask), len(mask[0])
    return {"text": rows, "images": columns, "cells": rows * columns,
            "attended": sum(sum(row) for row in mask),
            "row_sums": sorted({sum(row) for row in mask}),
            "answer_sees": sum(mask[-1]),
            "answer_index": mask[-1].index(True) if any(mask[-1]) else None}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {shots: mask_stats(ref, shots) for shots in SHOTS}
    widest = sweep[max(SHOTS)]
    return {
        "sweep": sweep,
        "answer_sees": sorted({stats["answer_sees"] for stats in sweep.values()}),
        "answer_is_own": all(stats["answer_index"] == shots
                             for shots, stats in sweep.items()),
        "row_sums": sorted({value for stats in sweep.values()
                            for value in stats["row_sums"]}),
        "cells": widest["cells"], "attended": widest["attended"],
        "density_pct": round(widest["attended"] / widest["cells"] * 100, 1),
        "images": widest["images"], "hidden": widest["images"] - 1,
        "hidden_latents": (widest["images"] - 1) * LATENTS,
        "total_latents": widest["images"] * LATENTS,
        "hidden_pct": round((widest["images"] - 1) / widest["images"] * 100, 1),
        "text_growth": [sweep[k]["text"] for k in SHOTS],
    }


def verify(result):
    sweep = result["sweep"]
    return [
        practice.Check(
            "ANSWER: the answer token sees exactly one image -- its own -- at every k",
            all([result["answer_sees"] == [1], result["answer_is_own"],
                 len(sweep) == len(SHOTS)]),
            f"across k = {SHOTS[0]} to {SHOTS[-1]} the final text position attends "
            f"{result['answer_sees']} image, and it is always the query's own. Under "
            "Flamingo's most-recent-image mask the example images are invisible to the token "
            "that produces the answer, so the few-shot signal is carried by the example text",
        ),
        practice.Check(
            "ANSWER: so the expected pattern is a step, not a curve",
            all([result["text_growth"] == [1, 3, 5, 7, 9, 11, 13, 15, 17],
                 sweep[0]["images"] == 1, sweep[4]["images"] == 5]),
            f"text positions grow {result['text_growth']} while the answer's visual context "
            "stays at one image, so every shot after the first adds colour words and nothing "
            "visual. Nearly all of the gain lands between 0 and 1 -- where the output format "
            "is established -- and the tail is flat, or drifts as the prompt's label "
            "distribution starts biasing the answer",
        ),
        practice.Check(
            "FINDING: the mask is 11.1% dense at 8 shots and never denser than one per row",
            all([result["cells"] == 153, result["attended"] == 17,
                 result["density_pct"] == 11.1, result["row_sums"] == [1]]),
            f"at 8 shots the mask is {sweep[8]['text']} text by {sweep[8]['images']} images "
            f"-- {result['cells']} cells with {result['attended']} True, "
            f"{result['density_pct']}% -- and every row sums to exactly 1 at every k. The "
            "cross-attention cost is linear in text length and independent of image count",
        ),
        practice.Check(
            "FINDING: 8 of the 9 images are encoded and then hidden",
            all([result["hidden"] == 8, result["hidden_latents"] == 512,
                 result["total_latents"] == 576, result["hidden_pct"] == 88.9]),
            f"{result['hidden']} of {result['images']} images -- {result['hidden_pct']}% -- "
            f"are run through the vision tower and resampler, {result['hidden_latents']} of "
            f"{result['total_latents']} latents at {LATENTS} an image, and no answer token "
            "can reach them. Exercise 3's padding waste arrived at from the other direction",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
