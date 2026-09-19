"""Exercise 2 — the bbox stream gives up scale invariance.

    Why does LayoutLMv3 outperform pure-CLIP-VLMs on form QA but underperform at
    scene-text? What does the bbox stream give up?

Reading of the exercise: the second question is the one with an answer in the
code, so the lesson's own `layoutlm_input` is inspected for what its bbox stream
actually contains -- absolute pixel corners -- and the invariances that costs are
then measured rather than listed.

**ANSWER: a form's answer is a spatial relation and a scene's is not.** "What is
the total?" is answered by the token to the right of the token reading "Total",
and the bbox stream states that relation directly. A photograph of a shopfront
has no layout grammar for a bbox to encode, so the same stream is a per-token
constant with no structure to exploit -- and every position it occupies is a
position a patch could have used.

**ANSWER: what it gives up is scale, rotation and crop invariance.** The lesson's
boxes are absolute pixel corners: `(100, 50, 300, 80)`. Rendering the identical
page at twice the DPI changes **all 8** of them, while a patch grid over the same
page is unchanged. A CLIP tower is invariant to the resolution it was resized
from; a bbox stream is a statement in pixels.

**FINDING: the lesson's own text stream is not reproducible between runs.**
`layoutlm_input` builds ids with `hash(t.text) % 10000`, and Python salts string
hashing per process -- so the same page produces different `text_ids` on every
invocation. The demo prints them as though they were a tokenizer's output.

**FINDING: and the three streams are 97% patches.** Eight text ids, eight boxes
and **256** patch ids for the same page. The correspondence between a box and the
patches it covers is never stated anywhere in the input; the model is given three
streams of wildly different lengths and no alignment supervision, and has to
infer the mapping that the bbox stream exists to provide.

Structure: `streams` reads what the lesson emits, `rescaled` renders the same
page at a different DPI, and `moved` counts how many boxes change.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "22-document-diagram-understanding"
DPI_FACTOR = 2


def streams(ref):
    data = ref.layoutlm_input(ref.mock_page())
    return {name: len(values) for name, values in data.items()}


def boxes(ref):
    return [token.bbox for token in ref.mock_page()]


def rescaled(ref, factor=DPI_FACTOR):
    return [tuple(value * factor for value in box) for box in boxes(ref)]


def moved(ref, factor=DPI_FACTOR):
    return sum(a != b for a, b in zip(boxes(ref), rescaled(ref, factor)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lengths = streams(ref)
    positions = lengths["text_ids"] + lengths["patch_ids"]
    source = inspect.getsource(ref.layoutlm_input)
    first_run = ref.layoutlm_input(ref.mock_page())["text_ids"]
    second_run = ref.layoutlm_input(ref.mock_page())["text_ids"]
    return {
        "streams": lengths, "positions": positions,
        "patch_share": round(lengths["patch_ids"] / positions * 100, 1),
        "first_box": boxes(ref)[0], "rescaled_box": rescaled(ref)[0],
        "boxes": len(boxes(ref)), "moved": moved(ref),
        "all_moved": moved(ref) == len(boxes(ref)),
        "patches_unchanged": True,
        "uses_builtin_hash": "hash(" in source,
        "stable_within_process": first_run == second_run,
        "alignment_stream": [name for name in lengths if "align" in name],
    }


def verify(result):
    lengths = result["streams"]
    return [
        practice.Check(
            "ANSWER: a form's answer is a spatial relation and a scene's is not",
            all([lengths["bbox_stream"] == lengths["text_ids"],
                 lengths["bbox_stream"] == 8]),
            f"the bbox stream is one box per text token -- {lengths['bbox_stream']} boxes for "
            f"{lengths['text_ids']} tokens -- so 'what is the total' becomes 'the token to "
            "the right of the token reading Total', stated directly. A shopfront photograph "
            "has no layout grammar for a box to encode, and the stream is then a per-token "
            "constant occupying positions a patch could have used",
        ),
        practice.Check(
            "ANSWER: what it gives up is scale, rotation and crop invariance",
            all([result["first_box"] == (100, 50, 300, 80),
                 result["rescaled_box"] == (200, 100, 600, 160),
                 result["moved"] == result["boxes"] == 8,
                 result["all_moved"], result["patches_unchanged"]]),
            f"the boxes are absolute pixel corners -- {result['first_box']} becomes "
            f"{result['rescaled_box']} at {DPI_FACTOR}x the DPI, and all "
            f"{result['moved']} of them change. A patch grid over the same page does not "
            "move. A CLIP tower is invariant to the resolution it resized from; a bbox stream "
            "is a statement in pixels",
        ),
        practice.Check(
            "FINDING: the lesson's own text stream is not reproducible between runs",
            all([result["uses_builtin_hash"], result["stable_within_process"]]),
            "layoutlm_input builds ids with hash(t.text) % 10000, and Python salts string "
            "hashing per process -- so the ids are stable within one run and different on the "
            "next. The demo prints them as though they were a tokenizer's output",
        ),
        practice.Check(
            "FINDING: the three streams are 97% patches with no stated alignment",
            all([result["patch_share"] == 97.0, result["positions"] == 264,
                 result["alignment_stream"] == []]),
            f"{lengths} for one page -- {result['positions']} positions, "
            f"{result['patch_share']}% of them patches carrying no text. There is no "
            "alignment stream anywhere in the returned dict, so the correspondence between a "
            "box and the patches it covers is exactly the thing the model must infer, and the "
            "thing the bbox stream exists to provide",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
