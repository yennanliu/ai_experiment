"""Exercise 1 — HWC to CHW and back, and why the round trip cannot fail.

    **(Easy)** Create a 2x2 RGB `uint8` array with four distinct colors. Convert
    HWC to CHW and back, print both shapes, and prove the round trip preserves
    every value.

Reading of the exercise: "prove the round trip preserves every value" is check 1, and checks
2-4 ask what that proof is worth — `hwc_to_chw` is `transpose`, so no value is ever copied,
the result aliases its input, and the thing that actually costs (contiguity) is the thing the
exercise does not mention. Check 5 records that this exercise's English and Chinese texts ask
for two different pieces of work.
"""

from __future__ import annotations

from harness import coverage, parity, practice

PHASE, LESSON = "04-computer-vision", "01-image-fundamentals"
COLORS = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]]   # red green blue yellow


def four_colors(numpy):
    return numpy.array(COLORS, dtype=numpy.uint8)


def alias(ref, numpy) -> tuple:
    """Write through the CHW array and see whether the HWC original moved."""
    hwc = four_colors(numpy)
    before = int(hwc[0, 0, 0])
    ref.hwc_to_chw(hwc)[0, 0, 0] = 7
    return before, int(hwc[0, 0, 0])


def texts() -> tuple:
    """This exercise as the lesson states it in each language."""
    return (coverage.exercise_block(parity.doc_text(PHASE, LESSON, "en"))[0],
            coverage.exercise_block(parity.doc_text(PHASE, LESSON, "zh"))[0])


def solve():
    numpy = parity.try_numpy()
    ref = parity.load_reference(PHASE, LESSON, "main")
    hwc = four_colors(numpy)
    chw = ref.hwc_to_chw(hwc)
    back = ref.chw_to_hwc(chw)
    english, chinese = texts()
    return {"shapes": (hwc.shape, chw.shape, back.shape),
            "equal": bool(numpy.array_equal(hwc, back)),
            "distinct": len({tuple(p) for row in COLORS for p in row}),
            "owns": bool(chw.flags["OWNDATA"]), "base": chw.base is hwc,
            "shared": bool(numpy.shares_memory(back, hwc)),
            "contiguous": (bool(hwc.flags["C_CONTIGUOUS"]), bool(chw.flags["C_CONTIGUOUS"])),
            "copy_contiguous": bool(numpy.ascontiguousarray(chw).flags["OWNDATA"]),
            "alias": alias(ref, numpy),
            "en_zh": (english[:70], chinese[:38]),
            "shared_words": len({w for w in english.lower().split() if len(w) > 3}
                                & {w for w in chinese.lower().split() if len(w) > 3})}


def verify(result):
    hwc_shape, chw_shape, back_shape = result["shapes"]
    before, after = result["alias"]
    return [
        practice.Check("ANSWER: the round trip preserves every value, and both shapes are what "
                       "the exercise asks for",
                       result["equal"] and hwc_shape == back_shape == (2, 2, 3)
                       and chw_shape == (3, 2, 2) and result["distinct"] == 4,
                       f"a {result['distinct']}-colour 2x2 image — red, green, blue, yellow — goes "
                       f"{hwc_shape} -> {chw_shape} -> {back_shape}, and all "
                       f"{hwc_shape[0] * hwc_shape[1] * hwc_shape[2]} values compare equal"),
        practice.Check("MECHANISM: it cannot fail — `transpose` permutes strides, it does not "
                       "move data",
                       not result["owns"] and result["base"] and result["shared"],
                       f"the CHW array does not own its buffer (OWNDATA "
                       f"{result['owns']}), its `.base` is the HWC input, and the round-tripped "
                       f"array still shares memory with it. Nothing is copied, cast or rounded "
                       f"anywhere in the path, so there is no arithmetic for the proof to catch"),
        practice.Check("FINDING: the two arrays alias, so 'convert' is the wrong word for it",
                       (before, after) == (255, 7),
                       f"writing 7 into the CHW array's [0, 0, 0] changes the HWC original from "
                       f"{before} to {after}. A reader who treats `hwc_to_chw` as a conversion and "
                       f"then edits the result has edited the input too"),
        practice.Check("FINDING: what the transpose does cost is contiguity, which the exercise "
                       "never mentions",
                       result["contiguous"] == (True, False) and result["copy_contiguous"],
                       f"C_CONTIGUOUS is {result['contiguous'][0]} for HWC and "
                       f"{result['contiguous'][1]} for CHW. Every consumer that wants a flat "
                       f"buffer — a framework tensor, `.tobytes()`, a memcpy to the GPU — pays "
                       f"for `ascontiguousarray`, which does allocate. That copy, not the "
                       f"transpose, is the cost of the layout change"),
        practice.Check("FINDING: the lesson asks two different questions in its two languages",
                       result["shared_words"] < 3,
                       f"en: {result['en_zh'][0]!r}...; zh: {result['en_zh'][1]!r}... — the "
                       f"English builds a 2x2 array in memory, the Chinese loads a JPEG with "
                       f"OpenCV and Pillow and reconciles their channel order. They share "
                       f"{result['shared_words']} words over three letters. Exercises 2 and 3 "
                       f"match; this one does not, so the manifest carries both verbatim"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
