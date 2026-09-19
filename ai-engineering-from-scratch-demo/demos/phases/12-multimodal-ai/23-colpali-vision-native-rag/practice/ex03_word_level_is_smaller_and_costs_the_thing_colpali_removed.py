"""Exercise 3 — word-level is smaller, and costs the thing ColPali removed.

    ColPali indexes pages as patch sets. What changes if we instead index at the
    word level (as ColBERT does)? Trade-offs?

Reading of the exercise: the index arithmetic is done first, because the obvious
expectation -- that words are far cheaper than patches -- turns out to be a 31%
difference rather than an order of magnitude, and once the storage argument is
that weak the trade-off is entirely about what each unit can represent.

**ANSWER: the index shrinks 31.4% and the pipeline grows an OCR pass.** A page at
a stated 500 words is **250.0 KiB** against 729 patches' **364.5 KiB** -- the
same order, because a word and a patch are about the same size on a page. What
changes is that word-level needs the text extracted first, which is the step
ColPali exists to delete.

**FINDING: quantisation beats changing the unit by 5.5x.** Word-level raw is
**250.0 KiB**; ColPali at PQ 8x is **45.6 KiB**. So the unit change is the more
expensive of the two ways to shrink the index, and it is the one that requires
reading the page first.

**FINDING: what the unit change actually costs is everything not in the text
stream.** A word vector cannot encode a checkbox, a stamp, a line in a table, a
chart's bars, a signature, or the fact that two numbers are in the same column.
Lesson 12.22 measures the alternative: LayoutLMv3's bbox stream restores position
and gives up scale invariance, and its text ids come from an OCR pass whose
per-field error that lesson puts at a stated 2%.

**FINDING: and the failure modes invert.** Patches degrade gracefully -- a
blurred region still yields vectors, just less distinctive ones. Words fail
discretely: OCR either produces a token or it does not, and a word that was never
extracted is unretrievable at any recall. MaxSim over patches has no equivalent
of a missing row.

Structure: `index_bytes` prices one unit type per page, `UNITS` holds the three
options, and `ratio` compares them against the compressed baseline.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "23-colpali-vision-native-rag"
DIM, FLOAT_BYTES = 128, 4
PATCHES, WORDS = 729, 500
PQ_FACTOR = 8
OCR_ERROR = 0.02          # Lesson 12.22 ex05's stated assumption, not a
                          # figure the lesson itself measures


def index_bytes(units, dim=DIM, size=FLOAT_BYTES):
    return units * dim * size


def kib(value):
    return round(value / 1024, 1)


def surviving_rows(unreadable):
    """Index rows left when `unreadable` of the page cannot be OCR'd.

    A patch grid is laid over the page geometrically, so every region yields a
    vector however illegible. A word index has a row only where OCR produced a
    token, so an unread word is an absent row rather than a poor one.
    """
    return {"patches": PATCHES, "words": round(WORDS * (1 - unreadable))}


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    patch_raw = index_bytes(PATCHES)
    word_raw = index_bytes(WORDS)
    patch_pq = patch_raw // PQ_FACTOR
    return {
        "units": {"patches": PATCHES, "words": WORDS},
        "patch_kib": kib(patch_raw), "word_kib": kib(word_raw),
        "shrink_pct": round((1 - word_raw / patch_raw) * 100, 1),
        "same_order": 0.5 < word_raw / patch_raw < 2.0,
        "patch_pq_kib": kib(patch_pq),
        "pq_beats_words": round(word_raw / patch_pq, 1),
        "needs_ocr": True, "ocr_error": OCR_ERROR,
        "colpali_ocr_steps": 0,
        "unrepresentable": ("checkbox", "stamp", "table rule", "chart bar",
                           "signature", "column alignment"),
        "graceful": "patches", "discrete": "words",
        "rows_clean": surviving_rows(0.0), "rows_tenth": surviving_rows(0.1),
        "rows_illegible": surviving_rows(1.0),
        "nontext_rows": 0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the index shrinks 31.4% and the pipeline grows an OCR pass",
            all([result["patch_kib"] == 364.5, result["word_kib"] == 250.0,
                 result["shrink_pct"] == 31.4, result["same_order"],
                 result["needs_ocr"], result["colpali_ocr_steps"] == 0]),
            f"a page at a stated {WORDS} words is {result['word_kib']} KiB against "
            f"{PATCHES} patches' {result['patch_kib']} KiB -- {result['shrink_pct']}%, the "
            "same order, because a word and a patch are about the same size on a page. What "
            "changes is that word-level needs the text extracted first, which is the step "
            "ColPali exists to delete",
        ),
        practice.Check(
            "FINDING: quantisation beats changing the unit by 5.5x",
            all([result["patch_pq_kib"] == 45.6, result["pq_beats_words"] == 5.5]),
            f"word-level raw is {result['word_kib']} KiB and ColPali at PQ {PQ_FACTOR}x is "
            f"{result['patch_pq_kib']} KiB -- {result['pq_beats_words']}x smaller. If storage "
            "is the concern, quantisation wins and does not require reading the page",
        ),
        practice.Check(
            "FINDING: the unit change costs everything not in the text stream",
            all([len(result["unrepresentable"]) == 6,
                 "checkbox" in result["unrepresentable"],
                 result["ocr_error"] == 0.02, result["nontext_rows"] == 0,
                 result["rows_clean"]["words"] == WORDS]),
            f"a word vector cannot encode {list(result['unrepresentable'])}: on a perfectly "
            f"read page the word index still holds {result['rows_clean']['words']} rows and "
            f"{result['nontext_rows']} of them describe any of those six, because the index "
            f"has no unit for them. Lesson 12.22 covers the alternative -- LayoutLMv3's bbox "
            f"stream restores position and gives up scale invariance -- and its exercise 5 "
            f"assumes an OCR per-field error of {result['ocr_error']:.0%}; the lesson itself "
            "measures no such rate",
        ),
        practice.Check(
            "FINDING: and the failure modes invert",
            all([result["graceful"] == "patches", result["discrete"] == "words",
                 result["rows_tenth"] == {"patches": PATCHES, "words": 450},
                 result["rows_illegible"] == {"patches": PATCHES, "words": 0},
                 result["rows_illegible"]["patches"] == result["rows_clean"]["patches"]]),
            f"losing a tenth of the page to illegibility leaves {result['rows_tenth']} index "
            f"rows and losing all of it leaves {result['rows_illegible']} -- the patch count "
            f"never moves, because the grid is laid over the page geometrically and a "
            f"blurred region still yields vectors, just less distinctive ones. Words fail "
            f"discretely: OCR either produces a token or it does not, and a word never "
            "extracted is unretrievable at any recall. MaxSim over patches has no equivalent "
            "of a missing row",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
