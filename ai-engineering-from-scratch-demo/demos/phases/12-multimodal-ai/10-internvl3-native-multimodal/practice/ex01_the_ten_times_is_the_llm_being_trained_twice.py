"""Exercise 1 — the 10x is the LLM being trained twice.

    Estimate the compute delta between InternVL3-8B (native pretrain) and
    LLaVA-OneVision-7B (post-hoc). Ratio of GPU-hours approximately? What
    explains the gap?

Reading of the exercise: the ratio is read off the lesson's own post-hoc-vs-
native table rather than estimated, since the table carries both numbers, and
"what explains the gap" is answered from the row immediately below them --
which is the only row in the table that could. The premium is then priced
against the two rows the table says it buys.

**ANSWER: 10x -- ~300k GPU-hours against ~30k.**

**FINDING: the explanation is the next row down: "base LLM reuse -- yes / no".**
Native pretraining repeats the LLM's own pretraining with vision in the corpus.
The 270k-GPU-hour premium is not a multimodal cost; it is a language-model cost
being paid a second time because the mixture changed.

**FINDING: the premium buys the two regression rows, at 54,000 GPU-hours per
MMLU point.** Post-hoc training is priced at "-2 to -8" MMLU and "-3 to -10"
GSM8K against native's 0. At the midpoints that is 270,000 GPU-hours to avoid 5
MMLU points and 6.5 GSM8K points -- **54,000** and **41,538** GPU-hours each.

**FINDING: the table is qualitative except in three rows, and two of those are
ranges.** Of the 8 comparison rows, 3 carry numbers; the MMLU range spans **4x**
end to end and the GSM8K range **3.3x**. The MMLU range's own width, 6 points,
is four times the 1.5-point gap between the two midpoints -- so the ranking of
the two regressions sits inside the width of the reporting.

Structure: `TABLE` transcribes the three numeric rows of the lesson's own
comparison, `per_point` prices the premium against a midpoint regression, and
`width` is the end-to-end span of each reported range.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "10-internvl3-native-multimodal"
GPU_HOURS = {"post-hoc": 30_000, "native": 300_000}
REGRESSIONS = {"MMLU": (2, 8), "GSM8K": (3, 10)}
NUMERIC_ROWS, TOTAL_ROWS = 3, 8


def midpoint(span):
    return sum(span) / len(span)


def per_point(premium, span):
    return round(premium / midpoint(span))


def width(span):
    return round(span[1] / span[0], 2)


def table_text(ref):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.posthoc_vs_native_table()
    return buffer.getvalue()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    text = table_text(ref)
    premium = GPU_HOURS["native"] - GPU_HOURS["post-hoc"]
    midpoints = {name: midpoint(span) for name, span in REGRESSIONS.items()}
    return {
        "ratio": GPU_HOURS["native"] // GPU_HOURS["post-hoc"],
        "premium": premium,
        "reuse_row": "base LLM reuse" in text,
        "reuse_values": ("yes" in text and "no" in text),
        "midpoints": midpoints,
        "per_point": {name: per_point(premium, span) for name, span in REGRESSIONS.items()},
        "widths": {name: width(span) for name, span in REGRESSIONS.items()},
        "mmmu_span": REGRESSIONS["MMLU"][1] - REGRESSIONS["MMLU"][0],
        "midpoint_gap": round(abs(midpoints["GSM8K"] - midpoints["MMLU"]), 1),
        "numeric_rows": NUMERIC_ROWS, "total_rows": TOTAL_ROWS,
        "approximate": text.count("~"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 10x -- ~300k GPU-hours against ~30k",
            all([result["ratio"] == 10, result["premium"] == 270_000,
                 result["approximate"] == 2]),
            f"the lesson's own table gives {GPU_HOURS['native']:,} against "
            f"{GPU_HOURS['post-hoc']:,} GPU-hours -- a factor of {result['ratio']} and a "
            f"premium of {result['premium']:,}. Both figures carry a tilde; there are "
            f"{result['approximate']} of them in the table and they are on this row",
        ),
        practice.Check(
            "FINDING: the explanation is the next row down -- base LLM reuse, yes / no",
            all([result["reuse_row"], result["reuse_values"]]),
            "native pretraining repeats the language model's own pretraining with vision in "
            "the corpus, which is what the 'base LLM reuse' row records. The "
            f"{result['premium']:,}-GPU-hour premium is not a multimodal cost -- it is a "
            "language-model cost paid a second time because the mixture changed",
        ),
        practice.Check(
            "FINDING: the premium buys the two regression rows, at 54,000 GPU-hours a point",
            all([result["midpoints"] == {"MMLU": 5.0, "GSM8K": 6.5},
                 result["per_point"] == {"MMLU": 54_000, "GSM8K": 41_538}]),
            f"post-hoc is priced at {REGRESSIONS['MMLU']} MMLU and {REGRESSIONS['GSM8K']} "
            f"GSM8K against native's 0. At the midpoints {result['midpoints']} the premium "
            f"works out at {result['per_point']} GPU-hours per point avoided",
        ),
        practice.Check(
            "FINDING: the table is qualitative except in three rows, and two are ranges",
            all([result["widths"] == {"MMLU": 4.0, "GSM8K": 3.33},
                 result["mmmu_span"] == 6, result["midpoint_gap"] == 1.5,
                 result["mmmu_span"] == 4 * result["midpoint_gap"],
                 result["numeric_rows"] < result["total_rows"]]),
            f"{result['numeric_rows']} of {result['total_rows']} comparison rows carry "
            f"numbers, and the two regression rows span {result['widths']} end to end. The "
            f"MMLU range's own width, {result['mmmu_span']} points, is 4x the "
            f"{result['midpoint_gap']}-point gap between the two midpoints -- so which "
            "regression is larger sits inside the width of the reporting",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
