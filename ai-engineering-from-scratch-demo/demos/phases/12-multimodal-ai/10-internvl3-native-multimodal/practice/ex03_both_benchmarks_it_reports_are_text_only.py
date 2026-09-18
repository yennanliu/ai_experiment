"""Exercise 3 — both benchmarks it reports are text-only.

    Read MM1.5 Section 4 on forgetting. Name the exact benchmark where post-hoc
    training showed the largest regression. How much did the regression cost?

Reading of the exercise: the paper is not reachable from here, so the answer is
given from the two regression rows the lesson does ship, with the ranking's
reliability checked rather than assumed -- and "how much did it cost" is
converted into the only unit this lesson supplies a conversion for, the
GPU-hours from its own compute row.

**ANSWER: GSM8K, on the lesson's own evidence.** It is the larger regression at
both ends of the reported range -- **-3 to -10** against MMLU's -2 to -8 -- and
at the midpoint, -6.5 against -5.0.

**FINDING: the ranking sits inside the width of its own reporting.** The two
midpoints are **1.5** points apart and the MMLU range alone is **6** points
wide, four times that gap. The orderings at the two ends happen to agree, but a
1.5-point separation read off two 6- and 7-point ranges is not a measurement of
which benchmark regressed more.

**FINDING: both benchmarks are text-only.** MMLU and GSM8K are language
evaluations. Everything this lesson reports about the cost of post-hoc
multimodal training is measured on the modality that training did not add --
there is no multimodal regression number anywhere in the table, in either
direction.

**ANSWER: the cost, in the only unit the lesson converts to, is 41,538
GPU-hours per GSM8K point.** The native column pays 270,000 GPU-hours to hold
both regressions at zero; against GSM8K's 6.5-point midpoint that is 41,538 a
point, and against MMLU's 5.0 it is 54,000. The cheaper point to buy is the
larger regression, which is the opposite of how the trade is usually stated.

Structure: `REGRESSIONS` transcribes the lesson's two rows, `rank_at` orders
them at one end of the range, and `per_point` converts the compute premium into
a price per benchmark point.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "10-internvl3-native-multimodal"
REGRESSIONS = {"MMLU": (2, 8), "GSM8K": (3, 10)}
PREMIUM = 270_000
TEXT_ONLY = ("MMLU", "GSM8K")


def rank_at(end):
    """The benchmark with the larger regression, read at one end of the ranges."""
    return max(REGRESSIONS, key=lambda name: REGRESSIONS[name][end])


def midpoint(span):
    return sum(span) / len(span)


def per_point(span, premium=PREMIUM):
    return round(premium / midpoint(span))


def table_text(ref):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.posthoc_vs_native_table()
    return buffer.getvalue()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    text = table_text(ref)
    midpoints = {name: midpoint(span) for name, span in REGRESSIONS.items()}
    prices = {name: per_point(span) for name, span in REGRESSIONS.items()}
    return {
        "regressions": REGRESSIONS,
        "at_best": rank_at(0), "at_worst": rank_at(1),
        "agree": rank_at(0) == rank_at(1),
        "midpoints": midpoints,
        "midpoint_gap": round(abs(midpoints["GSM8K"] - midpoints["MMLU"]), 1),
        "widest_range": max(span[1] - span[0] for span in REGRESSIONS.values()),
        "rows_in_table": [name for name in REGRESSIONS if name in text],
        "multimodal_rows": [name for name in text.split() if name.startswith("MM")
                            and name not in TEXT_ONLY],
        "prices": prices,
        "cheaper_point": min(prices, key=prices.get),
        "larger_regression": rank_at(1),
    }


def verify(result):
    prices, midpoints = result["prices"], result["midpoints"]
    return [
        practice.Check(
            "ANSWER: GSM8K, on the lesson's own evidence",
            all([result["at_worst"] == "GSM8K", result["at_best"] == "GSM8K",
                 result["agree"], midpoints == {"MMLU": 5.0, "GSM8K": 6.5},
                 sorted(result["rows_in_table"]) == ["GSM8K", "MMLU"]]),
            f"the table reports {result['regressions']} and GSM8K is larger at both ends of "
            f"the range and at the midpoint ({midpoints}). Both rows are present in the "
            f"lesson's own output: {sorted(result['rows_in_table'])}",
        ),
        practice.Check(
            "FINDING: the ranking sits inside the width of its own reporting",
            all([result["midpoint_gap"] == 1.5, result["widest_range"] == 7,
                 result["widest_range"] > 4 * result["midpoint_gap"] - 1]),
            f"the two midpoints are {result['midpoint_gap']} points apart and the widest "
            f"reported range is {result['widest_range']} points, several times that gap. The "
            "orderings at the two ends agree, but a 1.5-point separation read off two "
            "multi-point ranges is not a measurement of which regressed more",
        ),
        practice.Check(
            "FINDING: both benchmarks are text-only",
            all([result["multimodal_rows"] == [], len(TEXT_ONLY) == 2,
                 sorted(result["rows_in_table"]) == sorted(TEXT_ONLY)]),
            "MMLU and GSM8K are language evaluations, and they are the only regression rows "
            "in the table. Everything the lesson reports about the cost of post-hoc "
            "multimodal training is measured on the modality that training did not add; "
            "there is no multimodal regression number anywhere, in either direction",
        ),
        practice.Check(
            "ANSWER: 41,538 GPU-hours per GSM8K point, the cheaper of the two",
            all([prices == {"MMLU": 54_000, "GSM8K": 41_538},
                 result["cheaper_point"] == result["larger_regression"]]),
            f"the native column pays {PREMIUM:,} GPU-hours to hold both regressions at zero; "
            f"against the midpoints that is {prices} per point. The cheaper point to buy is "
            f"{result['cheaper_point']}, the larger regression -- the opposite of how the "
            "trade is usually stated",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
