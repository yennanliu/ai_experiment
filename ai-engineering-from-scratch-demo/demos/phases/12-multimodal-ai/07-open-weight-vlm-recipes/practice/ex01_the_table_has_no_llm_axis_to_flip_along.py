"""Exercise 1 — the table has no LLM axis to flip along.

    Read MM1 Section 3.2. For a fixed 2B LLM at budget 50M images, which encoder
    wins? Would the answer flip at 13B LLM? Why?

Reading of the exercise: the lesson ships the encoder evidence as a table, so
the table is parsed out of its own printed output and asked the question
directly, rather than the paper being paraphrased. The result is that the
question cannot be answered from this lesson, and saying exactly which column is
missing is more useful than guessing the answer.

**ANSWER: there is no encoder that wins, and no LLM axis to flip along.** The
lesson's `compare_encoders` table is fixed at one 7B LLM and carries no image
budget, so neither 2B nor 13B appears anywhere in it. Within the one setting it
does report, the winner is benchmark-dependent: **SigLIP+DINOv2 concat** takes
CV-Bench at 67.0 and **InternViT-6B** takes MMMU at 43.0 and DocVQA at 78.0.

**FINDING: encoder choice matters 4.3x more for DocVQA than for MMMU.** The same
five encoders span **6.0** points of MMMU, 11.0 of CV-Bench and **26.0** of
DocVQA. A single "image encoder: 20% of variance" figure is task-blind by
construction, and the task is where all the signal is.

**FINDING: the axis decomposition sums to 115%.** `axis_impact` computes
`total_weight`, prints a note saying the weights are "rebased from ~115% to
~100% after rounding", and then prints the unrebased numbers. The rebasing is
described and not performed.

**FINDING: an additive decomposition cannot hold the question.** "Would the
answer flip at 13B?" is an encoder x LLM-size interaction, and the model has one
number for encoder and one for LLM size with no term that multiplies them. The
shape of the evidence rules the question out before any of its values are read.

Structure: `encoder_rows` parses the lesson's own `compare_encoders` output,
`spread` is the per-benchmark range, and `axis_weights` parses the
decomposition to check what it sums to.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "07-open-weight-vlm-recipes"
BENCHMARKS = ("MMMU", "CV-Bench", "DocVQA")
ASKED = (2, 13)


def captured(fn):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn()
    return buffer.getvalue()


def encoder_rows(ref):
    """The lesson's own encoder table, parsed back out of its printed form."""
    rows = {}
    for line in captured(ref.compare_encoders).splitlines():
        numbers = re.findall(r"\d+\.\d", line)
        if len(numbers) == len(BENCHMARKS) and line.startswith(" ") is False:
            rows[line[:32].strip()] = [float(value) for value in numbers]
    return rows


def axis_weights(ref):
    """The bar-chart rows only -- the trailing note carries two more percentages."""
    return [int(match.group(1))
            for line in captured(ref.axis_impact).splitlines()
            if (match := re.match(r"^\S.*?(\d+)% #*$", line))]


def spread(rows, column):
    values = [row[column] for row in rows.values()]
    return round(max(values) - min(values), 1)


def winner(rows, column):
    return max(rows, key=lambda name: rows[name][column])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = encoder_rows(ref)
    weights = axis_weights(ref)
    text = captured(ref.compare_encoders)
    return {
        "encoders": rows, "count": len(rows),
        "winners": {name: winner(rows, i) for i, name in enumerate(BENCHMARKS)},
        "spreads": {name: spread(rows, i) for i, name in enumerate(BENCHMARKS)},
        "spread_ratio": round(spread(rows, 2) / spread(rows, 0), 1),
        "weights": weights, "weight_sum": sum(weights),
        "claims_rebase": "rebased" in captured(ref.axis_impact),
        "llm_sizes_present": [size for size in ASKED if f"{size}B" in text],
        "budget_mentioned": "50M" in text,
    }


def verify(result):
    rows, winners, spreads = result["encoders"], result["winners"], result["spreads"]
    return [
        practice.Check(
            "ANSWER: no encoder wins, and there is no LLM axis to flip along",
            all([result["count"] == 5, result["llm_sizes_present"] == [],
                 not result["budget_mentioned"],
                 winners["CV-Bench"] == "SigLIP + DINOv2 concat",
                 winners["MMMU"] == winners["DocVQA"] == "InternViT-6B @ 448"]),
            f"the table is fixed at one 7B LLM and names no image budget, so neither of "
            f"{list(ASKED)}B appears in it. Within the one setting it does report the winner "
            f"is benchmark-dependent: {winners}. Five encoders, three answers",
        ),
        practice.Check(
            "FINDING: encoder choice matters 4.3x more for DocVQA than for MMMU",
            all([spreads == {"MMMU": 6.0, "CV-Bench": 11.0, "DocVQA": 26.0},
                 result["spread_ratio"] == 4.3]),
            f"the same five encoders span {spreads} -- a ratio of {result['spread_ratio']} "
            "between the widest and the narrowest. One 'image encoder: 20% of variance' "
            "figure is task-blind by construction, and the task is where the signal is",
        ),
        practice.Check(
            "FINDING: the axis decomposition sums to 115%",
            all([result["weight_sum"] == 115, result["claims_rebase"],
                 result["weights"] == [60, 20, 5, 10, 15, 5]]),
            f"the weights are {result['weights']}, summing to {result['weight_sum']}%, and "
            "axis_impact prints a note saying they are rebased to ~100% after rounding. The "
            "rebasing is described in the note and never performed on the numbers above it",
        ),
        practice.Check(
            "FINDING: an additive decomposition cannot hold the question",
            all([len(result["weights"]) == 6, rows["DINOv2 ViT-g/14 @ 224"][2] == 52.0,
                 rows["SigLIP SO400m/14 @ 384"][2] == 75.0]),
            f"'would the answer flip at 13B' is an encoder x LLM-size interaction, and the "
            f"decomposition has {len(result['weights'])} additive terms with nothing that "
            "multiplies two of them. That DINOv2 scores 52.0 on DocVQA against SigLIP's 75.0 "
            "while beating it on CV-Bench is the same kind of interaction, one axis lower, "
            "and the decomposition cannot hold that either",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
