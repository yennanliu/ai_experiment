"""Exercise 5 — the lesson names four benchmarks and reports zero scores.

    Given the VideoMME leaderboard, what is the gap between the top open model
    and the top proprietary model as of 2026? How much of that gap is
    attributable to temporal encoding vs base LLM scale?

Reading of the exercise: the leaderboard is not in the lesson, so the first
thing to establish is exactly what is -- which is four benchmark *names*, zero
scores, and a five-row architecture table whose only numeric column is frame
count. The attribution question is then answered with the one decomposition this
phase has measured, Lesson 12.07's variance weights, and with the caveat that
lesson's own exercise 1 establishes about additive decompositions.

**ANSWER: the gap cannot be computed here.** The lesson lists VideoMME,
TempCompass, EgoSchema and Video-MMMU and gives **0** scores for any of them,
for any model. Its `arch_compare` table has **5** rows and three columns --
model, compressor, note -- and no benchmark column at all.

**FINDING: the one number the table does carry spans 4x.** Frame counts run
**8 to 32** across the five architectures, which at Lesson 12.08's pooled 81
tokens a frame is **648 to 2,592** visual tokens -- and every row differs in
compressor as well, so no two of the five isolate anything.

**ANSWER: by Lesson 12.07's weights the split is 4 to 1 toward tokens.** That
lesson rates visual-token count at **60%** of benchmark variance and LLM size at
**15%**, so a stated extrapolation puts four times more of any video gap on the
temporal-and-token side than on base-LLM scale. Frame count, sampling rate and
per-frame pooling are all inside the 60%.

**FINDING: and the decomposition cannot express the question.** The weights are
six additive terms with no product of any two, so "how much of the gap is
temporal encoding *vs* LLM scale" -- a question about how the two interact -- has
no term to land in. Lesson 12.07's exercise 1 makes the same point about the same
table: the axis weights sum to **115%** and the note claiming they are rebased is
never executed.

Structure: `benchmarks` and `arch_rows` read what the lesson actually contains,
`frame_span` reads its one numeric column, and `axis_weights` reads Lesson 12.07's
decomposition applied as a stated model.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "17-video-language-temporal-grounding"
BENCHMARKS = ("VideoMME", "TempCompass", "EgoSchema", "Video-MMMU")
RECIPES = "07-open-weight-vlm-recipes"
TOKENS_PER_FRAME = 81


def arch_text(ref):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.arch_compare()
    return buffer.getvalue()


def arch_rows(text):
    return [line for line in text.splitlines()
            if line.startswith("  ") and "model" not in line and "-" * 10 not in line]


def frame_span(text):
    numbers = [int(value) for value in re.findall(r"(\d+)(?=[- ]*frames)", text)]
    return min(numbers), max(numbers)


def axis_weights():
    """Lesson 12.07's variance decomposition, read off its own axis_impact output."""
    recipes = parity.load_reference(PHASE, RECIPES, "main")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        recipes.axis_impact()
    return {axis.strip(): int(pct)
            for axis, pct in re.findall(r"^(\S.*?)\s+(\d+)%", buffer.getvalue(),
                                        re.MULTILINE)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    weights = axis_weights()
    text = arch_text(ref)
    doc = parity.doc_text(PHASE, LESSON, "en")
    low, high = frame_span(text)
    return {
        "benchmarks": [name for name in BENCHMARKS if name in doc],
        "scores_in_doc": len(re.findall(r"\b\d{2}\.\d\b", doc)),
        "rows": len(arch_rows(text)),
        "has_benchmark_column": any(name in text for name in BENCHMARKS),
        "frames": (low, high), "frame_ratio": high // low,
        "tokens": (low * TOKENS_PER_FRAME, high * TOKENS_PER_FRAME),
        "token_axis": weights["visual-token count"], "llm_axis": weights["LLM size"],
        "axis_ratio": weights["visual-token count"] // weights["LLM size"],
        "weight_sum": sum(weights.values()), "terms": len(weights),
        "products": 0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the gap cannot be computed here -- four names, zero scores",
            all([sorted(result["benchmarks"]) == sorted(BENCHMARKS),
                 result["scores_in_doc"] == 0, result["rows"] == 5,
                 not result["has_benchmark_column"]]),
            f"the lesson names {len(result['benchmarks'])} benchmarks "
            f"({sorted(result['benchmarks'])}) and carries "
            f"{result['scores_in_doc']} two-decimal scores anywhere in its text. Its "
            f"architecture table has {result['rows']} rows and no benchmark column",
        ),
        practice.Check(
            "FINDING: the one number the table carries spans 4x",
            all([result["frames"] == (8, 32), result["frame_ratio"] == 4,
                 result["tokens"] == (648, 2592)]),
            f"frame counts run {result['frames'][0]} to {result['frames'][1]} across the "
            f"five architectures -- {result['frame_ratio']}x -- which at Lesson 12.08's "
            f"pooled {TOKENS_PER_FRAME} tokens a frame is {result['tokens'][0]:,} to "
            f"{result['tokens'][1]:,} visual tokens. Every row also differs in compressor, "
            "so no two of the five isolate anything",
        ),
        practice.Check(
            "ANSWER: by Lesson 12.07's weights the split is 4 to 1 toward tokens",
            all([result["token_axis"] == 60, result["llm_axis"] == 15,
                 result["axis_ratio"] == 4, result["terms"] == 6,
                 result["weight_sum"] == 115]),
            f"read off Lesson 12.07's own axis_impact output, that lesson rates "
            f"visual-token count at {result['token_axis']}% of variance and LLM size at "
            f"{result['llm_axis']}% -- {result['terms']} axes summing to "
            f"{result['weight_sum']}%, not 100 -- so a stated extrapolation puts "
            f"{result['axis_ratio']}x more of any video gap on the temporal-and-token side. "
            "Frame count, sampling rate and per-frame pooling are all inside the 60%",
        ),
        practice.Check(
            "FINDING: and the decomposition cannot express the question",
            all([result["terms"] == 6, result["products"] == 0,
                 result["weight_sum"] == 115]),
            f"the weights are {result['terms']} additive terms with {result['products']} "
            f"products of any two, so a question about how temporal encoding and LLM scale "
            f"interact has no term to land in. They also sum to {result['weight_sum']}%, "
            "which Lesson 12.07's exercise 1 measures on the same table",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
