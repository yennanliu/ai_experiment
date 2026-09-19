"""Exercise 4 — the plateau is a property of the benchmark.

    Molmo ships 4B and 72B models. The 4B is competitive with closed 7B VLMs;
    the 72B beats Llama-3.2-90B-Vision on 11/11 benchmarks. What does that tell
    you about the LLM-size plateau hypothesis?

Reading of the exercise: the lesson's `RECIPES` table carries two model families
at two LLM sizes each -- Molmo at 7B and 72B, MM1 at 3B and 30B -- which is the
only controlled LLM-size evidence in the lesson, so the hypothesis is tested
against those four rows rather than against the 4B model the exercise names,
which the table does not have. Each family gives a within-family delta across
roughly one decade of LLM.

**ANSWER: the plateau is not visible, and it is not one number.** A 10x LLM buys
**+8.8** MMMU in the Molmo family and **+6.1** in the MM1 family -- about +7 per
decade, with no sign of flattening. But the same 10x buys only **+1.1** DocVQA
in the Molmo family, against +8.8 MMMU: an **8.0x** difference between two
benchmarks measured on the same two checkpoints.

**FINDING: DocVQA has already plateaued at 7B and MMMU has not by 72B.**
Molmo-7B-D is at **92.4** DocVQA and Molmo-72B at 93.5; the remaining headroom
is 6.5 points and a decade of LLM takes 1.1 of it. On MMMU the same pair moves
45.3 -> 54.1 with 45.9 still above it. "The LLM-size plateau" is a claim about a
benchmark, not about LLMs.

**FINDING: the lesson's own plateau figure is its own largest model.**
`axis_impact` says "7B -> 70B plateau around MMMU 55", and the highest MMMU in
the whole table is Molmo-72B at **54.1**. The hypothesis is stated at a point the
evidence reaches and does not pass, so nothing in this lesson can falsify it.

**FINDING: the data axis is worth as much as the LLM axis and is weighted less.**
`compare_data` moves MMMU 40.0 -> 47.0 at a fixed encoder and a fixed 7B LLM --
**+7.0**, the same size as a decade of LLM -- while `axis_impact` weights data
mix at 10% and LLM size at 15%.

Structure: `families` pairs the two same-recipe-different-LLM rows, `decade`
normalises each delta to a factor of ten in LLM size, and `data_span` reads the
data table for the comparison.
"""

from __future__ import annotations

import contextlib
import io
import math
import re

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "07-open-weight-vlm-recipes"
FAMILIES = {"Molmo": ("Molmo-7B-D", "Molmo-72B"), "MM1": ("MM1-3B", "MM1-30B")}
BENCHMARKS = ("mmmu", "cv_bench", "docvqa")
CEILING = 100.0


def by_name(ref):
    return {recipe.name: recipe for recipe in ref.RECIPES}


def delta(table, small, large, field):
    return round(getattr(table[large], field) - getattr(table[small], field), 1)


def decade(table, small, large):
    """LLM-size ratio in decades, so two families can be compared."""
    return round(math.log10(table[large].llm_b / table[small].llm_b), 2)


def data_span(ref):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.compare_data()
    scores = [float(line[28:36]) for line in buffer.getvalue().splitlines()
              if re.match(r"^\S.*\d+\.\d", line) and line[28:36].strip()]
    return round(min(scores), 1), round(max(scores), 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = by_name(ref)
    low, high = data_span(ref)
    molmo, mm1 = FAMILIES["Molmo"], FAMILIES["MM1"]
    return {
        "deltas": {family: {field: delta(table, *pair, field) for field in BENCHMARKS}
                   for family, pair in FAMILIES.items()},
        "decades": {family: decade(table, *pair) for family, pair in FAMILIES.items()},
        "mmmu_vs_docvqa": round(delta(table, *molmo, "mmmu")
                                / delta(table, *molmo, "docvqa"), 1),
        "docvqa_pair": [table[molmo[0]].docvqa, table[molmo[1]].docvqa],
        "mmmu_pair": [table[molmo[0]].mmmu, table[molmo[1]].mmmu],
        "docvqa_headroom": round(CEILING - table[molmo[1]].docvqa, 1),
        "mmmu_headroom": round(CEILING - table[molmo[1]].mmmu, 1),
        "best_mmmu": max(recipe.mmmu for recipe in ref.RECIPES),
        "claimed_plateau": 55,
        "data_low": low, "data_high": high, "data_span": round(high - low, 1),
        "mm1_mmmu": delta(table, *mm1, "mmmu"),
    }


def verify(result):
    deltas, decades = result["deltas"], result["decades"]
    return [
        practice.Check(
            "ANSWER: the plateau is not visible, and it is not one number",
            all([deltas["Molmo"]["mmmu"] == 8.8, deltas["MM1"]["mmmu"] == 6.1,
                 decades == {"Molmo": 1.01, "MM1": 1.0},
                 result["mmmu_vs_docvqa"] == 8.0]),
            f"a decade of LLM buys {deltas['Molmo']['mmmu']} MMMU in the Molmo family and "
            f"{deltas['MM1']['mmmu']} in MM1 -- the two families span "
            f"{list(decades.values())} decades -- with no flattening. The same Molmo pair "
            f"moves DocVQA by {deltas['Molmo']['docvqa']}, "
            f"{result['mmmu_vs_docvqa']}x less, on the same two checkpoints",
        ),
        practice.Check(
            "FINDING: DocVQA has already plateaued at 7B and MMMU has not by 72B",
            all([result["docvqa_pair"] == [92.4, 93.5], result["mmmu_pair"] == [45.3, 54.1],
                 result["docvqa_headroom"] == 6.5, result["mmmu_headroom"] == 45.9]),
            f"Molmo goes {result['docvqa_pair']} on DocVQA -- {result['docvqa_headroom']} "
            f"points of headroom left and a decade of LLM taking "
            f"{deltas['Molmo']['docvqa']} of it -- against {result['mmmu_pair']} on MMMU "
            f"with {result['mmmu_headroom']} still above. The plateau is a claim about a "
            "benchmark, not about LLMs",
        ),
        practice.Check(
            "FINDING: the lesson's own plateau figure is its own largest model",
            all([result["best_mmmu"] == 54.1,
                 result["best_mmmu"] < result["claimed_plateau"],
                 result["claimed_plateau"] - result["best_mmmu"] < 1.0]),
            f"axis_impact says '7B -> 70B plateau around MMMU {result['claimed_plateau']}', "
            f"and the highest MMMU anywhere in the table is {result['best_mmmu']}. The "
            "hypothesis is stated at a point the evidence reaches and does not pass, so "
            "nothing in this lesson can falsify it",
        ),
        practice.Check(
            "FINDING: the data axis is worth as much as the LLM axis and is weighted less",
            all([result["data_low"] == 40.0, result["data_high"] == 47.0,
                 result["data_span"] == 7.0,
                 abs(result["data_span"] - deltas["Molmo"]["mmmu"]) <= 2.0]),
            f"compare_data moves MMMU {result['data_low']} -> {result['data_high']} at a "
            f"fixed encoder and a fixed 7B LLM -- {result['data_span']} points, within 2 of "
            f"the {deltas['Molmo']['mmmu']} a decade of LLM buys -- while axis_impact weights "
            "data mix at 10% and LLM size at 15%",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
