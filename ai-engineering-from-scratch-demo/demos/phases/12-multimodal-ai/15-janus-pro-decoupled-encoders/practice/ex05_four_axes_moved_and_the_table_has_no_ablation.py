"""Exercise 5 — four axes moved, and the table has no ablation.

    Read Janus-Pro Section 4.2 on data scaling. Which data stage contributes
    most to the T2I quality gain vs Janus?

Reading of the exercise: the answer is read off the lesson's own Janus-vs-
Janus-Pro table, in both relative and absolute terms since they can disagree,
and then the attribution itself is checked -- because the table records four
axes moving at once and reports one outcome, which is the shape Lesson 12.07's
exercise 5 shows cannot separate anything.

**ANSWER: stage 2, the unified stage, at +176.9%.** 26M pairs to 72M against
stage 1's +25.0% and stage 3's +16.7%, and it leads in absolute terms too: **46M**
new pairs against 18M and 0.2M.

**FINDING: stage 3's percentage is in a different unit.** Stages 1 and 2 count
image-text *pairs*; stage 3 counts *instructions*. Its +16.7% is 200,000
instructions against stage 2's 46,000,000 pairs -- a **230x** difference in
absolute terms that the percentage column hides by putting all three in the same
row format.

**FINDING: four axes moved and one outcome was measured.** Three data stages and
a **5.4x** parameter increase separate the two columns, and the table has **6**
rows and **0** ablations. Attributing the +0.19 GenEval to stage 2 is not
identified by this evidence -- the same table would look identical if stage 2
contributed nothing and the parameters did all of it.

**FINDING: even the one subtractable axis cannot be subtracted.** 1.3B to 7B is
**0.731** decades, and Lesson 12.07 measured **+7 MMMU** per decade -- but it has
no GenEval-per-decade row, because neither of its two families varied LLM size at
a fixed generation benchmark. So the honest answer to "which stage contributes
most" is that this table ranks the *inputs* and says nothing about which of them
produced the output.

Structure: `STAGES` transcribes the three data rows with their units, `growth`
gives relative and absolute change, and `axes_moved` counts what separates the
two columns.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "15-janus-pro-decoupled-encoders"
STAGES = {
    "stage 1 (alignment)": (72.0, 90.0, "M pairs"),
    "stage 2 (unified)": (26.0, 72.0, "M pairs"),
    "stage 3 (instruction)": (1.2, 1.4, "M instructions"),
}
PARAMS = (1.3, 7.0)
GENEVAL = (0.61, 0.80)
TABLE_ROWS = 6


def growth(before, after):
    return round((after / before - 1) * 100, 1), round(after - before, 1)


def axes_moved(stages=STAGES):
    return len(stages) + 1          # three data stages plus the parameter count


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    moves = {name: growth(before, after) for name, (before, after, _) in STAGES.items()}
    relative = {name: row[0] for name, row in moves.items()}
    absolute = {name: row[1] for name, row in moves.items()}
    units = {name: unit for name, (_, _, unit) in STAGES.items()}
    return {
        "relative": relative, "absolute": absolute, "units": units,
        "largest_relative": max(relative, key=relative.get),
        "largest_absolute": max(absolute, key=absolute.get),
        "agree": max(relative, key=relative.get) == max(absolute, key=absolute.get),
        "distinct_units": len(set(units.values())),
        "unit_ratio": round(absolute["stage 2 (unified)"]
                            / absolute["stage 3 (instruction)"]),
        "axes": axes_moved(), "rows": TABLE_ROWS, "ablations": 0,
        "param_ratio": round(PARAMS[1] / PARAMS[0], 1),
        "param_decades": round(math.log10(PARAMS[1] / PARAMS[0]), 3),
        "geneval_gain": round(GENEVAL[1] - GENEVAL[0], 2),
        "geneval_per_decade_known": False,
    }


def verify(result):
    relative, absolute, units = result["relative"], result["absolute"], result["units"]
    return [
        practice.Check(
            "ANSWER: stage 2, the unified stage, at +176.9%",
            all([relative == {"stage 1 (alignment)": 25.0, "stage 2 (unified)": 176.9,
                              "stage 3 (instruction)": 16.7},
                 result["largest_relative"] == "stage 2 (unified)",
                 result["largest_absolute"] == "stage 2 (unified)",
                 result["agree"], absolute["stage 2 (unified)"] == 46.0]),
            f"the three data rows grow {relative}% and {absolute} in absolute terms, so "
            f"{result['largest_relative']} leads on both readings -- 26M pairs to 72M, "
            f"{absolute['stage 2 (unified)']}M new pairs against "
            f"{absolute['stage 1 (alignment)']}M and "
            f"{absolute['stage 3 (instruction)']}M",
        ),
        practice.Check(
            "FINDING: stage 3's percentage is in a different unit",
            all([result["distinct_units"] == 2,
                 units["stage 3 (instruction)"] == "M instructions",
                 result["unit_ratio"] == 230]),
            f"stages 1 and 2 count image-text pairs and stage 3 counts instructions: "
            f"{units}. Its +{relative['stage 3 (instruction)']}% is 200,000 instructions "
            f"against stage 2's 46,000,000 pairs -- a {result['unit_ratio']}x difference "
            "that the percentage column hides by giving all three the same row format",
        ),
        practice.Check(
            "FINDING: four axes moved and one outcome was measured",
            all([result["axes"] == 4, result["rows"] == TABLE_ROWS,
                 result["ablations"] == 0, result["param_ratio"] == 5.4]),
            f"three data stages and a {result['param_ratio']}x parameter increase separate "
            f"the two columns -- {result['axes']} axes -- and the table has "
            f"{result['rows']} rows and {result['ablations']} ablations. Attributing the "
            f"+{result['geneval_gain']} GenEval to stage 2 is not identified by this "
            "evidence: the same table would look identical if stage 2 contributed nothing",
        ),
        practice.Check(
            "FINDING: even the one subtractable axis cannot be subtracted",
            all([result["param_decades"] == 0.731,
                 not result["geneval_per_decade_known"]]),
            f"{PARAMS[0]}B to {PARAMS[1]}B is {result['param_decades']} decades, and Lesson "
            "12.07 measured +7 MMMU per decade within two model families -- but it has no "
            "GenEval-per-decade row, because neither of its families varied LLM size at a "
            "fixed generation benchmark. So the table ranks the inputs and says nothing "
            "about which produced the output",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
