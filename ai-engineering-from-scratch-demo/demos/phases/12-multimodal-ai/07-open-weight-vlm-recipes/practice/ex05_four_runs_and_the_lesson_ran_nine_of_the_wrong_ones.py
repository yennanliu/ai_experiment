"""Exercise 5 — four runs, and the lesson ran nine of the wrong ones.

    Design an ablation table to isolate data-mix quality from encoder quality on
    a 7B VLM. How many training runs minimum? Propose the four axis settings.

Reading of the exercise: "how many minimum" is read as the smallest design that
can *separate* the two effects rather than the smallest that touches both axes,
because those are different numbers and the difference is the interaction term.
The lesson's own two tables are then audited as a design, since they are an
attempt at exactly this and the exercise is asking the reader to do better.

**ANSWER: four runs -- a 2x2 factorial.** Three runs (a baseline plus one move
along each axis) measure both main effects and *assume* additivity; the fourth
cell is what turns the interaction from an assumption into a measurement. With
four you can state "SigLIP is worth +2.5 MMMU" and "PixMo is worth +5.3" and
also whether the two overlap.

**FINDING: the lesson runs 9 configurations and cannot separate anything.**
`compare_encoders` is 5 encoders at one data mix; `compare_data` is 5 mixes at
one encoder. Nine distinct cells, laid out as a cross -- and **zero** of them
are off both axes at once, which is the only place an interaction is visible.
Nine runs buy what three would.

**FINDING: the two tables do not share a baseline, and the lesson has three
numbers for the same cell.** SigLIP + 7B + LLaVA-Inst&ShareGPT4V appears as
**41.0** in `compare_encoders`, **42.0** in `compare_data`, and **40.0** as
`Prismatic-7B default` in `RECIPES`. A **2.0**-point spread -- **80%** of the
+2.5 encoder effect that the first table exists to report, in the baseline the
two tables were supposed to share.

**ANSWER: so the four settings are {SigLIP SO400m, CLIP L/14} x
{LLaVA-Inst-150k, PixMo}.** The encoder pair is the widest measured gap that
does not change the visual token count, so the comparison is not secretly a
resolution ablation; the data pair is the distilled-versus-human-caption
contrast that is the entire Molmo thesis, worth **5.3** MMMU in the lesson's own
table against the encoder pair's 2.5. Two axes the lesson measures separately,
whose sizes are close enough that an interaction of either sign would change the
conclusion.

Structure: `runs_needed` is the design arithmetic, `cross_design` counts the
lesson's own cells, and `shared_cell` collects the three values the lesson gives
for one configuration.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "07-open-weight-vlm-recipes"
LEVELS = 2
AXES = 2
SHARED = "SigLIP SO400m/14 @ 384"
SHARED_DATA = "+ ShareGPT4V"
SHARED_RECIPE = "Prismatic-7B default"


def runs_needed(axes=AXES, levels=LEVELS):
    """Full factorial, and the additive shortcut that assumes the interaction away."""
    return {"factorial": levels ** axes, "additive": axes * (levels - 1) + 1}


def captured(fn):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn()
    return buffer.getvalue()


def encoder_table(ref):
    return {line[:32].strip(): float(re.findall(r"\d+\.\d", line)[0])
            for line in captured(ref.compare_encoders).splitlines()
            if len(re.findall(r"\d+\.\d", line)) == 3}


def data_table(ref):
    return {line[:28].strip(): float(line[28:36])
            for line in captured(ref.compare_data).splitlines()
            if line[28:36].strip() and re.match(r"^\S.*\d\.\d", line)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    encoders, data = encoder_table(ref), data_table(ref)
    recipes = {recipe.name: recipe.mmmu for recipe in ref.RECIPES}
    shared = [encoders[SHARED], data[SHARED_DATA], recipes[SHARED_RECIPE]]
    return {
        **runs_needed(),
        "encoder_cells": len(encoders), "data_cells": len(data),
        "total_cells": len(encoders) + len(data) - 1,
        "off_axis_cells": 0,
        "shared": shared, "shared_spread": round(max(shared) - min(shared), 1),
        "encoder_effect": round(encoders[SHARED] - encoders["CLIP ViT-L/14 @ 336"], 1),
        "data_effect": round(data["PixMo (human caps only)"] - data["LLaVA-Inst-150k"], 1),
        "encoder_levels": ["SigLIP SO400m/14 @ 384", "CLIP ViT-L/14 @ 336"],
        "data_levels": ["LLaVA-Inst-150k", "PixMo (human caps only)"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four runs -- a 2x2 factorial",
            all([result["factorial"] == 4, result["additive"] == 3,
                 result["factorial"] - result["additive"] == 1]),
            f"{AXES} axes at {LEVELS} levels is {result['factorial']} cells. "
            f"{result['additive']} runs -- a baseline plus one move along each axis -- "
            "measure both main effects and assume additivity; the fourth cell is what turns "
            "the interaction from an assumption into a measurement",
        ),
        practice.Check(
            "FINDING: the lesson runs 9 configurations and cannot separate anything",
            all([result["encoder_cells"] == 5, result["data_cells"] == 5,
                 result["total_cells"] == 9, result["off_axis_cells"] == 0]),
            f"compare_encoders is {result['encoder_cells']} encoders at one data mix and "
            f"compare_data is {result['data_cells']} mixes at one encoder -- "
            f"{result['total_cells']} distinct cells laid out as a cross, with "
            f"{result['off_axis_cells']} off both axes at once. Nine runs buy what "
            f"{result['additive']} would",
        ),
        practice.Check(
            "FINDING: the lesson has three numbers for the same cell",
            all([result["shared"] == [41.0, 42.0, 40.0], result["shared_spread"] == 2.0,
                 round(result["shared_spread"] / result["encoder_effect"], 2) == 0.8]),
            f"SigLIP + 7B + LLaVA-Inst&ShareGPT4V reads {result['shared'][0]} in "
            f"compare_encoders, {result['shared'][1]} in compare_data and "
            f"{result['shared'][2]} as {SHARED_RECIPE} in RECIPES -- a "
            f"{result['shared_spread']}-point spread -- 80% of the "
            f"{result['encoder_effect']}-point encoder effect the first table exists to "
            "report. "
            "The two tables do not share a baseline, so their deltas cannot be added",
        ),
        practice.Check(
            "ANSWER: the four settings are {SigLIP, CLIP} x {LLaVA-Inst-150k, PixMo}",
            all([result["encoder_effect"] == 2.5, result["data_effect"] == 5.3,
                 len(result["encoder_levels"]) == len(result["data_levels"]) == LEVELS]),
            f"the encoder pair {result['encoder_levels']} is the widest measured gap that "
            f"does not change the visual token count, worth {result['encoder_effect']} MMMU, "
            f"so the comparison is not secretly a resolution ablation. The data pair "
            f"{result['data_levels']} is the distilled-versus-human-caption contrast, worth "
            f"{result['data_effect']}. Close enough in size that an interaction of either "
            "sign would change the conclusion",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
