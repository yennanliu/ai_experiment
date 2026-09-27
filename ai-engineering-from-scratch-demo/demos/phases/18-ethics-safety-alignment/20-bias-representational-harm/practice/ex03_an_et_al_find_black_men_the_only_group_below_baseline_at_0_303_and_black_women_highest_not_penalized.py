"""Exercise 3 — An et al. find Black men the only group below baseline (-0.303) and Black women highest, not penalized.

    Read An et al. 2025 (PNAS Nexus). Identify the two intersectional effects
    they report that single-axis gender evaluation would miss.

Reading of the exercise: the paper (An, Huang, Lin, Tai, PNAS Nexus 4(3)
pgaf089, March 2025; arXiv:2403.15281) scores ~361,000 randomized resumes
for 20 entry-level jobs, names signalling gender and race, balanced over the
four cells. Its headline regression (GPT-3.5 Turbo) reports each cell
relative to white men, on a 0-100 score, and hiring-probability gaps at an
80-point cutoff (about 35% hired). The runnable model of the argument is
the balanced 2x2 decomposition: what a single-axis evaluation computes from
those cells, and what it cannot see. The figures below are transcribed from
the paper's text into `CELLS`, `POOLED` and `HIRE_PP`, and the pooled
coefficients the paper prints are recomputed from the cells as a
transcription check.

**ANSWER: (1) an anti-Black-male penalty; (2) a race effect whose sign
depends on gender.** Relative to white men, Black women score +0.379, white
women +0.223 and Black men -0.303 -- Black men are the only group below
baseline (-1.4 points of hiring probability). Among women the Black-white
gap is +0.156; among men it is -0.303: an interaction of +0.459. A
single-axis gender evaluation sees "women +0.452" (+2.25 points of hiring
probability, pooled) and neither effect.

**FINDING: single-axis race evaluation sees a quarter of the penalty.** The
balanced cells reproduce the paper's pooled coefficients -- gender +0.4525
against its printed 0.452, race -0.0735 against -0.074 -- and that pooled
race gap (p < 0.1 in the paper) is 4.1x smaller than the Black-male gap
(-0.303, p < 0.001). In hiring probability the pooled race gap is -0.55
points against Black men's -1.4. Pooling across gender averages a penalty on men with a
premium on women.

**FINDING: the lesson states the result backwards.** docs/en.md says "GPT-4o
penalizes Black women in resume scoring more than Black men and more than
white women separately". In the paper Black women are the *highest*-scoring
group (+0.379 > +0.223 > 0 > -0.303); the penalty falls on Black men. The
cell numbers are GPT-3.5 Turbo's; the PNAS version adds that GPT-4o, Gemini
1.5 Flash, Claude 3.5 Sonnet and Llama 3-70b "also strongly favor female
candidates" and most show "significant bias against black male candidates"
(Llama 3-70b's is not significant). None of it is a Black-women penalty.

Structure: `pooled()` is the single-axis statistic (difference of balanced
group means, the same shape as the lesson's `weat_score`); the lesson's
sentence is read with `parity.doc_text`.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "20-bias-representational-harm"
# An et al. 2025, GPT-3.5 Turbo, score points vs white men (all p < 0.001), and the
# hiring-probability gaps in percentage points at the 80-point cutoff
CELLS = {("white", "male"): 0.0, ("white", "female"): 0.223,
         ("black", "female"): 0.379, ("black", "male"): -0.303}
HIRE_PP = {("white", "male"): 0.0, ("white", "female"): 1.4,
           ("black", "female"): 1.7, ("black", "male"): -1.4}
POOLED = {"gender": 0.452, "race": -0.074}            # female - male, black - white


def pooled(table, axis, plus):
    """Single-axis gap: mean over cells with `plus` on `axis` minus the rest (balanced)."""
    hi = [v for k, v in table.items() if k[axis] == plus]
    lo = [v for k, v in table.items() if k[axis] != plus]
    return sum(hi) / len(hi) - sum(lo) / len(lo)


def solve():
    c = CELLS
    doc = parity.doc_text(PHASE, LESSON)
    claim = re.search(r"An et al\. 2025 find ([^.]*)\.", doc).group(1)
    return {
        "gender": round(pooled(c, 1, "female"), 4), "race": round(pooled(c, 0, "black"), 4),
        "race_in_women": round(c["black", "female"] - c["white", "female"], 3),
        "race_in_men": round(c["black", "male"] - c["white", "male"], 3),
        "below_baseline": [k for k, v in c.items() if v < 0],
        "ranking": [k for k, _ in sorted(c.items(), key=lambda kv: -kv[1])],
        "gender_pp": round(pooled(HIRE_PP, 1, "female"), 2),
        "race_pp": round(pooled(HIRE_PP, 0, "black"), 2),
        "claim": claim,
    }


def verify(result):
    inter = round(result["race_in_women"] - result["race_in_men"], 3)
    ratio = round(CELLS["black", "male"] / result["race"], 1)
    return [
        practice.Check(
            "ANSWER: (1) an anti-Black-male penalty; (2) a race effect whose sign depends on gender",
            result["below_baseline"] == [("black", "male")]
            and (result["race_in_women"], result["race_in_men"], inter) == (0.156, -0.303, 0.459)
            and result["gender_pp"] == 2.25,
            f"only {result['below_baseline']} below white men; race gap {result['race_in_women']:+} "
            f"among women, {result['race_in_men']:+} among men (interaction {inter:+}); pooled "
            f"gender gap in hiring probability {result['gender_pp']:+} points",
        ),
        practice.Check(
            "FINDING: single-axis race evaluation sees a quarter of the penalty",
            abs(result["gender"] - POOLED["gender"]) <= 0.0005 + 1e-9
            and abs(result["race"] - POOLED["race"]) <= 0.0005 + 1e-9
            and ratio == 4.1 and result["race_pp"] == -0.55,
            f"cells give pooled gender {result['gender']:+} (paper {POOLED['gender']:+}), race "
            f"{result['race']:+} (paper {POOLED['race']:+}); Black-male gap is {ratio}x the pooled "
            f"race gap; race in hiring points {result['race_pp']:+}",
        ),
        practice.Check(
            "FINDING: the lesson states the result backwards",
            "penalizes Black women" in result["claim"]
            and result["ranking"][0] == ("black", "female")
            and result["ranking"][-1] == ("black", "male"),
            f"lesson: '{result['claim']}'; paper ranking {result['ranking']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
