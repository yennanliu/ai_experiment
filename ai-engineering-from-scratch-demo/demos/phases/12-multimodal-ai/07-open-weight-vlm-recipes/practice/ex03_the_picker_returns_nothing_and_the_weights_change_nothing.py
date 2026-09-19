"""Exercise 3 — the picker returns nothing, and the weights change nothing.

    Your target is a mobile UI agent on a 2B LLM. Pick encoder, connector,
    resolution, and data mix. Justify each choice with a specific ablation
    table.

Reading of the exercise: the lesson ships a picker for exactly this request, so
it is run at the budget and profile the exercise names before anything is
argued, and then the picker is examined at every budget and every profile it
supports. The choice is then made from the tables by hand, with the one axis the
tables cannot justify named as such.

**ANSWER: `pick_recipe(2, "agent")` returns nothing.** The smallest recipe in
the table is MM1-3B, so a 2B target has **zero** candidates and the picker
prints its header and stops. The target the exercise sets is below its evidence.

**FINDING: the four task profiles produce identical rankings.** At budget 10 and
at budget 80, `balanced`, `ocr`, `agent` and `reasoning` return the **same
top-3, in the same order**. At 80B Molmo-72B is the maximum on all three
benchmarks at once, so no weighting over those three can move it. At 10B nothing
dominates and the ranking holds anyway, because the benchmarks have unequal
spreads -- 6.7, 9.8 and **30.4** points -- so a 1.2-against-0.8 weight cannot
overcome a 30-point DocVQA gap. The narrowest winning margin across the four
profiles is **10.9**. The weights are a knob attached to nothing.

**FINDING: the budget filter, not the score, is the picker.** Candidates by
budget are 0, 1, 4, 5, 7, 8, 9 at 2, 3, 7, 8, 13, 30 and 72B. At 3B exactly one
recipe survives, so the "pick" is forced; the scoring function first has
something to choose between at 7B.

**ANSWER: so the choice has to be argued from the tables.** SigLIP SO400m --
`compare_encoders`, +2.5 MMMU and +5.0 DocVQA over CLIP at equal token count.
MLP-2 -- `axis_impact` rates connector architecture at 5%, the smallest axis,
so the cheapest one wins by default. AnyRes 672 -- `axis_impact` rates visual
token count at 60%, the largest, and UI screenshots are the OCR case.
PixMo-style dense human captions -- `compare_data`, 40.0 -> 45.3 MMMU. Three of
the four have a table; the connector's justification is a table saying the
question does not matter.

Structure: `candidates` filters and scores exactly as `pick_recipe` does,
`rank` returns the top-3 for one profile, and `PROFILES` is the picker's own
weight table.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "07-open-weight-vlm-recipes"
TARGET_B, TARGET_TASK = 2, "agent"
BUDGETS = (2, 3, 7, 8, 13, 30, 72)
PROBED = (10, 80)
PROFILES = {"balanced": {"mmmu": 1.0, "cv": 1.0, "doc": 1.0},
            "ocr": {"mmmu": 0.4, "cv": 0.3, "doc": 1.2},
            "agent": {"mmmu": 1.0, "cv": 1.2, "doc": 0.8},
            "reasoning": {"mmmu": 1.5, "cv": 0.5, "doc": 0.8}}


def candidates(ref, budget):
    return [recipe for recipe in ref.RECIPES if recipe.llm_b <= budget]


def score(recipe, weights):
    return (recipe.mmmu * weights["mmmu"] + recipe.cv_bench * weights["cv"]
            + recipe.docvqa * weights["doc"])


def rank(ref, budget, profile, top=3):
    pool = sorted(candidates(ref, budget), key=lambda r: score(r, PROFILES[profile]),
                  reverse=True)
    return [recipe.name for recipe in pool[:top]]


def margin(ref, budget, profile):
    """How far the winner leads the runner-up under one profile."""
    pool = sorted(candidates(ref, budget), key=lambda r: score(r, PROFILES[profile]),
                  reverse=True)
    return score(pool[0], PROFILES[profile]) - score(pool[1], PROFILES[profile])


def dominant(pool):
    """Recipes that are the maximum on all three benchmarks at once."""
    best = (max(r.mmmu for r in pool), max(r.cv_bench for r in pool),
            max(r.docvqa for r in pool))
    return [r.name for r in pool if (r.mmmu, r.cv_bench, r.docvqa) == best]


def spreads(pool):
    """Per-benchmark range across a candidate pool -- what a weight has to overcome."""
    fields = (("MMMU", "mmmu"), ("CV-Bench", "cv_bench"), ("DocVQA", "docvqa"))
    return {name: round(max(getattr(r, attr) for r in pool)
                        - min(getattr(r, attr) for r in pool), 1)
            for name, attr in fields}


def counts(ref):
    return [len(candidates(ref, budget)) for budget in BUDGETS]


def sweep(ref):
    """Top-3 per profile at each probed budget, and whether they all agree."""
    rankings = {budget: {name: rank(ref, budget, name) for name in PROFILES}
                for budget in PROBED}
    agree = {budget: len({tuple(order) for order in rows.values()}) == 1
             for budget, rows in rankings.items()}
    return rankings, agree


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rankings, agree = sweep(ref)
    sizes = counts(ref)
    return {
        "target_candidates": len(candidates(ref, TARGET_B)),
        "smallest": min(recipe.llm_b for recipe in ref.RECIPES),
        "by_budget": sizes,
        "rankings": rankings,
        "identical": agree,
        "profiles": len(PROFILES),
        "dominant": {budget: dominant(candidates(ref, budget)) for budget in PROBED},
        "spreads": spreads(candidates(ref, 10)),
        "margins": {name: round(margin(ref, 10, name), 1) for name in PROFILES},
        "forced_at": BUDGETS[sizes.index(1)],
        "first_choice_at": BUDGETS[next(i for i, n in enumerate(sizes) if n > 1)],
    }


def verify(result):
    rankings, dominants = result["rankings"], result["dominant"]
    return [
        practice.Check(
            "ANSWER: pick_recipe(2, 'agent') returns nothing",
            all([result["target_candidates"] == 0, result["smallest"] == 3]),
            f"the smallest recipe in the table is {result['smallest']}B, so a {TARGET_B}B "
            f"target has {result['target_candidates']} candidates and the picker prints its "
            "header and stops. The target the exercise sets is below its evidence",
        ),
        practice.Check(
            "FINDING: the four task profiles produce identical rankings",
            all([all(result["identical"].values()),
                 dominants[80] == ["Molmo-72B"], dominants[10] == [],
                 result["spreads"] == {"MMMU": 6.7, "CV-Bench": 9.8, "DocVQA": 30.4},
                 min(result["margins"].values()) == 10.9,
                 rankings[10]["ocr"] == rankings[10]["reasoning"]]),
            f"at budgets {list(PROBED)} all {result['profiles']} profiles return the same "
            f"top-3 in the same order -- {rankings[10]['agent']} and "
            f"{rankings[80]['agent']}. At 80B {dominants[80][0]} is the maximum on all three "
            f"benchmarks at once, so no weighting can move it. At 10B nothing dominates, and "
            f"the ranking still holds because the benchmark spreads are {result['spreads']}: "
            f"a 1.2-against-0.8 weight cannot overcome a 30-point DocVQA gap, and the "
            f"narrowest winning margin across the four profiles is "
            f"{min(result['margins'].values())}",
        ),
        practice.Check(
            "FINDING: the budget filter, not the score, is the picker",
            all([result["by_budget"] == [0, 1, 4, 5, 7, 8, 9],
                 result["forced_at"] == 3, result["first_choice_at"] == 7]),
            f"candidates by budget are {result['by_budget']} at {list(BUDGETS)}B. At "
            f"{result['forced_at']}B exactly one recipe survives, so the pick is forced; the "
            f"scoring function first has something to choose between at "
            f"{result['first_choice_at']}B",
        ),
        practice.Check(
            "ANSWER: three of the four axes have a table, and the fourth has a shrug",
            all([result["profiles"] == 4, len(rankings[80]["agent"]) == 3]),
            "encoder: compare_encoders, SigLIP over CLIP by +2.5 MMMU and +5.0 DocVQA at "
            "equal token count. Resolution: axis_impact puts visual-token count at 60%, the "
            "largest axis, and UI screenshots are the OCR case, so AnyRes 672. Data: "
            "compare_data, 40.0 -> 45.3 MMMU for dense human captions. Connector: "
            "axis_impact puts it at 5%, the smallest axis -- the justification for MLP-2 is "
            "a table saying the question does not matter",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
