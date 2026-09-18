"""Exercise 1 — the saving the exercise asks about does not exist.

    Your product supports 80% single-image, 10% multi-image (2-4 images), 10%
    video (8-16 frames). Design the token budget. Where would you put the extra
    budget you save from not doing heavy multi-image?

Reading of the exercise: the budget is designed by running the lesson's own
three planners and then weighting their answers by the stated mix, so the
"design" is a measured expected cost rather than a proposal. The second question
is checked before it is answered, because it presumes multi-image is the
expensive scenario and the planner says otherwise.

**ANSWER: 1,746 tokens per sample expected, against a 4,096 budget -- 57.4%
unused.** 0.8 x 1,690 + 0.1 x 1,352 + 0.1 x 2,592. There is nowhere to put a
saving because the budget was never spent.

**ANSWER: and there is no saving from skipping heavy multi-image.** The
planner's multi-image configuration is **1,352** tokens, the *cheapest* of the
three -- below single-image's 1,690 and half of video's 2,592. Dropping it moves
the expected cost by +39 tokens, because the 10% it frees is reweighted onto
scenarios that cost more.

**FINDING: the planner maximises tiles, not tokens.** Its loops run tile count
outer and resolution-and-pooling inner, so it returns the first fit rather than
the best: **9 pooled tiles at 1,690** when **4 unpooled tiles at 3,645** also
fit. It uses **41.3%** of the budget where **89.0%** was available, and the same
inversion costs multi-image 33.0% against 71.2%.

**FINDING: the lesson misses its own spread target by 0.3 points.** The three
plans span 2,592 - 1,352 = **1,240** tokens, **30.3%** of the budget, directly
under a line that reads "keep the spread under 30% for predictable LLM cost".

**ANSWER: so the budget goes where the mix is -- single-image.** At 80% of
samples, moving single-image from 1,690 to the 3,645 the planner passed over
costs 1,564 tokens per sample on average and still lands at 3,310 of 4,096. The
same move on video would be weighted by 0.1.

Structure: `plans` runs the lesson's three planners, `space` enumerates what
each planner could have returned, and `expected` weights the result by the
product mix.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "08-llava-onevision-single-multi-video"
BUDGET = 4096
MIX = {"single": 0.8, "multi": 0.1, "video": 0.1}
SPREAD_TARGET = 30.0
SINGLE_SPACE = tuple((tiles, res, pool) for tiles in (9, 4, 1)
                     for res, pool in ((384, 2), (384, 1), (336, 2)))
MULTI_SPACE = tuple((count, res, pool) for count in (8, 6, 4, 2)
                    for res, pool in ((384, 2), (384, 1), (336, 2)))
VIDEO_SPACE = tuple((count, res, pool) for count in (32, 16, 8)
                    for res, pool in ((384, 3), (384, 2), (336, 2)))


def plans(ref, budget=BUDGET):
    return {"single": ref.plan_single_image(budget)["total"],
            "multi": ref.plan_multi_image(budget)["total"],
            "video": ref.plan_video(budget)["total"]}


def best_fit(ref, space, thumbnail, budget=BUDGET):
    """The largest configuration in a planner's own search space that still fits."""
    options = [(count + thumbnail) * ref.per_tile_tokens(res, 14, pool)
               for count, res, pool in space]
    return max(total for total in options if total <= budget)


def expected(costs, mix=MIX):
    return round(sum(costs[name] * share for name, share in mix.items()), 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    costs = plans(ref)
    best = {"single": best_fit(ref, SINGLE_SPACE, 1),
            "multi": best_fit(ref, MULTI_SPACE, 0),
            "video": best_fit(ref, VIDEO_SPACE, 0)}
    without_multi = {"single": costs["single"], "video": costs["video"]}
    spread = max(costs.values()) - min(costs.values())
    return {
        "costs": costs, "expected": expected(costs),
        "headroom_pct": round((1 - expected(costs) / BUDGET) * 100, 1),
        "cheapest": min(costs, key=costs.get), "dearest": max(costs, key=costs.get),
        "drop_multi": round(expected(without_multi, {"single": 8 / 9, "video": 1 / 9})
                            - expected(costs), 1),
        "best": best,
        "used_pct": {name: round(total / BUDGET * 100, 1) for name, total in costs.items()},
        "available_pct": {name: round(total / BUDGET * 100, 1)
                          for name, total in best.items()},
        "spread": spread, "spread_pct": round(spread / BUDGET * 100, 1),
        "spread_target": SPREAD_TARGET,
        "upgraded": expected({**costs, "single": best["single"]}),
    }


def verify(result):
    costs, used, available = result["costs"], result["used_pct"], result["available_pct"]
    return [
        practice.Check(
            "ANSWER: 1,746 tokens per sample expected, 57.4% of the budget unused",
            all([costs == {"single": 1690, "multi": 1352, "video": 2592},
                 result["expected"] == 1746.4, result["headroom_pct"] == 57.4]),
            f"the lesson's three planners return {costs} at a {BUDGET:,}-token budget, and "
            f"the {MIX} mix weights them to {result['expected']} per sample -- "
            f"{result['headroom_pct']}% of the budget never spent",
        ),
        practice.Check(
            "ANSWER: there is no saving from skipping heavy multi-image",
            all([result["cheapest"] == "multi", result["dearest"] == "video",
                 result["drop_multi"] == 43.8]),
            f"multi-image is the cheapest of the three at {costs['multi']:,} -- below "
            f"single-image's {costs['single']:,} and half of video's {costs['video']:,}. "
            f"Dropping it moves the expected cost by {result['drop_multi']:+}, because the "
            "10% it frees is reweighted onto scenarios that cost more",
        ),
        practice.Check(
            "FINDING: the planner maximises tiles, not tokens",
            all([result["best"]["single"] == 3645, used["single"] == 41.3,
                 available["single"] == 89.0, used["multi"] == 33.0,
                 available["multi"] == 71.2]),
            f"the loops run count outer and resolution-and-pooling inner, so each planner "
            f"returns the first fit rather than the best: {used} of the budget where {available}"
            f" was available inside its own search space. Nine pooled tiles at "
            f"{costs['single']:,} against four unpooled at {result['best']['single']:,}",
        ),
        practice.Check(
            "FINDING: the lesson misses its own spread target by 0.3 points",
            all([result["spread"] == 1240, result["spread_pct"] == 30.3,
                 result["spread_pct"] > SPREAD_TARGET]),
            f"the three plans span {costs['video']:,} - {costs['multi']:,} = "
            f"{result['spread']:,} tokens, {result['spread_pct']}% of the budget, printed "
            f"directly above a line that reads 'keep the spread under "
            f"{SPREAD_TARGET:.0f}% for predictable LLM cost'",
        ),
        practice.Check(
            "ANSWER: so the budget goes where the mix is -- single-image",
            all([result["upgraded"] == 3310.4, result["upgraded"] < BUDGET,
                 MIX["single"] == 0.8]),
            f"at {MIX['single']:.0%} of samples, moving single-image to the "
            f"{result['best']['single']:,} the planner passed over takes the expected cost "
            f"from {result['expected']} to {result['upgraded']} -- still inside "
            f"{BUDGET:,}. The same move on video would be weighted by {MIX['video']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
