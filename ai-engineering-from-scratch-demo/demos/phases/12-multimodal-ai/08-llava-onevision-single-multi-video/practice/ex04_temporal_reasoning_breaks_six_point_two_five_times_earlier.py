"""Exercise 4 — temporal reasoning breaks 6.25x earlier.

    The paper reports video benchmarks trained on only 8 frames per sample. Does
    that generalize to 30-second videos at inference? What breaks first — the
    token budget or the temporal reasoning?

Reading of the exercise: "which breaks first" is read as a race with two
measurable thresholds -- the frame at which the budget is exhausted and the
frame at which the model is outside its training distribution -- so it can be
answered with a ratio rather than an opinion. The per-frame cost comes from the
lesson's own `plan_video`, which is where both thresholds live.

**ANSWER: temporal reasoning, by 6.25x.** At the lesson's own video plan a frame
costs 81 tokens, so a 4,096 budget holds **50** frames. Training saw **8**.
Extrapolation begins at frame 9 and the budget is not touched until frame 51.

**FINDING: "30 seconds" is not a frame count, and the answer depends on the
rate the exercise does not state.** At 1 FPS that is 30 frames and 2,430 tokens
-- 59% of the budget, and 3.75x the training length. At 2 FPS it is 60 frames
and **4,860** tokens, over budget. At 30 FPS it is 900 frames and **72,900**,
**17.8x** over. The two failure modes swap places between 1 and 2 FPS.

**FINDING: the lesson's own default is already outside the training
distribution.** `plan_video(4096)` returns **32 frames** -- 4x the 8 the paper
trained on -- chosen by a loop that tries 32 first and stops at the first fit.
The extrapolation the exercise asks about is the planner's default setting.

**FINDING: the budget is a hyperbola and training pinned one point on it.** Of
the **7** (frames, tokens-per-frame) pairs that fit 4,096, three sit at the
trained 8 frames and the best of those spends **1,352** tokens -- **33%** of the
budget. Staying inside the training distribution means leaving two thirds of the
budget unspent; spending it means leaving the distribution.

Structure: `plan` reads the lesson's own video plan, `frames_for` prices a
duration at a sampling rate, and `RATES` is the three rates compared.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "08-llava-onevision-single-multi-video"
BUDGET, TRAINED_FRAMES, SECONDS = 4096, 8, 30
RATES = (1, 2, 30)
VIDEO_SPACE = tuple((count, res, pool) for count in (32, 16, 8)
                    for res, pool in ((384, 3), (384, 2), (336, 2)))


def frames_for(rate, seconds=SECONDS):
    return rate * seconds


def fits(ref, budget=BUDGET):
    """Every (frames, per-frame) pair in the planner's own space that fits."""
    return sorted({(count, ref.per_tile_tokens(res, 14, pool))
                   for count, res, pool in VIDEO_SPACE
                   if count * ref.per_tile_tokens(res, 14, pool) <= budget})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plan = ref.plan_video(BUDGET)
    per_frame = plan["per_frame"]
    budget_limit = BUDGET // per_frame
    at_trained = [pair for pair in fits(ref) if pair[0] == TRAINED_FRAMES]
    return {
        "per_frame": per_frame, "planned_frames": plan["n_frames"],
        "planned_total": plan["total"],
        "budget_limit": budget_limit, "trained": TRAINED_FRAMES,
        "ratio": round(budget_limit / TRAINED_FRAMES, 2),
        "breaks_first": "temporal" if TRAINED_FRAMES < budget_limit else "budget",
        "by_rate": {rate: (frames_for(rate), frames_for(rate) * per_frame)
                    for rate in RATES},
        "over_budget": [rate for rate in RATES
                        if frames_for(rate) * per_frame > BUDGET],
        "worst_overrun": round(frames_for(RATES[-1]) * per_frame / BUDGET, 1),
        "planner_vs_trained": plan["n_frames"] // TRAINED_FRAMES,
        "frontier": fits(ref),
        "trained_best": max(count * per for count, per in at_trained),
        "trained_used_pct": round(max(count * per for count, per in at_trained)
                                  / BUDGET * 100),
    }


def verify(result):
    rates = result["by_rate"]
    return [
        practice.Check(
            "ANSWER: temporal reasoning, by 6.25x",
            all([result["per_frame"] == 81, result["budget_limit"] == 50,
                 result["trained"] == 8, result["ratio"] == 6.25,
                 result["breaks_first"] == "temporal"]),
            f"a frame costs {result['per_frame']} tokens at the lesson's own video plan, so "
            f"a {BUDGET:,} budget holds {result['budget_limit']} frames against the "
            f"{result['trained']} training saw. Extrapolation begins at frame "
            f"{result['trained'] + 1} and the budget is untouched until frame "
            f"{result['budget_limit'] + 1} -- a factor of {result['ratio']}",
        ),
        practice.Check(
            "FINDING: '30 seconds' is not a frame count",
            all([rates[1] == (30, 2430), rates[2] == (60, 4860),
                 rates[30] == (900, 72900), result["over_budget"] == [2, 30],
                 result["worst_overrun"] == 17.8]),
            f"{SECONDS} seconds is {rates[1][0]} frames and {rates[1][1]:,} tokens at 1 FPS "
            f"({rates[1][1] / BUDGET:.0%} of budget), {rates[2][0]} frames and "
            f"{rates[2][1]:,} at 2 FPS, {rates[30][0]} and {rates[30][1]:,} at 30 FPS -- "
            f"{result['worst_overrun']}x over. The two failure modes swap places between 1 "
            "and 2 FPS, and the exercise names a duration without a rate",
        ),
        practice.Check(
            "FINDING: the lesson's own default is already outside the training distribution",
            all([result["planned_frames"] == 32, result["planner_vs_trained"] == 4,
                 result["planned_total"] == 2592]),
            f"plan_video({BUDGET}) returns {result['planned_frames']} frames -- "
            f"{result['planner_vs_trained']}x the {result['trained']} the paper trained on, "
            f"{result['planned_total']:,} tokens -- chosen by a loop that tries "
            f"{result['planned_frames']} first and stops at the first fit. The extrapolation "
            "the exercise asks about is the planner's default",
        ),
        practice.Check(
            "FINDING: the budget is a hyperbola and training pinned one point on it",
            all([result["frontier"] == [(8, 81), (8, 144), (8, 169), (16, 81),
                                       (16, 144), (16, 169), (32, 81)],
                 result["trained_best"] == 1352, result["trained_used_pct"] == 33]),
            f"the {len(result['frontier'])} fitting configurations are {result['frontier']} as "
            f"(frames, tokens per frame) -- frames traded against detail. The best at the "
            f"trained frame count uses {result['trained_best']:,} tokens, "
            f"{result['trained_used_pct']}% of {BUDGET:,}, so staying inside the training "
            "distribution means leaving two thirds of the budget unspent",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
