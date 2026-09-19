"""Exercise 1 — the budget parameter cannot reach the budget.

    For a 3-minute cooking demo, pick uniform vs dynamic FPS. Justify with a
    token count.

Reading of the exercise: both samplers are run at the stated duration through
the lesson's own code, and the token counts come from Lesson 12.08's 3x-pooled
81 tokens a frame. The justification turns out not to be about motion at all --
`dynamic_sample` has a floor of one frame per second that the budget argument
cannot get under, so at three minutes the two strategies are not within an order
of magnitude of each other.

**ANSWER: uniform, by 5.6x.** Thirty-two uniform frames is **2,592** tokens. The
dynamic sampler asked for the same 32 returns **180** frames -- **14,580**
tokens, **44.5%** of a 32k context for a cooking video.

**FINDING: `total_budget` does not bound anything.** Its per-second allocation is
`min(fps_cap, max(1, round(raw)))`, so every second gets at least one frame
whatever the budget says. At 180 seconds the reachable range is **180 to 720**
frames -- 1 to 4 FPS -- and asking for 8 frames and asking for 32 both return
**180**.

**FINDING: the lesson's own demo overshoots its stated budget by 25%.** It calls
`dynamic_sample(motion, fps_cap=4, total_budget=12)` and prints "dynamic (12
frames total)"; the call returns **15**. Six of the ten seconds round their raw
allocation to 0 and are lifted to 1 by the `max`.

**FINDING: so the honest justification is the floor, not the motion.** A cooking
demo is mostly low motion, which is exactly the case the sampler cannot exploit:
the seconds that should get fewer frames are the ones the floor protects. Dynamic
sampling pays off only above 1 FPS uniform, and at 180 seconds that is already
14,580 tokens before any motion is considered.

Structure: `frames` runs each sampler at a duration, `tokens` prices a frame
count at Lesson 12.08's pooled rate, and `reachable` sweeps the budget parameter
to find the range it can actually produce.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "17-video-language-temporal-grounding"
DURATION, UNIFORM_FRAMES = 180.0, 32
TOKENS_PER_FRAME = 81          # Lesson 12.08: 384/14 = 27, pooled by 3
FPS_CAP, CONTEXT = 4, 32768
BUDGETS = (8, 32, 1000)
DEMO_MOTION = (0.1, 0.1, 0.8, 0.9, 0.9, 0.2, 0.1, 0.5, 0.9, 0.9)
DEMO_BUDGET = 12


def tokens(frame_count, per_frame=TOKENS_PER_FRAME):
    return frame_count * per_frame


def dynamic_frames(ref, seconds, budget, motion_level=0.2):
    motion = [motion_level] * int(seconds)
    return len(ref.dynamic_sample(motion, FPS_CAP, budget))


def reachable(ref, seconds):
    counts = [dynamic_frames(ref, seconds, budget) for budget in BUDGETS]
    return min(counts), max(counts), counts


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    uniform = len(ref.uniform_sample(DURATION, UNIFORM_FRAMES))
    low, high, counts = reachable(ref, DURATION)
    demo = len(ref.dynamic_sample(list(DEMO_MOTION), FPS_CAP, DEMO_BUDGET))
    return {
        "uniform_frames": uniform, "uniform_tokens": tokens(uniform),
        "dynamic_frames": counts[1], "dynamic_tokens": tokens(counts[1]),
        "ratio": round(tokens(counts[1]) / tokens(uniform), 1),
        "context_share": round(tokens(counts[1]) / CONTEXT * 100, 1),
        "by_budget": dict(zip(BUDGETS, counts)),
        "floor_frames": low, "cap_frames": high,
        "floor_tokens": tokens(low), "cap_tokens": tokens(high),
        "floor_fps": round(low / DURATION), "cap_fps": round(high / DURATION),
        "budget_insensitive": counts[0] == counts[1],
        "demo_returned": demo, "demo_asked": DEMO_BUDGET,
        "demo_overshoot_pct": round((demo / DEMO_BUDGET - 1) * 100),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: uniform, by 5.6x -- 2,592 tokens against 14,580",
            all([result["uniform_frames"] == 32, result["uniform_tokens"] == 2592,
                 result["dynamic_frames"] == 180, result["dynamic_tokens"] == 14580,
                 result["ratio"] == 5.6, result["context_share"] == 44.5]),
            f"{UNIFORM_FRAMES} uniform frames is {result['uniform_tokens']:,} tokens at "
            f"{TOKENS_PER_FRAME} a frame; the dynamic sampler asked for the same 32 returns "
            f"{result['dynamic_frames']} frames and {result['dynamic_tokens']:,} tokens -- "
            f"{result['ratio']}x, and {result['context_share']}% of a {CONTEXT:,} context "
            "for a cooking video",
        ),
        practice.Check(
            "FINDING: total_budget does not bound anything",
            all([result["by_budget"] == {8: 180, 32: 180, 1000: 720},
                 result["budget_insensitive"],
                 result["floor_fps"] == 1, result["cap_fps"] == FPS_CAP]),
            f"the per-second allocation is min(fps_cap, max(1, round(raw))), so every second "
            f"gets at least one frame whatever the budget says: {result['by_budget']} frames "
            f"at budgets {list(BUDGETS)}. The reachable range at {DURATION:.0f}s is "
            f"{result['floor_frames']} to {result['cap_frames']} frames -- "
            f"{result['floor_fps']} to {result['cap_fps']} FPS -- and two of the three "
            "budgets give the same answer",
        ),
        practice.Check(
            "FINDING: the lesson's own demo overshoots its stated budget by 25%",
            all([result["demo_returned"] == 15, result["demo_asked"] == DEMO_BUDGET,
                 result["demo_overshoot_pct"] == 25]),
            f"the demo calls dynamic_sample(motion, fps_cap=4, total_budget="
            f"{result['demo_asked']}) and prints '{result['demo_asked']} frames total'; the "
            f"call returns {result['demo_returned']}, {result['demo_overshoot_pct']}% over. "
            "Six of its ten seconds round their raw allocation to 0 and are lifted to 1",
        ),
        practice.Check(
            "FINDING: so the justification is the floor, not the motion",
            all([result["floor_tokens"] == 14_580, result["cap_tokens"] == 58_320,
                 result["floor_tokens"] > result["uniform_tokens"]]),
            f"a cooking demo is mostly low motion, which is the case the sampler cannot "
            f"exploit -- the seconds that should get fewer frames are the ones the floor "
            f"protects. Its cheapest possible output at this duration is "
            f"{result['floor_tokens']:,} tokens and its dearest {result['cap_tokens']:,}, "
            f"against uniform's {result['uniform_tokens']:,}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
