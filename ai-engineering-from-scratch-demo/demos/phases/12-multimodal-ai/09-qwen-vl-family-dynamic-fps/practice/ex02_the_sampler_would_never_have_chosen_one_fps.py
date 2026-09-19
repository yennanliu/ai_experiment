"""Exercise 2 — the sampler would never have chosen one FPS.

    A 10-minute security-camera recording at 1 FPS produces how many frames? At
    384 resolution with 3x pool, how many total tokens? Does Qwen2.5-VL's
    default 32k context handle it?

Reading of the exercise: the arithmetic is done at the rate the exercise names,
and then the same clip is handed to the lesson's own `VideoPlan` -- because the
lesson ships a sampler whose entire purpose is to refuse the configuration the
exercise is asking about, and what it picks instead is the more useful answer.

**ANSWER: 600 frames, 48,600 tokens, and no -- 1.48x a 32,768 context.**
384/14 is 27, pooled by 3 gives a 9x9 grid, so 81 tokens a frame.

**FINDING: the lesson's own sampler never offers 1 FPS here.** Its `fps_max` for
this clip is **0.674**, so the low-motion ladder returns **0.5**: 300 frames,
**24,300** tokens, 74.2% of the context. The exercise poses a configuration the
lesson's logic rejects one line before it is asked about.

**FINDING: the ladder discards a quarter of the budget it just computed.** The
sampler takes the first candidate at or below `fps_max` from a fixed ladder, and
the gap between 0.674 and 0.5 is thrown away -- 26% of the budget. Across the
lesson's four demo clips the utilisation is 59.3%, 29.7%, 74.2% and 59.3%: never
above three quarters, and never the same twice.

**FINDING: for short clips the budget does not bind at all.** The ladder's top
rung binds only beyond **50.6s** at high motion, **101.1s** at medium and
**404.5s** at low. A 30-second clip gets its ladder's maximum whatever the
budget is, so "dynamic FPS" is, below a minute, a lookup on the motion label.

Structure: `frame_tokens` reproduces the lesson's pooling arithmetic, `plan`
runs its `VideoPlan` on one clip, and `binds_beyond` solves for the duration at
which each ladder's top rung stops fitting.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "09-qwen-vl-family-dynamic-fps"
RESOLUTION, PATCH, POOL = 384, 14, 3
DURATION, ASKED_FPS, CONTEXT = 600.0, 1.0, 32768
LADDER_TOPS = {"high": 8, "medium": 4, "low": 1}
DEMOS = (("tennis 30s", 30.0, "high"), ("recipe 30s", 30.0, "medium"),
         ("security 600s", 600.0, "low"), ("ui replay 60s", 60.0, "medium"))


def frame_tokens():
    """The lesson's pooling arithmetic: 384/14 = 27, pooled by 3, squared."""
    return ((RESOLUTION // PATCH) // POOL) ** 2


def plan(ref, duration, motion, tokens):
    video = ref.VideoPlan(duration_s=duration, tokens_per_frame=tokens,
                          budget=CONTEXT, motion=motion)
    return {"fps": video.fps(), "frames": len(video.frame_times()),
            "tokens": video.total_tokens(),
            "used_pct": round(video.total_tokens() / CONTEXT * 100, 1)}


def binds_beyond(top, tokens, budget=CONTEXT):
    return round(budget / (top * tokens), 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tokens = frame_tokens()
    asked_frames = int(DURATION * ASKED_FPS)
    chosen = plan(ref, DURATION, "low", tokens)
    return {
        "grid": (RESOLUTION // PATCH) // POOL, "tokens_per_frame": tokens,
        "asked_frames": asked_frames, "asked_tokens": asked_frames * tokens,
        "context": CONTEXT,
        "overrun": round(asked_frames * tokens / CONTEXT, 2),
        "fits": asked_frames * tokens <= CONTEXT,
        "fps_max": round(CONTEXT / (DURATION * tokens), 3),
        "chosen": chosen,
        "wasted_pct": round((1 - chosen["fps"] / (CONTEXT / (DURATION * tokens))) * 100),
        "demos": {name: plan(ref, duration, motion, tokens)["used_pct"]
                  for name, duration, motion in DEMOS},
        "binds": {motion: binds_beyond(top, tokens) for motion, top in LADDER_TOPS.items()},
    }


def verify(result):
    chosen, demos = result["chosen"], result["demos"]
    return [
        practice.Check(
            "ANSWER: 600 frames, 48,600 tokens, and no -- 1.48x a 32,768 context",
            all([result["asked_frames"] == 600, result["tokens_per_frame"] == 81,
                 result["grid"] == 9, result["asked_tokens"] == 48_600,
                 result["overrun"] == 1.48, not result["fits"]]),
            f"{RESOLUTION}/{PATCH} is 27, pooled by {POOL} gives a {result['grid']}x"
            f"{result['grid']} grid and {result['tokens_per_frame']} tokens a frame; "
            f"{result['asked_frames']} frames is {result['asked_tokens']:,} tokens, "
            f"{result['overrun']}x a {CONTEXT:,} context",
        ),
        practice.Check(
            "FINDING: the lesson's own sampler never offers 1 FPS here",
            all([result["fps_max"] == 0.674, chosen["fps"] == 0.5,
                 chosen["frames"] == 300, chosen["tokens"] == 24_300,
                 chosen["used_pct"] == 74.2, chosen["fps"] < ASKED_FPS]),
            f"VideoPlan computes fps_max = {result['fps_max']} for this clip, so the "
            f"low-motion ladder returns {chosen['fps']}: {chosen['frames']} frames, "
            f"{chosen['tokens']:,} tokens, {chosen['used_pct']}% of the context. The "
            f"configuration the exercise asks about is one the lesson's own logic rejects",
        ),
        practice.Check(
            "FINDING: the ladder discards a quarter of the budget it just computed",
            all([result["wasted_pct"] == 26,
                 demos == {"tennis 30s": 59.3, "recipe 30s": 29.7,
                           "security 600s": 74.2, "ui replay 60s": 59.3},
                 max(demos.values()) < 75]),
            f"the sampler takes the first ladder rung at or below fps_max, so the gap "
            f"between {result['fps_max']} and {chosen['fps']} -- {result['wasted_pct']}% of "
            f"the budget -- is discarded. Across the lesson's four demo clips the "
            f"utilisation is {demos}: never above three quarters, never the same twice",
        ),
        practice.Check(
            "FINDING: for short clips the budget does not bind at all",
            all([result["binds"] == {"high": 50.6, "medium": 101.1, "low": 404.5},
                 min(result["binds"].values()) > 30]),
            f"the top rung of each ladder stops fitting only beyond {result['binds']} "
            f"seconds. A 30-second clip gets its ladder's maximum whatever the budget is, so "
            "below a minute 'dynamic FPS' is a lookup on the motion label",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
