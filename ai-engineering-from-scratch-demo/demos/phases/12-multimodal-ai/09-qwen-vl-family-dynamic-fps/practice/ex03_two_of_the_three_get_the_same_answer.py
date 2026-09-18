"""Exercise 3 — two of the three get the same answer.

    Pick FPS for a 30-second tennis rally vs a 30-second recipe demo vs a
    30-second UI-agent recording. Justify each with the dynamic-FPS logic.

Reading of the exercise: each clip is handed to the lesson's own `VideoPlan`
with the motion label its description implies, and the justification is then
read off what the sampler actually did rather than argued around it. The UI
recording is the interesting one, because it is the case where a rate is the
wrong kind of answer and the lesson has no other kind to give.

**ANSWER: 8 FPS for the rally, 4 for the recipe, 4 for the UI replay.** 240
frames and 19,440 tokens against 120 and 9,720 twice -- the lesson's logic
cannot separate the second from the third.

**FINDING: at 30 seconds the budget binds for none of them.** `fps_max` is
**13.48** for all three, above every ladder's top rung, so all three answers are
ladder tops. The sampler's budget arithmetic runs and changes nothing; the
motion label is the entire decision, and nothing in the lesson computes it.

**FINDING: relabelling the UI clip "high" doubles its cost and nothing else.**
8 FPS, 240 frames, 19,440 tokens -- a 2x swing from a string. The one input the
answer depends on is the one input the pipeline does not measure.

**FINDING: `frame_times` is exactly uniform, which is the wrong shape for the UI
case.** Its inter-frame gaps have a single distinct value at every rate. A UI
recording is mostly a cursor not moving: on a stated model where the interesting
events occupy 5 of the 30 seconds, uniform sampling at 4 FPS puts **20 of 120**
frames on them -- **16.7%** -- and spends the rest on a still screen. The rate
is not the lever; *where* the frames go is, and the sampler has no way to
express it.

Structure: `plan` runs the lesson's `VideoPlan` on one clip, `gaps` checks the
uniformity of `frame_times`, and `EVENT_WINDOW` is the stated model for how much
of a UI recording is worth a frame.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "09-qwen-vl-family-dynamic-fps"
DURATION, TOKENS_PER_FRAME, CONTEXT = 30.0, 81, 32768
CLIPS = (("tennis rally", "high"), ("recipe demo", "medium"), ("ui replay", "medium"))
EVENT_WINDOW = 5.0


def plan(ref, motion, duration=DURATION):
    video = ref.VideoPlan(duration_s=duration, tokens_per_frame=TOKENS_PER_FRAME,
                          budget=CONTEXT, motion=motion)
    return {"fps": video.fps(), "frames": len(video.frame_times()),
            "tokens": video.total_tokens(), "times": video.frame_times()}


def gaps(times):
    return sorted({round(later - earlier, 6)
                   for earlier, later in zip(times, times[1:])})


def on_events(times, window=EVENT_WINDOW, duration=DURATION):
    """Frames landing in the last `window` seconds -- the stated event model."""
    return sum(1 for moment in times if moment >= duration - window)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plans = {name: plan(ref, motion) for name, motion in CLIPS}
    ui = plans["ui replay"]
    relabelled = plan(ref, "high")
    return {
        "fps": {name: row["fps"] for name, row in plans.items()},
        "frames": {name: row["frames"] for name, row in plans.items()},
        "tokens": {name: row["tokens"] for name, row in plans.items()},
        "tied": [name for name, row in plans.items()
                 if row["fps"] == plans["recipe demo"]["fps"]],
        "fps_max": round(CONTEXT / (DURATION * TOKENS_PER_FRAME), 2),
        "tops": sorted({row["fps"] for row in plans.values()}),
        "budget_binds": any(row["fps"] > CONTEXT / (DURATION * TOKENS_PER_FRAME)
                            for row in plans.values()),
        "relabelled": {"fps": relabelled["fps"], "frames": relabelled["frames"],
                       "tokens": relabelled["tokens"]},
        "swing": round(relabelled["tokens"] / ui["tokens"], 1),
        "gaps": gaps(ui["times"]),
        "on_events": on_events(ui["times"]),
        "event_share": round(on_events(ui["times"]) / ui["frames"] * 100, 1),
        "window_share": round(EVENT_WINDOW / DURATION * 100, 1),
    }


def verify(result):
    fps, frames, tokens = result["fps"], result["frames"], result["tokens"]
    return [
        practice.Check(
            "ANSWER: 8 FPS for the rally, 4 for the recipe, 4 for the UI replay",
            all([fps == {"tennis rally": 8, "recipe demo": 4, "ui replay": 4},
                 frames == {"tennis rally": 240, "recipe demo": 120, "ui replay": 120},
                 tokens["tennis rally"] == 19_440, tokens["ui replay"] == 9_720,
                 len(result["tied"]) == 2]),
            f"the lesson's sampler returns {fps} -- {frames} frames and {tokens} tokens. "
            f"{result['tied']} get the identical answer, so the logic cannot separate a "
            "recipe demo from a UI recording",
        ),
        practice.Check(
            "FINDING: at 30 seconds the budget binds for none of them",
            all([result["fps_max"] == 13.48, not result["budget_binds"],
                 result["tops"] == [4, 8], max(result["tops"]) < result["fps_max"]]),
            f"fps_max is {result['fps_max']} for all three clips, above every ladder top "
            f"({result['tops']}), so all three answers are ladder tops. The budget "
            "arithmetic runs and changes nothing; the motion label is the whole decision, "
            "and nothing in the lesson computes it",
        ),
        practice.Check(
            "FINDING: relabelling the UI clip 'high' doubles its cost and nothing else",
            all([result["relabelled"] == {"fps": 8, "frames": 240, "tokens": 19_440},
                 result["swing"] == 2.0]),
            f"the same 30 seconds labelled high returns {result['relabelled']} -- a "
            f"{result['swing']}x swing in cost from a string. The one input the answer "
            "depends on is the one input the pipeline does not measure",
        ),
        practice.Check(
            "FINDING: frame_times is exactly uniform, the wrong shape for the UI case",
            all([result["gaps"] == [0.25], result["on_events"] == 20,
                 result["event_share"] == 16.7, result["window_share"] == 16.7]),
            f"the inter-frame gaps have one distinct value, {result['gaps']}s. On a stated "
            f"model where the interesting events occupy {EVENT_WINDOW:.0f} of the "
            f"{DURATION:.0f} seconds, uniform sampling puts {result['on_events']} of "
            f"{frames['ui replay']} frames on them -- {result['event_share']}%, exactly the "
            "time share, because uniform sampling knows nothing about the content. The rate "
            "is not the lever; where the frames go is",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
