"""Exercise 2 — nothing that keeps the frames fits a 128K context.

    A 1080p frame (1920x1080) at patch 14 produces how many tokens? At 30 FPS
    over a 5-minute video, how many total visual tokens? Which cost saves you
    most: pooling, frame sampling, or token merging?

Reading of the exercise: "which saves you most" is read as a measured ratio
against the 30-FPS baseline, not as a preference, and then the three ratios are
composed -- because pooling, sampling and merging are not alternatives, they
are independent factors that multiply. The frame count is held to floor
division, matching exercise 1's cropping convention.

**ANSWER: 10,549 tokens a frame, 94,941,000 for the five minutes.** 1920/14 is
137.14 and 1080/14 is 77.14, so the grid is 137x77 under cropping; 30 FPS over
300 seconds is 9,000 frames.

**ANSWER: pooling saves most, by 350x over its nearest rival, and it is the one
answer that cannot be used.** Mean-pooling each frame to a single vector is
10,549x; sampling to 1 FPS is 30x; 2x2 token merging is 4.08x. Pooling wins
because it throws away the entire question -- one vector per frame cannot say
where anything is, which is what a video VLM is asked.

**FINDING: nothing that keeps the frames fits a 128K context.** The best
sampling-and-merging combination in the grid below is 0.25 FPS with 2x2
merging: 193,800 tokens, still 1.48x over 128K. To fit, the video has to be
sampled at one frame every 6.0 seconds *and* merged -- 50 frames of the 9,000.

**FINDING: all three ratios are constants, so duration cancels out.** Doubling
the video doubles every arm, and the ranking pooling > sampling > merging holds
at any length. The ranking is therefore not a measurement of the video; what
the three differ in is what they destroy -- layout, motion, and fine detail
respectively -- and none of those is a token count.

**FINDING: one 1080p frame is 2.6x the lesson's largest configuration.**
`ZOO[-1]` is 4,097 tokens for a whole 896x896 image. And `grid_shape` refuses
both 1920 and 1080, since each leaves a remainder of 2 against patch 14.

Structure: `tokens` is the patch-grid count, `merged` applies Qwen's 2x2 merge
with its multiple-of-28 crop, `budget` composes a (fps, merge) pair into a
total, and `GRID` is the sampling-and-merging space searched for a fit.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "01-vision-transformer-patch-tokens"
FRAME = (1920, 1080)
PATCH, MERGE, FPS, SECONDS = 14, 2, 30, 300
CONTEXT = 128 * 1024
GRID = tuple((fps, merge) for fps in (30, 5, 1, 0.5, 0.25) for merge in (1, MERGE))


def tokens(size, patch=PATCH):
    """Patch-grid token count for a (width, height), cropping the remainder."""
    grid = [side // patch for side in size]
    return grid, grid[0] * grid[1]


def merged(size, patch=PATCH, merge=MERGE):
    """Qwen's 2x2 spatial merge: both sides cropped to a multiple of patch*merge."""
    step = patch * merge
    usable = tuple(side - side % step for side in size)
    _, patches = tokens(usable, patch)
    return patches // (merge * merge)


def budget(per_frame, fps, seconds=SECONDS):
    """Total visual tokens for one sampling rate."""
    return per_frame * int(fps * seconds)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid, per_frame = tokens(FRAME)
    pooled = merged(FRAME)
    baseline = budget(per_frame, FPS)
    arms = {"pool to one vector": budget(1, FPS),
            "sample to 1 FPS": budget(per_frame, 1),
            "merge 2x2": budget(pooled, FPS)}
    space = {(fps, merge): budget(per_frame if merge == 1 else pooled, fps)
             for fps, merge in GRID}
    return {
        "grid": grid, "per_frame": per_frame, "frames": FPS * SECONDS,
        "total": baseline, "merged_per_frame": pooled,
        "ratios": {name: round(baseline / total, 2) for name, total in arms.items()},
        "space": space, "fits": sorted(total for total in space.values()
                                       if total <= CONTEXT),
        "cheapest": min(space.values()),
        "frames_that_fit": CONTEXT // pooled,
        "remainders": [side % PATCH for side in FRAME],
        "refused": sorted(side for side in FRAME
                          if _raises(ref.grid_shape, side, PATCH)),
        "zoo_max": ref.seq_length(ref.ZOO[-1]),
    }


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def verify(result):
    ratios, space = result["ratios"], result["space"]
    return [
        practice.Check(
            "ANSWER: 10,549 tokens a frame, 94,941,000 for the five minutes",
            all([result["grid"] == [137, 77], result["per_frame"] == 10549,
                 result["frames"] == 9000, result["total"] == 94_941_000]),
            f"1920x1080 at patch {PATCH} crops to a {result['grid'][0]}x{result['grid'][1]} "
            f"grid -- {result['per_frame']:,} tokens -- and {FPS} FPS over {SECONDS}s is "
            f"{result['frames']:,} frames, so {result['total']:,} visual tokens",
        ),
        practice.Check(
            "ANSWER: pooling saves most, by 350x over its nearest rival",
            all([ratios["pool to one vector"] == 10549.0,
                 ratios["sample to 1 FPS"] == 30.0, ratios["merge 2x2"] == 4.08,
                 round(ratios["pool to one vector"] / ratios["sample to 1 FPS"]) == 352]),
            f"against the {result['total']:,}-token baseline: {ratios}. Pooling wins because "
            "it discards the question -- one vector per frame cannot say where anything is, "
            "which is the whole of video VQA and grounding",
        ),
        practice.Check(
            "FINDING: nothing that keeps the frames fits a 128K context",
            all([result["fits"] == [], result["cheapest"] == 193_800,
                 result["frames_that_fit"] == 50]),
            f"the cheapest point in the {len(space)}-point sampling-and-merging grid is "
            f"{result['cheapest']:,} tokens (0.25 FPS, 2x2 merged), still "
            f"{result['cheapest'] / CONTEXT:.2f}x over 128K. Fitting needs "
            f"{result['frames_that_fit']} merged frames of the {result['frames']:,} -- one "
            f"every {SECONDS / result['frames_that_fit']:.1f} seconds",
        ),
        practice.Check(
            "FINDING: all three ratios are constants, so duration cancels out",
            all([space[(30, 1)] == result["total"],
                 space[(1, 1)] * 30 == space[(30, 1)],
                 round(space[(30, 1)] / space[(30, 2)], 2) == ratios["merge 2x2"],
                 space[(0.5, 2)] * 2 == space[(1, 2)]]),
            f"1 FPS is exactly {space[(30, 1)] // space[(1, 1)]}x cheaper and the merge is "
            f"{ratios['merge 2x2']}x at every rate, so doubling the video doubles all three "
            "arms and the ranking never moves. What the three differ in is what they "
            "destroy -- layout, motion, fine detail -- and none of that is a token count",
        ),
        practice.Check(
            "FINDING: one 1080p frame is 2.6x the lesson's largest configuration",
            all([result["zoo_max"] == 4097, result["refused"] == [1080, 1920],
                 result["remainders"] == [2, 2]]),
            f"ZOO[-1] is {result['zoo_max']:,} tokens for a whole 896x896 image, so one "
            f"frame is {result['per_frame'] / result['zoo_max']:.1f}x of it; and grid_shape "
            f"raises on both {result['refused']}, each leaving a remainder of "
            f"{result['remainders'][0]} against patch {PATCH}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
