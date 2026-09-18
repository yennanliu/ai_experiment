"""Exercise 5 — both of them fit, so the question starts at 227 frames.

    For a 10-minute video at 1 FPS sampled to 60 frames, compute the per-frame
    token cost at (Q-Former → 32 tokens/frame) vs (MLP projector → 576
    tokens/frame). Which fits into a 128k-token LLM context window?

Reading of the exercise: the arithmetic is done first and then the question is
checked, because "which fits" presumes one of them does not. The exercise's own
sampling step is kept explicit -- 10 minutes at 1 FPS is 600 frames, and 60 is
a further 10x decimation the exercise applies before either bridge is chosen --
so the sampling and the bridge can be priced separately.

**ANSWER: both fit, with room to spare.** 60 frames is **1,920** tokens through
the Q-Former (**1.5%** of 128k) and **34,560** through the MLP projector
(**26.4%**). The question the exercise asks does not discriminate between the
two configurations it names.

**FINDING: the decimation the exercise performs is larger than the decision it
asks about -- and not by much.** Unsampled, 600 frames is 19,200 tokens through
the Q-Former and **345,600** through the projector, **2.64x** over the window.
So it is the 10x frame drop, applied before the question, that makes the
projector viable; the bridge itself is 18x.

**FINDING: the question begins at 227 frames.** 128k / 576 = **227** frames for
the projector and **4,096** for the Q-Former. At 1 FPS that is **3m47s** of
video against **68m16s** -- an 18.04x span, which is 576/32 plus the floor at
227, because both arms are linear in frames and the ratio never moves.

**FINDING: 576 is not this lesson's own projector.** The lesson states a ViT
producing **256** patch tokens; 576 is LLaVA's 24x24 grid, from Lesson 12.05. At
256 the projector caps the video at **512** frames, 8m32s -- so the exercise's
two numbers come from two different vision towers.

Structure: `tokens` prices one arm, `capacity` inverts it into a frame cap and
a duration, and `ARMS` is the three per-frame costs compared.
"""

from __future__ import annotations

from harness import practice

CONTEXT = 128 * 1024
MINUTES, FPS, SAMPLED = 10, 1, 60
ARMS = {"q-former": 32, "mlp @ 256": 256, "mlp @ 576": 576}


def tokens(frames, per_frame):
    return frames * per_frame


def capacity(per_frame, context=CONTEXT, fps=FPS):
    """Frames that fit, and the video duration they represent."""
    frames = context // per_frame
    seconds = frames / fps
    return frames, f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def solve():
    full = MINUTES * 60 * FPS
    sampled = {name: tokens(SAMPLED, cost) for name, cost in ARMS.items()}
    unsampled = {name: tokens(full, cost) for name, cost in ARMS.items()}
    caps = {name: capacity(cost) for name, cost in ARMS.items()}
    return {
        "full_frames": full, "sampled_frames": SAMPLED,
        "decimation": full // SAMPLED,
        "sampled": sampled, "unsampled": unsampled,
        "share": {name: round(count / CONTEXT * 100, 1) for name, count in sampled.items()},
        "fits_sampled": [name for name, count in sampled.items() if count <= CONTEXT],
        "fits_unsampled": [name for name, count in unsampled.items() if count <= CONTEXT],
        "overflow": round(unsampled["mlp @ 576"] / CONTEXT, 2),
        "caps": caps,
        "bridge_ratio": ARMS["mlp @ 576"] // ARMS["q-former"],
        "cap_ratio": round(caps["q-former"][0] / caps["mlp @ 576"][0], 2),
    }


def verify(result):
    sampled, caps, share = result["sampled"], result["caps"], result["share"]
    return [
        practice.Check(
            "ANSWER: both fit, with room to spare",
            all([sampled["q-former"] == 1920, sampled["mlp @ 576"] == 34560,
                 share["q-former"] == 1.5, share["mlp @ 576"] == 26.4,
                 len(result["fits_sampled"]) == len(ARMS)]),
            f"{SAMPLED} frames is {sampled['q-former']:,} tokens through the Q-Former "
            f"({share['q-former']}% of 128k) and {sampled['mlp @ 576']:,} through the "
            f"projector ({share['mlp @ 576']}%). All {len(result['fits_sampled'])} arms fit, "
            "so the question the exercise asks does not discriminate between the two "
            "configurations it names",
        ),
        practice.Check(
            "FINDING: the decimation happens before the question",
            all([result["full_frames"] == 600, result["decimation"] == 10,
                 result["unsampled"]["mlp @ 576"] == 345_600,
                 result["overflow"] == 2.64,
                 result["fits_unsampled"] == ["q-former"]]),
            f"10 minutes at {FPS} FPS is {result['full_frames']} frames, so '60 frames' is a "
            f"{result['decimation']}x decimation applied before either bridge is chosen. "
            f"Unsampled, the projector needs {result['unsampled']['mlp @ 576']:,} tokens -- "
            f"{result['overflow']}x the window -- and only {result['fits_unsampled']} "
            f"survives. The frame drop is {result['decimation']}x; the bridge is "
            f"{result['bridge_ratio']}x",
        ),
        practice.Check(
            "FINDING: the question begins at 227 frames",
            all([caps["mlp @ 576"] == (227, "3m47s"),
                 caps["q-former"] == (4096, "68m16s"),
                 result["cap_ratio"] == 18.04,
                 round(result["cap_ratio"]) == result["bridge_ratio"]]),
            f"128k / 576 is {caps['mlp @ 576'][0]} frames ({caps['mlp @ 576'][1]} at "
            f"{FPS} FPS) against {caps['q-former'][0]:,} for the Q-Former "
            f"({caps['q-former'][1]}). The span is {result['cap_ratio']}x -- "
            f"{ARMS['mlp @ 576']}/{ARMS['q-former']} = {result['bridge_ratio']} plus the "
            "floor at 227 -- because both arms are linear in frames and the ratio never "
            "moves",
        ),
        practice.Check(
            "FINDING: 576 is not this lesson's own projector",
            all([caps["mlp @ 256"] == (512, "8m32s"),
                 result["sampled"]["mlp @ 256"] == 15_360]),
            f"the lesson states a ViT producing 256 patch tokens; 576 is LLaVA's 24x24 grid "
            f"from Lesson 12.05. At 256 the projector caps the video at "
            f"{caps['mlp @ 256'][0]} frames, {caps['mlp @ 256'][1]}, and the sampled clip "
            f"costs {result['sampled']['mlp @ 256']:,} tokens -- so the exercise's two "
            "numbers come from two different vision towers",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
