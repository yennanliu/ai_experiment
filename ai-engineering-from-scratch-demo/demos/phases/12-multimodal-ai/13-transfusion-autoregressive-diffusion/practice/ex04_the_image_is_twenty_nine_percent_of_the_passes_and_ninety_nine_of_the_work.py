"""Exercise 4 — the image is 29% of the passes and 99% of the work.

    Generation: given a text prompt, the model runs NTP for 50 tokens, then hits
    `<image>`, then runs diffusion on 256 patches over 20 denoise steps. How
    many forward passes total?

Reading of the exercise: the count is asked for in forward passes, so it is
given in forward passes -- and then in positions processed, because the two
answers disagree by two orders of magnitude and only one of them predicts
latency. The comparison arm is the same 256 patches generated autoregressively,
which is what Lessons 12.11 and 12.12 cost out.

**ANSWER: 70 forward passes.** 50 for the text, one per token with a KV cache,
and **20** for the image -- one per denoise step, not one per patch, because the
image block is bidirectional and all 256 patches are denoised together.

**FINDING: the image is 29% of the passes and 99% of the positions.** A text
pass processes one new position; a denoise pass processes all **256**. That is
50 positions of text against **5,120** of image -- **99.0%** of the position-work
in 28.6% of the calls.

**FINDING: against an autoregressive arm, Transfusion does 20x the work in 12.8x
fewer passes.** Generating the same 256 patches one token at a time is 256
passes and 256 positions. Transfusion is **20** passes and **5,120** positions.
The trade is parallelism, not total compute -- and it is exactly the denoise-step
count in both directions.

**FINDING: the breakeven is 256 steps, and the shipped 20 is 12.8 patches per
step.** Below 256 denoise steps Transfusion wins on forward passes; at 256 it
matches the autoregressive arm on both counts; above it loses on both. The knob
that decides whether this architecture is cheaper than Chameleon's is the one
the exercise treats as a given.

Structure: `passes` and `positions` count the two arms, `autoregressive` is the
same patch count generated one token at a time, and `breakeven` solves for the
step count at which the two cost the same.
"""

from __future__ import annotations

from harness import practice

TEXT_TOKENS, PATCHES, DENOISE_STEPS = 50, 256, 20


def passes(text=TEXT_TOKENS, steps=DENOISE_STEPS):
    return {"text": text, "image": steps, "total": text + steps}


def positions(text=TEXT_TOKENS, patches=PATCHES, steps=DENOISE_STEPS):
    return {"text": text, "image": patches * steps, "total": text + patches * steps}


def autoregressive(patches=PATCHES):
    """The same patch count generated one token at a time, with a KV cache."""
    return {"passes": patches, "positions": patches}


def breakeven(patches=PATCHES, steps=DENOISE_STEPS):
    return round(patches / steps, 1)


def solve():
    call, cell = passes(), positions()
    arm = autoregressive()
    return {
        "passes": call, "positions": cell,
        "image_pass_share": round(call["image"] / call["total"] * 100, 1),
        "image_position_share": round(cell["image"] / cell["total"] * 100, 1),
        "per_denoise_pass": PATCHES,
        "ar_passes": arm["passes"], "ar_positions": arm["positions"],
        "pass_advantage": round(arm["passes"] / call["image"], 1),
        "work_penalty": cell["image"] // arm["positions"],
        "breakeven_patches_per_step": breakeven(),
        "steps": DENOISE_STEPS,
        "loses_both_above": PATCHES,
    }


def verify(result):
    call, cell = result["passes"], result["positions"]
    return [
        practice.Check(
            "ANSWER: 70 forward passes -- 50 text, 20 denoise",
            all([call == {"text": 50, "image": 20, "total": 70},
                 result["per_denoise_pass"] == PATCHES]),
            f"{call['text']} passes for the text, one per token with a KV cache, and "
            f"{call['image']} for the image -- one per denoise step, not one per patch, "
            f"because the image block is bidirectional and all {result['per_denoise_pass']} "
            f"patches are denoised together. {call['total']} in total",
        ),
        practice.Check(
            "FINDING: the image is 29% of the passes and 99% of the positions",
            all([cell == {"text": 50, "image": 5120, "total": 5170},
                 result["image_pass_share"] == 28.6,
                 result["image_position_share"] == 99.0]),
            f"a text pass processes one new position and a denoise pass processes all "
            f"{PATCHES}: {cell['text']} positions of text against {cell['image']:,} of image. "
            f"That is {result['image_position_share']}% of the work in "
            f"{result['image_pass_share']}% of the calls, so counting passes predicts "
            "latency backwards",
        ),
        practice.Check(
            "FINDING: 20x the work in 12.8x fewer passes",
            all([result["ar_passes"] == 256, result["ar_positions"] == 256,
                 result["pass_advantage"] == 12.8,
                 result["work_penalty"] == DENOISE_STEPS]),
            f"generating the same {PATCHES} patches one token at a time is "
            f"{result['ar_passes']} passes and {result['ar_positions']} positions; "
            f"Transfusion is {call['image']} passes and {cell['image']:,} positions -- "
            f"{result['pass_advantage']}x fewer calls for {result['work_penalty']}x the "
            "position-work. The trade is parallelism, not total compute",
        ),
        practice.Check(
            "FINDING: the breakeven is at one denoise step per 12.8 patches",
            all([result["breakeven_patches_per_step"] == 12.8,
                 result["steps"] < result["loses_both_above"]]),
            f"below {result['loses_both_above']} steps Transfusion wins on passes; at "
            f"{result['loses_both_above']} it matches the autoregressive arm on both and "
            f"above it loses on both. The shipped {result['steps']} steps put it at "
            f"{result['breakeven_patches_per_step']} patches per step -- and that step count, "
            "which the exercise treats as a given, is what decides whether the architecture "
            "is cheaper than Chameleon's at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
