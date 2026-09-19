"""Exercise 1 — the lesson balances magnitudes; the mix balances nothing.

    A Transfusion-style model trains 70% text tokens and 30% image patches. The
    image diffusion loss is ~10x the text NTP loss in magnitude. What loss
    weights balance them?

Reading of the exercise: "balance" has two readings and they give different
numbers, so both are computed and the lesson's own shipped weights are placed
against them. The lesson's `train` hard-codes `text_w = 1.0, img_w = 0.1`, which
answers one of the two readings exactly and the other not at all.

**ANSWER: 0.1 balances the magnitudes; 0.2333 balances the contributions.**
Per token, `w_img / w_text = 1/10` cancels the loss-magnitude gap. Weighted by
the 70/30 mix, the image term then contributes `0.3 x 0.1 x 10 = 0.3` against
text's `0.7 x 1.0 x 1 = 0.7` -- so text carries **70%** of the gradient.
Equalising the contributions needs `0.7 / (0.3 x 10) = **0.2333**`.

**FINDING: the lesson ships the first answer and never asks the second.**
`train` sets `img_w = 0.1` exactly, and `two_loss_step` computes
`text_w * text_loss + img_w * img_loss` with no token-count term anywhere -- so
the 70/30 mix the exercise is built on cannot enter the calculation at all.

**FINDING: neither loss in the toy depends on the data.** `text_loss` is
`-log(0.3 + 0.05k)` for a step counter k, and `img_loss` is
`(0.8 + 0.02k - 1)^2` times the noise magnitude. Both are closed-form
countdowns, and the image one reaches **exactly 0** at k = 10, the last step the
demo runs.

**FINDING: the text loss goes negative at step 14.** `0.3 + 0.05k` passes 1.0
there, and `cross_entropy_toy` clamps only the lower end, so a "probability" of
1.05 yields **-0.0488**. The demo stops at 10, four steps before its own
arithmetic breaks.

Structure: `magnitude_weight` and `contribution_weight` are the two readings,
`contributions` weights a candidate by the mix, and `trajectory` evaluates the
toy's two closed-form losses at a step index.
"""

from __future__ import annotations

import inspect
import math
import re

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "13-transfusion-autoregressive-diffusion"
TEXT_SHARE, IMAGE_SHARE, MAGNITUDE = 0.70, 0.30, 10.0
DEMO_STEPS, BREAK_STEP = 10, 14


def magnitude_weight(ratio=MAGNITUDE):
    return round(1 / ratio, 4)


def contribution_weight(text=TEXT_SHARE, image=IMAGE_SHARE, ratio=MAGNITUDE):
    return round(text / (image * ratio), 4)


def contributions(img_w, text_w=1.0):
    return (round(TEXT_SHARE * text_w, 4), round(IMAGE_SHARE * img_w * MAGNITUDE, 4))


def trajectory(ref, step):
    """The toy's two losses as closed forms in the step counter."""
    return (round(ref.cross_entropy_toy(0.3 + 0.05 * step), 4),
            round((0.8 + 0.02 * step - 1) ** 2, 5))


def shipped_weights(ref):
    """The loss weights train() hard-codes, read out of its own source."""
    source = inspect.getsource(ref.train)
    return {key: float(value)
            for key, value in re.findall(r'"(text_w|img_w)":\s*([\d.]+)', source)}


def weight_keys(ref):
    """Every weights[...] key two_loss_step reads."""
    source = inspect.getsource(ref.two_loss_step)
    return sorted(set(re.findall(r'weights\["(\w+)"\]', source)))


def total_line(ref):
    """The line in two_loss_step that combines the two losses."""
    return next(line.strip() for line in inspect.getsource(ref.two_loss_step).splitlines()
                if line.strip().startswith("total ="))


def raw_text_loss(step):
    """The same text loss without the toy's lower clamp."""
    probability = 0.3 + 0.05 * step
    return round(-math.log(probability), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    weights, combine = shipped_weights(ref), total_line(ref)
    shipped = weights["img_w"]
    return {
        "magnitude": magnitude_weight(), "contribution": contribution_weight(),
        "shipped": shipped, "shipped_text_w": weights["text_w"],
        "weight_keys": weight_keys(ref), "total_line": combine,
        "shipped_contributions": contributions(shipped),
        "balanced_contributions": contributions(contribution_weight()),
        "text_gradient_share": round(contributions(shipped)[0]
                                     / sum(contributions(shipped)) * 100),
        "uses_mix": any(term in combine
                        for term in ("share", "count", "len(", str(TEXT_SHARE),
                                     str(IMAGE_SHARE))),
        "trajectory": {step: trajectory(ref, step) for step in (0, 5, DEMO_STEPS)},
        "image_zero_at": next(step for step in range(30)
                              if trajectory(ref, step)[1] == 0.0),
        "demo_steps": DEMO_STEPS,
        "negative_at": raw_text_loss(BREAK_STEP + 1),
        "break_step": BREAK_STEP,
        "clamped": trajectory(ref, BREAK_STEP + 1)[0],
    }


def verify(result):
    trajectory_rows = result["trajectory"]
    return [
        practice.Check(
            "ANSWER: 0.1 balances the magnitudes; 0.2333 balances the contributions",
            all([result["magnitude"] == 0.1, result["contribution"] == 0.2333,
                 result["shipped_contributions"] == (0.7, 0.3),
                 result["text_gradient_share"] == 70]),
            f"per token, w_img/w_text = 1/{MAGNITUDE:.0f} cancels the magnitude gap. Weighted "
            f"by the {TEXT_SHARE:.0%}/{IMAGE_SHARE:.0%} mix that leaves contributions of "
            f"{result['shipped_contributions']} -- text carrying "
            f"{result['text_gradient_share']}% of the gradient -- so equalising them needs "
            f"{TEXT_SHARE} / ({IMAGE_SHARE} x {MAGNITUDE:.0f}) = {result['contribution']}",
        ),
        practice.Check(
            "FINDING: the lesson ships the first answer and never asks the second",
            all([result["shipped"] == result["magnitude"],
                 result["shipped_text_w"] == 1.0, not result["uses_mix"],
                 result["weight_keys"] == ["img_scale", "img_w", "text_scale", "text_w"]]),
            f"read out of train's own source, it hard-codes img_w = {result['shipped']} and "
            f"text_w = {result['shipped_text_w']} -- the magnitude reading exactly. "
            f"two_loss_step combines them with `{result['total_line']}` and reads only "
            f"{result['weight_keys']} from the weights dict, so there is no token-count term "
            f"and the {TEXT_SHARE:.0%}/{IMAGE_SHARE:.0%} mix the exercise is built on cannot "
            "enter the calculation",
        ),
        practice.Check(
            "FINDING: neither loss in the toy depends on the data",
            all([trajectory_rows[0] == (1.204, 0.04), trajectory_rows[5] == (0.5978, 0.01),
                 trajectory_rows[DEMO_STEPS] == (0.2231, 0.0),
                 result["image_zero_at"] == DEMO_STEPS]),
            f"text_loss is -log(0.3 + 0.05k) and img_loss is (0.8 + 0.02k - 1)^2 times the "
            f"noise magnitude -- closed-form countdowns in the step index, not functions of "
            f"the pair. At steps 0, 5 and {DEMO_STEPS} they read {trajectory_rows}, and the "
            f"image loss reaches exactly 0 at step {result['image_zero_at']}, the last one "
            "the demo runs",
        ),
        practice.Check(
            "FINDING: the text loss goes negative at step 14",
            all([result["negative_at"] < 0, result["negative_at"] == -0.0488,
                 result["clamped"] == result["negative_at"],
                 result["demo_steps"] < result["break_step"]]),
            f"0.3 + 0.05k passes 1.0 at k = {result['break_step']}, and cross_entropy_toy "
            f"clamps only the lower end, so a probability of 1.05 yields "
            f"{result['negative_at']}. The demo stops at {result['demo_steps']}, four steps "
            "before its own arithmetic breaks",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
