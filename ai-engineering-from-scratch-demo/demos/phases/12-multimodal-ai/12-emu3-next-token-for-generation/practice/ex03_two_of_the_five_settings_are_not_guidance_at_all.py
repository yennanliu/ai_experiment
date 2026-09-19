"""Exercise 3 — two of the five settings are not guidance at all.

    Classifier-free guidance weight 5.0 vs 3.0: what visual effect? Trace the
    math in `code/main.py`.

Reading of the exercise: the trace is run on the lesson's own logits through its
own `cfg_mix` and `softmax`, and the "visual effect" is reported as the three
quantities that actually change -- top probability, margin over the runner-up,
and entropy -- because "sharper" is not an effect, it is a description of the
same number the mixing already produced.

**ANSWER: going 3.0 -> 5.0 moves the top probability 0.7398 -> 0.8438** (+14%),
the margin over second place **0.4936 -> 0.6896** (+40%), and the entropy
**0.6293 -> 0.4445** (-29%). The visual effect is fewer, more confident token
choices: closer adherence to the prompt, less variety between samples, and the
saturation and contrast artefacts that come from a sampler that stops exploring.

**FINDING: the mixing is affine and the effect is saturating.** The top-two
logit margin works out to `0.2 + 0.3*gamma` -- a line whose intercept is the
*unconditional* margin, so the growth from 3.0 to 5.0 is **1.55x** and not the
5/3 a proportional term would give. The probability margin grows **1.40x**,
because the softmax saturates what the mixing scales, and a gap of
probabilities can never exceed 1.

**FINDING: two of the five settings the lesson prints are not guidance.** At
gamma = 1.0 `cfg_mix` returns the conditional logits exactly and at gamma = 0.0
the unconditional ones. The table's first two rows are the two endpoints being
interpolated, not settings of the method.

**FINDING: two thirds of the entropy survives the conditional alone.**
Five classes cap entropy at ln(5) = 1.6094, and the conditional distribution
alone (gamma = 1.0) sits at **1.0549**. Guidance at 7.0 takes it to 0.3078 --
so most of the range is spent between gamma 1 and 3, and the 3-to-5 step the
exercise asks about moves it by a fifth of what the first step did.

Structure: `mix` runs the lesson's own `cfg_mix` and `softmax`, `profile`
measures the three quantities, and `GAMMAS` is the lesson's own sweep.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "12-emu3-next-token-for-generation"
COND = [2.0, 4.0, 1.0, 3.5, 0.5]
UNCOND = [1.0, 2.0, 1.5, 1.8, 1.2]
GAMMAS = (0.0, 1.0, 3.0, 5.0, 7.0)
ASKED = (3.0, 5.0)


def mix(ref, gamma):
    logits = ref.cfg_mix(COND, UNCOND, gamma)
    return logits, ref.softmax(logits)


def profile(logits, probs):
    ranked = sorted(probs, reverse=True)
    ranked_logits = sorted(logits, reverse=True)
    return {"top": round(max(probs), 4),
            "margin": round(ranked[0] - ranked[1], 4),
            "logit_margin": round(ranked_logits[0] - ranked_logits[1], 4),
            "entropy": round(-sum(p * math.log(p) for p in probs), 4)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {gamma: profile(*mix(ref, gamma)) for gamma in GAMMAS}
    low, high = (sweep[gamma] for gamma in ASKED)
    return {
        "sweep": sweep,
        "top_change_pct": round((high["top"] / low["top"] - 1) * 100),
        "margin_growth": round(high["margin"] / low["margin"], 2),
        "logit_margin_growth": round(high["logit_margin"] / low["logit_margin"], 2),
        "intercept": round(UNCOND[1] - UNCOND[3], 4),
        "slope": round((COND[1] - COND[3]) - (UNCOND[1] - UNCOND[3]), 4),
        "entropy_change_pct": round((high["entropy"] / low["entropy"] - 1) * 100),
        "is_conditional": mix(ref, 1.0)[0] == COND,
        "is_unconditional": mix(ref, 0.0)[0] == UNCOND,
        "max_entropy": round(math.log(len(COND)), 4),
        "conditional_share": round(sweep[1.0]["entropy"] / math.log(len(COND)) * 100),
        "first_step": round(sweep[1.0]["entropy"] - sweep[3.0]["entropy"], 4),
        "asked_step": round(sweep[3.0]["entropy"] - sweep[5.0]["entropy"], 4),
    }


def verify(result):
    sweep = result["sweep"]
    return [
        practice.Check(
            "ANSWER: 3.0 -> 5.0 moves the top probability 0.7398 -> 0.8438",
            all([sweep[3.0]["top"] == 0.7398, sweep[5.0]["top"] == 0.8438,
                 sweep[3.0]["margin"] == 0.4936, sweep[5.0]["margin"] == 0.6896,
                 sweep[3.0]["entropy"] == 0.6293, sweep[5.0]["entropy"] == 0.4445,
                 result["top_change_pct"] == 14, result["entropy_change_pct"] == -29]),
            f"top probability {sweep[3.0]['top']} -> {sweep[5.0]['top']} "
            f"({result['top_change_pct']:+}%), margin over second "
            f"{sweep[3.0]['margin']} -> {sweep[5.0]['margin']}, entropy "
            f"{sweep[3.0]['entropy']} -> {sweep[5.0]['entropy']} "
            f"({result['entropy_change_pct']}%). Fewer, more confident choices: closer "
            "prompt adherence, less variety, and the artefacts of a sampler that has stopped "
            "exploring",
        ),
        practice.Check(
            "FINDING: the mixing is linear and the effect is not",
            all([result["logit_margin_growth"] == 1.55,
                 result["margin_growth"] == 1.4,
                 result["intercept"] == 0.2, result["slope"] == 0.3,
                 result["logit_margin_growth"] > result["margin_growth"]]),
            f"cfg_mix is u + gamma*(c - u), so the top-two logit margin is "
            f"{result['intercept']} + {result['slope']}*gamma -- affine, with the "
            f"unconditional margin as a nonzero intercept, which is why it grows "
            f"{result['logit_margin_growth']}x from 3.0 to 5.0 rather than the 5/3 a "
            f"proportional term would give. The probability margin grows "
            f"{result['margin_growth']}x, because the softmax saturates what the mixing "
            "scales, and a probability gap can never exceed 1",
        ),
        practice.Check(
            "FINDING: two of the five settings the lesson prints are not guidance",
            all([result["is_conditional"], result["is_unconditional"],
                 len(GAMMAS) == 5]),
            f"at gamma = 1.0 cfg_mix returns the conditional logits exactly and at 0.0 the "
            f"unconditional ones. The first two of the {len(GAMMAS)} rows the lesson prints "
            "are the two endpoints being interpolated, not settings of the method",
        ),
        practice.Check(
            "FINDING: two thirds of the entropy survives the conditional alone",
            all([result["max_entropy"] == 1.6094, result["conditional_share"] == 66,
                 result["first_step"] == 0.4256, result["asked_step"] == 0.1848,
                 result["first_step"] > 2 * result["asked_step"]]),
            f"five classes cap entropy at ln(5) = {result['max_entropy']}, and the "
            f"conditional distribution alone sits at {sweep[1.0]['entropy']} -- "
            f"{result['conditional_share']}% of it. Gamma 1 to 3 removes "
            f"{result['first_step']} nats and 3 to 5 removes {result['asked_step']}, so the "
            "step the exercise asks about is worth less than half the one before it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
