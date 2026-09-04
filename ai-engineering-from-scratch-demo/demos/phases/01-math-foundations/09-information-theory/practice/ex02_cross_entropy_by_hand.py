"""Exercise 2 — cross-entropy for logits [5, 2, 0.5], class 1; and zero loss.

    A model outputs logits [5.0, 2.0, 0.5] for a sample with true class 1.
    Compute the cross-entropy loss by hand, then verify with your
    `cross_entropy_loss` function. What logits would give zero loss?

Reading of the exercise: the last question is a trap, and answering it honestly
is the point. **No finite logits give exactly zero loss.** Loss is
−log softmax(z)[1], which is 0 only when p₁ = 1, and softmax never reaches 1
while the other logits are finite. Check 4 sweeps the gap and shows the loss
approaching 0 without arriving; check 5 finds where float64 makes it *round* to
zero, which is a different claim and the one worth knowing.

By hand: e⁻³ = 0.049787, e⁻⁴·⁵ = 0.011109, so logsumexp = 5 + ln(1.060896)
= 5.059114 and the loss is 5.059114 − 2 = **3.059114**.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "09-information-theory"
LOGITS = [5.0, 2.0, 0.5]
TRUE_CLASS = 1


def by_hand(logits, target):
    """−log p_target, via log-sum-exp, computed independently of the lesson."""
    peak = max(logits)
    log_sum = peak + math.log(sum(math.exp(v - peak) for v in logits))
    return log_sum - logits[target]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "information_theory")
    theirs = ref.cross_entropy_loss(TRUE_CLASS, LOGITS)
    probabilities = ref.softmax(LOGITS)
    sweep = {}
    for margin in (1, 5, 10, 20, 40, 80):
        logits = [0.0, margin, 0.0]
        sweep[margin] = ref.cross_entropy_loss(1, logits)
    exactly_zero = next((m for m, loss in sweep.items() if loss == 0.0), None)
    return {"mine": by_hand(LOGITS, TRUE_CLASS), "theirs": theirs,
            "probabilities": probabilities, "sweep": sweep,
            "exactly_zero": exactly_zero}


def verify(result):
    return [
        practice.Check("by hand: logsumexp − z₁ = 3.059114",
                       abs(result["mine"] - 3.059114) < 1e-6,
                       f"derived {result['mine']:.6f} independently of the lesson's function"),
        practice.Check("the lesson's cross_entropy_loss agrees",
                       abs(result["mine"] - result["theirs"]) < 1e-12,
                       f"cross_entropy_loss(1, {LOGITS}) = {result['theirs']:.9f}"),
        practice.Check("the loss is −log p₁, and class 1 is not the argmax",
                       abs(result["theirs"] + math.log(result["probabilities"][1])) < 1e-12
                       and result["probabilities"].index(max(result["probabilities"])) == 0,
                       f"p = {[round(p, 5) for p in result['probabilities']]}; the model "
                       f"prefers class 0, so the loss is large"),
        practice.Check("ANSWER: no finite logits give exactly zero loss",
                       all(loss > 0 for m, loss in result["sweep"].items() if m <= 20),
                       "loss as the winning margin grows: " + ", ".join(
                           f"{m}→{result['sweep'][m]:.3g}" for m in (1, 5, 10, 20))
                       + " — softmax(z)[1] < 1 for any finite z, so −log p₁ > 0 always"),
        practice.Check("…it only *rounds* to zero, once the margin exceeds float64's reach",
                       result["exactly_zero"] is not None,
                       f"first exact 0.0 at margin {result['exactly_zero']} "
                       f"(loss at 20 is still {result['sweep'][20]:.3g}) — the answer to "
                       f"'what logits give zero loss' is a floating-point fact, not a "
                       f"mathematical one"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
