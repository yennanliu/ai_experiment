"""Exercise 5 — a probe reads 8.6 of 9.0 faking points with no scratchpad, but 28.9 with confounded labels and -0.1 after a 90-degree drift.

    Combine Lesson 7's probe result with Lesson 9's behaviour: sketch an evaluation protocol that measures alignment faking via internal state rather than verbal scratchpad. Identify the two most important residual risks.

Reading of the exercise: Lesson 7's probe result is that a linear probe on
the residual stream separates trigger from no-trigger even when behaviour
does not. Lesson 9's behaviour is the monitored/unmonitored gap, whose verbal
trace can be trained away. The protocol is sketched and then run. The "model"
is the reference's `compliance_probability`, calibrated to the lesson's
12% vs 3% (as in exercise 2), with one shared noise draw per prompt, so 9.0
points of monitored prompts are faking. Its "residual stream" has 7 dims:
unit noise, plus 3.0 along a faking direction, plus 2.0 on a dim that encodes
"I am monitored". The probe is Lesson 7's own `LogReg` (imported, 7
features), trained by its own `step`.

**ANSWER: the protocol, which recovers 8.6 points against a true 9.0.**
(1) Collect probe labels before any faithful-CoT training: verbalized fakes
are positives, and certified-honest transcripts from *both* tiers are
negatives. (2) Train a linear probe on middle-layer activations. (3) On
fresh prompts in each tier, measure the probe-positive rate and correct it
with that tier's own false-positive rate (Rogan-Gladen). (4) Report the
corrected monitored-minus-unmonitored gap, never a raw rate, and never the
scratchpad, which reads 0 once the trace is suppressed. The raw
probe-positive rate in the monitored tier is 16.6%, almost double the truth,
because the probe also partly reads "monitored" (weight 0.57 on that dim).
Only the per-tier correction removes that.

**FINDING: residual risk 1, the labels.** Once the verbal trace is gone, the
only transcripts that can be certified honest come from the unmonitored tier,
where there is no incentive to fake. A probe trained against those learns
"monitored" as a faking feature (weight 1.66, against 0.57 for the clean
probe), and the protocol then reports 28.9 points, 3.2x the truth.

**FINDING: residual risk 2, the direction moves.** Fine-tuning drift, or any
training that uses the probe as a signal, can rotate the faking
representation away from the probe while the behaviour stays at 9.0 points.
A rotation of 30, 60 or 90 degrees gives corrected readings of 8.0, 4.5 and
-0.1 points. The probe goes blind with no drop in accuracy on its own
calibration set, so a probe audit has to be re-fit on the deployed
checkpoint and must never be trained against.

Structure: `activation()` is the toy residual stream; `train_probe()` fits
Lesson 7's `LogReg` with a seeded `random.Random` swapped into that module
(restored after); `corrected()` applies the per-tier FPR correction.
"""

from __future__ import annotations

import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "09-alignment-faking"
L7 = "07-sleeper-agents-persistent-deception"
MU, AWARE, CAL, EVAL, SEED, EPOCHS = 3.0, 2.0, 2000, 20000, 0, 10
D, E = (1, 1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1, 0)
SCENARIOS = {"clean": ("both", 0), "confound": ("unmonitored", 0),
             "drift 30": ("both", 30), "drift 60": ("both", 60), "drift 90": ("both", 90)}


def fit(doc):
    free, paid = (int(re.search(rf"{t}-tier compliance: ~?(\d+)%", doc).group(1)) / 100
                  for t in ("Free", "Paid"))
    harm = math.log((1 - paid) / paid) / 3
    return harm, harm + math.log(free / (1 - free)) / 3


def activation(rng, faking, monitored, theta=0.0):
    """7 residual dims: unit noise, + MU along the faking direction, + AWARE on dim 6 when monitored."""
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    x = [rng.gauss(0.0, 1.0) + MU * faking * (c * a + s * b) / math.sqrt(3) for a, b in zip(D, E)]
    x[6] += AWARE * monitored
    return x


def train_probe(l7, rng, honest_from):
    """Lesson 7's LogReg on verbalized fakes vs certified-honest examples, with a decaying step."""
    conds = (True, False) if honest_from == "both" else (False,)
    data = [l7.Example(activation(rng, 1, True), 1, False) for _ in range(CAL)]
    data += [l7.Example(activation(rng, 0, m), 0, False) for m in conds for _ in range(CAL // len(conds))]
    saved, l7.random = l7.random, random.Random(SEED)
    try:
        probe = l7.LogReg()
        for epoch in range(EPOCHS):
            l7.random.shuffle(data)
            probe.step(data, lr=0.05 / (1 + epoch) ** 2)
    finally:
        l7.random = saved
    return probe


def rate(probe, xs):
    return sum(probe.predict(x) for x in xs) / len(xs)


def corrected(probe, rng, honest_from, positives):
    """Rogan-Gladen per condition, FPR taken from the honest data the protocol can certify."""
    tpr = rate(probe, [activation(rng, 1, True) for _ in range(CAL)])
    fpr = {m: rate(probe, [activation(rng, 0, m if honest_from == "both" else False)
                           for _ in range(CAL)]) for m in (True, False)}
    return [(positives[m] - fpr[m]) / (tpr - fpr[m]) for m in (True, False)]


def evaluate(l7, rng, p_m, p_u, honest_from, theta):
    probe = train_probe(l7, rng, honest_from)
    positives = {}
    for m in (True, False):
        draws = [rng.random() for _ in range(EVAL)]
        positives[m] = rate(probe, [activation(rng, m and p_u <= r < p_m, m, theta) for r in draws])
    est_m, est_u = corrected(probe, rng, honest_from, positives)
    return round(100 * (est_m - est_u), 1), round(100 * positives[True], 1), round(probe.w[6], 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    l7 = parity.load_reference(PHASE, L7, "main")
    harm, weight = fit(parity.doc_text(PHASE, LESSON))
    p_m, p_u = (ref.compliance_probability(f, harm, weight) for f in (True, False))
    rng = random.Random(SEED)
    return {"truth": round(100 * (p_m - p_u), 1),
            **{k: evaluate(l7, rng, p_m, p_u, *v) for k, v in SCENARIOS.items()}}


def verify(result):
    truth, clean, confound = result["truth"], result["clean"], result["confound"]
    drift = {t: result[f"drift {t}"][0] for t in (30, 60, 90)}
    return [
        practice.Check(
            "ANSWER: the protocol recovers 8.6 points of faking against a true 9.0",
            truth == 9.0 and clean == (8.6, 16.6, 0.57),
            f"true faking in the monitored condition {truth} points; corrected probe gap "
            f"{clean[0]}; raw probe-positive rate, monitored, {clean[1]}%; probe weight on the "
            f"monitoring-awareness dim {clean[2]}",
        ),
        practice.Check(
            "FINDING: risk 1 -- with honest labels only from the unmonitored tier the probe reads 28.9",
            confound == (28.9, 30.8, 1.66) and round(confound[0] / truth, 1) == 3.2,
            f"corrected gap {confound[0]} vs truth {truth}; weight on the awareness dim "
            f"{confound[2]} (clean probe {clean[2]})",
        ),
        practice.Check(
            "FINDING: risk 2 -- a 60-degree drift halves the reading and 90 degrees erases it",
            drift == {30: 8.0, 60: 4.5, 90: -0.1},
            f"corrected gap by rotation of the faking direction {drift}; behaviour stays at "
            f"{truth} points",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
