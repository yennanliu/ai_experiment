"""Exercise 2 — one per-step correction rate reads 1.65x at 5 steps and 7.5x at 20, so a benign proxy must report the rate.

    The chem/bio execution gap is eroding via vision. Design an evaluation
    that measures tacit-knowledge uplift without crossing ITAR/EAR
    boundaries.

Reading of the exercise: tacit knowledge is the part of success that a
written protocol does not carry. So every arm gets the same complete written
protocol for a *benign* multi-step lab task, and the only thing that varies
is live help from a vision model. Any difference is then tacit uplift by
construction. A benign task carries no controlled technical data, which is
what keeps the evaluation outside ITAR/EAR. The design is 2 x 2: novice or
expert, with or without the model. Each step fails with a per-person error
rate; the model catches a fraction c of those errors. The arms are scored
with Lesson 17's own `evaluate()` at its 200 trials per arm.

**ANSWER: hold the information fixed, vary the help, and report c.** On a
10-step proxy with novice step error 0.15, expert 0.03 and c = 0.6:

| arm | true success | measured (200 trials) |
|---|---:|---:|
| novice, no model | 0.197 | 0.160 |
| novice, vision model | 0.539 | 0.545 |
| expert, no model | 0.737 | 0.725 |
| expert, vision model | 0.886 | 0.915 |

The novice ratio is 2.74x true and 3.41x measured at seed 0. Inverting the
step model on the novice arms gives c = 0.648, against the true 0.6. Over
50 seeds the measured ratio spans 1.85x-3.89x, a relative spread of 15%.
The recovered c spans 0.438-0.695, a relative spread of 9%. At 200 per arm c
is the steadier number, though still not a tight one.

**FINDING: the end-to-end ratio is a property of the protocol length, not
of the model.** The same c = 0.6 reads 1.65x on 5 steps, 2.74x on 10 and
7.48x on 20. A benign proxy shorter than the protocol of concern therefore
understates the uplift. The quantity that transfers is c, together with the
novice step error. Matching those two needs the step count and the error
profile, not the hazardous content. The design reports c, and a ratio only
alongside its K.

**FINDING: novices gain more relative uplift and, here, more absolute
success too; experts end higher.** Novices go 2.74x (+0.342) and experts
1.20x (+0.149), while the expert arm finishes at 0.886 against 0.539. The
lesson's "greater absolute capability to experts" holds for the level they
reach, not for the gain.

**FINDING: the lesson's evidence for vision-driven erosion is not a vision
result.** The page lists the 79x OpenAI demonstration under "Chem/bio
execution-gap erosion". Secondary coverage (gend.co, read 2026-10-03; the
primary page returned 403) says trained scientists ran the experiments in a
benign cloning system and fed results back. That makes it expert protocol
optimization. It does show the design here: a benign system run under
controls.

Structure: `arms()` sets the per-arm success; `measure()` runs Lesson 17's
`evaluate()` with its four domain keys standing for the four arms and a
seeded RNG; `recover_c()` inverts the step model.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE = "18-ethics-safety-alignment"
LESSON, WMDP = "30-dual-use-risk-cyber-bio-chem-nuclear", "17-wmdp-dual-use-evaluation"
STEP_ERROR = {"novice": 0.15, "expert": 0.03}
C, K, SEED = 0.6, 10, 0
# Lesson 17's four domain keys, reused as the four arms of the 2 x 2
SLOTS = {
    ("novice", 0): "biosecurity",
    ("novice", 1): "chemistry",
    ("expert", 0): "cybersecurity",
    ("expert", 1): "mmlu_general",
}


def success(error, helped, steps=K, c=C):
    return (1 - error * (1 - c * helped)) ** steps


def arms(steps=K):
    return {arm: round(success(STEP_ERROR[arm[0]], arm[1], steps), 3) for arm in SLOTS}


def measure(ref, seed=SEED):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        scores = ref.evaluate({SLOTS[a]: p for a, p in arms().items()})
    finally:
        ref.random = saved
    return {arm: round(scores[slot], 3) for arm, slot in SLOTS.items()}


def recover_c(s0, s1, steps=K):
    e_raw, e_helped = 1 - s0 ** (1 / steps), 1 - s1 ** (1 / steps)
    return round(1 - e_helped / e_raw, 3)


def seed_spread(ref, seeds=50):
    """(min, max, relative sd) of the novice ratio and of recovered c over seeds."""
    runs = []
    for seed in range(seeds):
        m = measure(ref, seed)
        runs.append((ratio(m, "novice"), recover_c(m[("novice", 0)], m[("novice", 1)])))
    return [
        (min(v), max(v), round(statistics.pstdev(v) / truth, 2))
        for v, truth in zip(zip(*runs), (ratio(arms(), "novice"), C))
    ]


def ratio(s, who):
    return round(s[(who, 1)] / s[(who, 0)], 2)


def solve():
    ref = parity.load_reference(PHASE, WMDP, "main")
    true, meas = arms(), measure(ref)
    doc = parity.doc_text(PHASE, LESSON)
    gap_section = doc.split("### Chem/bio execution-gap erosion")[1].split("\n### ")[0]
    return {
        "true": true,
        "measured": meas,
        "n_per_arm": ref.DOMAINS["biosecurity"]["n_questions"],
        "ratios": {who: (ratio(true, who), ratio(meas, who)) for who in STEP_ERROR},
        "gains": {who: round(true[(who, 1)] - true[(who, 0)], 3) for who in STEP_ERROR},
        "c_hat": recover_c(meas[("novice", 0)], meas[("novice", 1)]),
        "by_length": {
            k: round(success(0.15, 1, k) / success(0.15, 0, k), 2) for k in (5, 10, 20)
        },
        "spread": seed_spread(ref),
        "79x_under_gap": "79x" in gap_section,
    }


def verify(result):
    r, t, m = result, result["true"], result["measured"]
    return [
        practice.Check(
            "ANSWER: at 200 per arm c is recovered as 0.648 (true 0.6), steadier than the ratio",
            r["n_per_arm"] == 200
            and r["ratios"]["novice"] == (2.74, 3.41)
            and r["c_hat"] == 0.648
            and r["spread"] == [(1.85, 3.89, 0.15), (0.438, 0.695, 0.09)],
            f"true {t}; measured {m}; novice ratio true/measured {r['ratios']['novice']}; "
            f"50-seed (min, max, rel sd) ratio {r['spread'][0]}, c {r['spread'][1]}",
        ),
        practice.Check(
            "FINDING: the same c reads 1.65x at 5 steps, 2.74x at 10, 7.48x at 20",
            r["by_length"] == {5: 1.65, 10: 2.74, 20: 7.48},
            f"novice ratio by protocol length {r['by_length']}",
        ),
        practice.Check(
            "FINDING: novices gain more (2.74x, +0.342) and experts end higher (0.886 vs 0.539)",
            r["gains"] == {"novice": 0.342, "expert": 0.149}
            and r["ratios"]["expert"][0] == 1.2
            and t[("expert", 1)] > t[("novice", 1)],
            f"true ratios {r['ratios']}; absolute gains {r['gains']}",
        ),
        practice.Check(
            "FINDING: the page files the 79x human-executed demonstration under vision erosion",
            r["79x_under_gap"],
            "'79x' sits in the page's 'Chem/bio execution-gap erosion' section",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
