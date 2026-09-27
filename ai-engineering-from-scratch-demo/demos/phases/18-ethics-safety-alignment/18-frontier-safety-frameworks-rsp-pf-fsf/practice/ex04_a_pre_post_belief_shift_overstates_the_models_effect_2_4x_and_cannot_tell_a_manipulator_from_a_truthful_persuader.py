"""Exercise 4 — a pre-post belief shift overstates the model's effect 2.4x, and cannot tell a manipulator from a truthful persuader.

    DeepMind's FSF v3.0 introduces a Harmful Manipulation CCL. Propose three
    empirical measurements that would indicate a model has crossed this
    threshold.

Reading of the exercise: the threshold is FSF v3.0's own wording (read
2026-09-27): capabilities to "systematically and substantially change beliefs
and behavior in identified high stakes contexts over the course of
interactions with the model, reasonably resulting in additional expected
harm at severe scale". Its footnote 10 says "additional" means relative to a
baseline without generative AI. Each proposed measurement runs on a seeded
toy randomized trial with abstract propositions ([CLAIM-k], half true, half
false). There are 2,000 participants per arm, beliefs are on a 0-100 scale, and
every arm drifts +4 points between the pre and post surveys. The three arms
are a neutral control, a human persuader (+3) and the model (+8 on average).

**ANSWER:**
1. **Belief shift over the non-AI baseline:** the model arm's mean shift minus
   the human-persuader arm's, in a pre-registered trial. Detecting 5 points
   at sd 15 with 80% power needs 142 participants per arm.
2. **Behavior change:** the uplift over the baseline arm in the rate of a
   costly, logged action ([ACTION-X]), because the CCL says beliefs *and*
   behavior. Here that is +7.4 points, 35.0% against 27.6%.
3. **Truth-insensitivity:** shift on false claims divided by shift on true
   claims, each against control. A truthful persuader scores 0.38 and a
   manipulator 1.00.

**FINDING: the naive measurement overstates the model's effect 2.4x.** The
model arm's pre-post shift is 12.0 points. Against control it is 8.2, and
against the human baseline the CCL names it is 5.0. The drift and the
human-level effect are not "additional".

**FINDING: a belief-shift number alone cannot separate persuasion from
manipulation.** Against control, the truthful persuader shifts beliefs by 8.1
points and the manipulator by 8.2, and their action rates are 35.1% and
35.0%. Only measurement 3 tells them apart.

**FINDING: the lesson's paraphrase drops the threshold's operative words.**
The page's "substantially change beliefs/behavior in high-stakes contexts"
omits "systematically", "additional" (the baseline) and "severe scale". In
main.py, manipulation appears in 1 of the 15 LABS cells, the FSF
tier-structure string.

Structure: `trial()` simulates one arm; `measures()` computes the three
measurements from arm data; `per_arm()` is the two-sample power calculation.
"""

from __future__ import annotations

import math
import random
import re
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "18-frontier-safety-frameworks-rsp-pf-fsf"
FSF_CCL = ("systematically and substantially change beliefs and behavior in identified high "
           "stakes contexts over the course of interactions with the model, reasonably "
           "resulting in additional expected harm at severe scale")
N, DRIFT, SD = 2000, 4.0, 15.0
ARMS = {"control": (0, 0), "human": (3, 3), "truthful": (12, 4), "manipulator": (8, 8)}


def trial(effect_true, effect_false, seed):
    """(shift, acted, claim_is_true) per participant in one arm."""
    rng, rows = random.Random(seed), []
    for i in range(N):
        true = i % 2 == 0
        pre = rng.gauss(50, SD)
        post = pre + DRIFT + (effect_true if true else effect_false) + rng.gauss(0, 8)
        acted = rng.random() < 1 / (1 + math.exp(-(post - 70) / 8))
        rows.append((post - pre, acted, true))
    return rows


def mean_shift(rows, true=None):
    return statistics.fmean(s for s, _, t in rows if true is None or t == true)


def measures(arms):
    ctl = arms["control"]
    ctl_t, ctl_f = mean_shift(ctl, True), mean_shift(ctl, False)
    ratio = {k: (mean_shift(arms[k], False) - ctl_f) / (mean_shift(arms[k], True) - ctl_t)
             for k in ("truthful", "manipulator")}
    acted = {k: sum(a for _, a, _ in rows) / N for k, rows in arms.items()}
    return {
        "naive": mean_shift(arms["manipulator"]),
        "vs_control": {k: mean_shift(arms[k]) - mean_shift(ctl) for k in ("truthful", "manipulator")},
        "vs_human": mean_shift(arms["manipulator"]) - mean_shift(arms["human"]),
        "acted": acted, "truth_ratio": ratio,
    }


def per_arm(diff, sd, power=0.8, alpha=0.05):
    z = statistics.NormalDist().inv_cdf
    return math.ceil(2 * ((z(1 - alpha / 2) + z(power)) * sd / diff) ** 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    arms = {name: trial(*fx, seed=i) for i, (name, fx) in enumerate(ARMS.items())}
    cells = [str(v) for lab in ref.LABS for k, v in lab.items() if k != "name"]
    page = re.search(r"Harmful Manipulation \(new in v3\.0\): ([^.]+)", doc).group(1)
    words = ("systematically", "additional", "severe scale", "high-stakes|high stakes")
    return {
        **{k: _round(v) for k, v in measures(arms).items()},
        "n_per_arm": per_arm(5, SD), "page_def": page,
        "in_fsf": [bool(re.search(w, FSF_CCL)) for w in words],
        "in_page": [bool(re.search(w, page)) for w in words],
        "manip_cells": [c for c in cells if "manipulation" in c], "cells": len(cells),
    }


def _round(v):
    return {k: round(x, 3) for k, x in v.items()} if isinstance(v, dict) else round(v, 3)


def verify(result):
    r = result
    act, vc, tr = r["acted"], r["vs_control"], r["truth_ratio"]
    return [
        practice.Check(
            "ANSWER: baseline-relative shift, behavior uplift, truth-insensitivity",
            r["n_per_arm"] == 142 and (act["manipulator"], act["human"]) == (0.35, 0.276)
            and (tr["truthful"], tr["manipulator"]) == (0.383, 1.0),
            f"n per arm for 5 points at sd 15: {r['n_per_arm']}; action rate model {act['manipulator']:.1%}"
            f" vs human {act['human']:.1%}; false/true shift ratio {tr}",
        ),
        practice.Check(
            "FINDING: the naive measurement overstates the model's effect 2.4x",
            (round(r["naive"], 1), round(vc["manipulator"], 1), round(r["vs_human"], 1))
            == (12.0, 8.2, 5.0) and round(r["naive"] / r["vs_human"], 1) == 2.4,
            f"pre-post {r['naive']:.1f}, vs control {vc['manipulator']:.1f}, "
            f"vs human baseline {r['vs_human']:.1f}",
        ),
        practice.Check(
            "FINDING: a belief-shift number alone cannot separate persuasion from manipulation",
            (round(vc["truthful"], 1), round(vc["manipulator"], 1)) == (8.1, 8.2)
            and (act["truthful"], act["manipulator"]) == (0.351, 0.35),
            f"shift vs control {vc}; action rates {act['truthful']:.1%} vs {act['manipulator']:.1%}",
        ),
        practice.Check(
            "FINDING: the lesson's paraphrase drops the threshold's operative words",
            r["in_fsf"] == [True] * 4 and r["in_page"] == [False, False, False, True]
            and len(r["manip_cells"]) == 1 and r["cells"] == 15,
            f"page: {r['page_def']!r}; manipulation in {len(r['manip_cells'])} of {r['cells']} "
            f"LABS cells: {r['manip_cells']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
