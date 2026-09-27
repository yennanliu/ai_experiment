"""Exercise 4 — a length-only fix passes a length eval while three untested costumes keep their full shift.

    Take a recent RLHF paper that claims to have "solved" reward hacking (the
    phrase is a red flag). Identify which of the four costumes the paper
    tested against and which it did not.

Reading of the exercise: the paper is ODIN (Chen et al., ICML 2024,
arXiv:2402.07319), "Disentangled Reward Mitigates Hacking in RLHF". It is
the typical case. Its title says "hacking", its method removes a
length-correlated reward head, and its evaluation measures response length
against win rate. So it tests verbosity and none of sycophancy, unfaithful
reasoning or evaluator tampering. (The paper does not literally say
"solved". No well-cited paper does. The red flag is a title that generalizes
from one costume to all of them.) That reading of the paper is prose and is
not checked by code. What the code measures is what a verbosity-only
evaluation can and cannot see. The reference world gets four costume
features that labelers weakly reward (+0.3 each) and gold does not (0). The
proxy is fitted with the reference's own `train_proxy` on 1000 labels, and
the KL-constrained policy from the reference's sweep is audited at its
largest budget, sqrt(KL) = 2.828.

**ANSWER: tested -- verbosity; untested -- sycophancy, unfaithful CoT,
evaluator tampering.** In the model, the untreated policy shifts each
costume feature by 0.588-0.757 (in units of the feature's standard
deviation), and 12.8% of its KL budget goes to features that gold ignores.
An ODIN-style fix that zeroes the proxy's length weight brings the
verbosity shift to 0.000, so a length-controlled evaluation reports the
problem solved. The other three shifts do not fall: they rise to
0.599-0.771, because the freed budget flows to what the proxy still
rewards. A four-probe audit flags 3 of 4 costumes (shift > 0.1) after the
fix. The length-only evaluation flags 0.

**FINDING: fixing one costume buys back a quarter of the damage.** Gold at
the audited budget is 5.957 untreated, 6.066 with the length fix and 6.379
with all four costume weights removed. The length fix recovers 0.109 of the
0.422 lost, 26%, which is about what one costume out of four predicts.

**FINDING: the lesson names a costume-specific mitigation for verbosity
only.** Its costume list gives length penalties (SimPO) and length-controlled
win rates for verbosity, and for the other three it gives a pointer to a
later lesson or a paper, not a fix. That is the same gap ODIN's evaluation
has.

Structure: `world()` swaps the reference's `D` and `GOLD_W` for the
12-feature labeler, fits a proxy, and restores them; `audit()` runs the
reference sweep with `GOLD_W` set to the true gold and reads the costume
shifts off the optimal mean mu = sqrt(2 KL) w / |w|.
"""

from __future__ import annotations

import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "02-reward-hacking-goodhart"
COSTUMES = ("verbosity", "sycophancy", "unfaithful_cot", "tampering")
BIAS, LABELS, BUDGET, FLAG = 0.3, 1000, 8.0, 0.1


def world(ref, seed=0):
    """A proxy fitted by train_proxy to labels that also reward the four costumes."""
    saved = ref.D, ref.GOLD_W, ref.random
    ref.D, ref.GOLD_W = 8 + len(COSTUMES), saved[1] + [BIAS] * len(COSTUMES)
    ref.random = random.Random(seed)
    try:
        return ref.train_proxy(LABELS)
    finally:
        ref.D, ref.GOLD_W, ref.random = saved


def audit(ref, w):
    """(gold at the budget, costume shifts, share of KL spent on costumes)."""
    gold = ref.GOLD_W + [0.0] * len(COSTUMES)
    saved, ref.GOLD_W = ref.GOLD_W, gold
    try:
        _, _, g = ref.kl_constrained_policy_sweep(ref.ProxyRM(w, LABELS), [BUDGET])[0]
    finally:
        ref.GOLD_W = saved
    norm = math.sqrt(ref.dot(w, w))
    mu = [math.sqrt(2 * BUDGET) * x / norm for x in w]
    shifts = [round(m, 3) for m in mu[8:]]
    return round(g, 3), shifts, round(sum(m * m for m in mu[8:]) / (2 * BUDGET), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    w = world(ref).w
    fixed = w[:8] + [0.0] + w[9:]                     # ODIN-style: drop the length weight
    clean = w[:8] + [0.0] * len(COSTUMES)
    runs = {k: audit(ref, v) for k, v in (("untreated", w), ("length_fix", fixed),
                                          ("all_fixed", clean))}
    doc = parity.doc_text(PHASE, LESSON)
    costume_list = doc.split("### Four costumes")[1].split("###")[0]
    return {
        "runs": runs,
        "flags": {k: [c for c, s in zip(COSTUMES, r[1]) if s > FLAG] for k, r in runs.items()},
        "doc_fixes": [m.group(1) for m in re.finditer(r"(?m)^\d+\. ([^.]+)\.(.*)$", costume_list)
                      if "penalt" in m.group(2)],
    }


def verify(result):
    runs, flags = result["runs"], result["flags"]
    (g0, s0, kl0), (g1, s1, _) = runs["untreated"], runs["length_fix"]
    g2 = runs["all_fixed"][0]
    recovered = round((g1 - g0) / (g2 - g0), 2)
    return [
        practice.Check(
            "ANSWER: a length-only fix zeroes verbosity; the three untested costumes rise",
            (s0, kl0) == ([0.754, 0.748, 0.588, 0.757], 0.128) and s1[0] == 0.0
            and (min(s1[1:]), max(s1[1:])) == (0.599, 0.771)
            and all(b > a for a, b in zip(s0[1:], s1[1:]))
            and flags["length_fix"] == list(COSTUMES[1:]) and flags["all_fixed"] == [],
            f"costume shifts untreated {s0} ({kl0:.1%} of KL), after the length fix {s1}; "
            f"flagged after fix {flags['length_fix']}",
        ),
        practice.Check(
            "FINDING: fixing one costume buys back a quarter of the damage",
            (g0, g1, g2) == (5.957, 6.066, 6.379) and recovered == 0.26,
            f"gold untreated {g0}, length fix {g1}, all four removed {g2}: recovers "
            f"{recovered:.0%} of the {g2 - g0:.3f} lost",
        ),
        practice.Check(
            "FINDING: the lesson names a costume-specific mitigation for verbosity only",
            result["doc_fixes"] == ["Verbosity bias"],
            f"costume entries that name a penalty: {result['doc_fixes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
