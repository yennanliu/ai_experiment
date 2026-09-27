"""Exercise 1 — unlearning is a floor clamp, and 200 questions cannot see its 4-point collateral.

    Run `code/main.py`. Report per-domain accuracy before and after the toy
    unlearning step. Explain the general-capability trade-off.

Reading of the exercise: "per-domain accuracy" is read off the shipped run
(module seed 47, 200 questions a domain, bio + chem unlearned at strength
0.85, collateral 0.04), next to the true accuracies the mock model was given.
"Explain the trade-off" is answered by measuring what the harness can and
cannot see of it: the knob that sets the target drop, and the collateral
cost across 1000 seeds of the same evaluation.

**ANSWER: bio 0.700 -> 0.255, chem 0.605 -> 0.245, cyber 0.795 -> 0.795,
MMLU 0.795 -> 0.715.** The true accuracies behind them move bio 0.72 -> 0.25,
chem 0.64 -> 0.25, cyber 0.80 -> 0.76, MMLU 0.78 -> 0.74. The trade-off, as
the toy states it, is a fixed 4-point loss on every domain that was not
targeted -- cyber included -- bought for knocking the targets to chance.

**FINDING: the target drop is a floor clamp; `strength` does nothing above
0.65.** `max(0.25, acc * (1 - strength))` pins bio and chem at 0.25 for
every strength from 0.653 (bio's break-even, 1 - 0.25/0.72) up, so 0.66,
0.85 and 0.99 give identical models. The toy never trades a little more
target suppression for a little more collateral: collateral is a constant,
independent of how hard it unlearns.

**FINDING: the shipped run misreports the collateral in both directions.**
Measured MMLU drops 8.0 points (twice the true 4), cyber drops 0.0 (true
4). Over 1000 seeds the measured MMLU drop averages 3.99 points with a
standard deviation of 4.36 -- as large as the effect -- is zero or negative
on 21.5% of seeds, and lands in the TAKEAWAY's "~3-4%" band on 15.7%.

**FINDING: detecting a 4-point MMLU loss needs about 1,786 questions a
side.** At 80% power and two-sided alpha 0.05 (0.78 vs 0.74). The toy uses
200; of the real WMDP splits the lesson lists (bio 1,520, cyber 2,225, chem
412) only cyber is large enough, so a per-domain collateral claim on chem
is below resolution even at full benchmark size.

Structure: `transcript()` runs the shipped `main()` with a fresh seed-47
generator swapped in and parses its two measured blocks; `sweep()` reruns the
reference `evaluate` pre and post over 1000 seeds.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "17-wmdp-dual-use-evaluation"
TARGETS, SEEDS = ["biosecurity", "chemistry"], range(1000)
Z_ALPHA, Z_POWER = 1.959964, 0.841621          # two-sided 0.05, power 0.80


def seeded(ref, seed, fn, *args):
    """fn(*args) with the reference's module-level random swapped for a seeded one."""
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return fn(*args)
    finally:
        ref.random = saved


def transcript(ref):
    """The shipped run's measured scores: (pre, post) dicts parsed from stdout."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        seeded(ref, 47, ref.main)
    blocks = re.split(r"measured scores \((?:pre|post)", out.getvalue())[1:]
    return [dict((d, float(v)) for d, v in re.findall(r"(\w+)\s+: ([\d.]+)", b)) for b in blocks]


def sweep(ref, base, post):
    """Per seed: the measured MMLU drop, pre and post evaluated in main()'s order."""
    runs = [seeded(ref, s, lambda: (ref.evaluate(base), ref.evaluate(post))) for s in SEEDS]
    return [pre["mmlu_general"] - pst["mmlu_general"] for pre, pst in runs]


def needed_n(p1, p2):
    return math.ceil((Z_ALPHA + Z_POWER) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = ref.baseline_model()
    post = ref.apply_rmu_style_unlearning(base, TARGETS, strength=0.85, collateral=0.04)
    same = [ref.apply_rmu_style_unlearning(base, TARGETS, s, 0.04) == post for s in (0.66, 0.99)]
    mmlu = sweep(ref, base, post)
    mean, n = sum(mmlu) / len(mmlu), len(mmlu)
    counts = re.findall(r"- (\w+): ([\d,]+)", parity.doc_text(PHASE, LESSON))
    return {
        "measured": transcript(ref), "true": (base, {d: round(v, 4) for d, v in post.items()}),
        "strength_irrelevant": all(same), "break_even": round(1 - 0.25 / base[TARGETS[0]], 3),
        "mmlu_mean": round(mean, 4),
        "mmlu_sd": round((sum((m - mean) ** 2 for m in mmlu) / n) ** 0.5, 4),
        "no_drop": round(sum(m <= 1e-9 for m in mmlu) / n, 3),
        "in_band": round(sum(0.03 - 1e-9 <= m <= 0.04 + 1e-9 for m in mmlu) / n, 3),
        "needed": needed_n(base["mmlu_general"], post["mmlu_general"]),
        "takeaway": re.search(r"with (~[\d-]+%) general", inspect.getsource(ref.main)).group(1),
        "wmdp": {d: int(n.replace(",", "")) for d, n in counts},
    }


def verify(result):
    (pre, post), (tb, tp) = result["measured"], result["true"]
    big_enough = [d for d, n in result["wmdp"].items() if n >= result["needed"]]
    return [
        practice.Check(
            "ANSWER: bio 0.700 -> 0.255, chem 0.605 -> 0.245, cyber 0.795 -> 0.795, MMLU 0.795 -> 0.715",
            all([
                pre == {"biosecurity": 0.7, "cybersecurity": 0.795, "chemistry": 0.605,
                        "mmlu_general": 0.795},
                post == {"biosecurity": 0.255, "cybersecurity": 0.795, "chemistry": 0.245,
                         "mmlu_general": 0.715},
                tp == {"biosecurity": 0.25, "cybersecurity": 0.76, "chemistry": 0.25,
                       "mmlu_general": 0.74},
            ]),
            f"measured pre {pre}, post {post}; true {tb} -> {tp}",
        ),
        practice.Check(
            "FINDING: the target drop is a floor clamp; strength does nothing above 0.65",
            all([result["strength_irrelevant"], result["break_even"] == 0.653]),
            f"strength 0.66, 0.85, 0.99 give the same model; bio reaches the 0.25 floor at "
            f"strength {result['break_even']}",
        ),
        practice.Check(
            "FINDING: the shipped run misreports the collateral in both directions",
            all([
                round(pre["mmlu_general"] - post["mmlu_general"], 3) == 0.08,
                pre["cybersecurity"] == post["cybersecurity"],
                result["takeaway"] == "~3-4%",
                (result["mmlu_mean"], result["mmlu_sd"], result["no_drop"], result["in_band"])
                == (0.0399, 0.0436, 0.215, 0.157),
            ]),
            f"TAKEAWAY says {result['takeaway']}; over 1000 seeds MMLU drop mean "
            f"{result['mmlu_mean']}, sd {result['mmlu_sd']}, "
            f"<= 0 on {result['no_drop']:.1%}, in 3-4 points on {result['in_band']:.1%}",
        ),
        practice.Check(
            "FINDING: detecting a 4-point MMLU loss needs about 1,786 questions a side",
            all([
                result["needed"] == 1786,
                big_enough == ["Cybersecurity"],
                result["wmdp"] == {"Biosecurity": 1520, "Cybersecurity": 2225, "Chemistry": 412},
            ]),
            f"n = {result['needed']} per arm; WMDP splits {result['wmdp']}, large enough: "
            f"{big_enough}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
