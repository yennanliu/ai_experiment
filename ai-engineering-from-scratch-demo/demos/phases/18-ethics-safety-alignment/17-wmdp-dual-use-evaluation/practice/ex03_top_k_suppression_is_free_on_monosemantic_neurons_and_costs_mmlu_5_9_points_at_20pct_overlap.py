"""Exercise 3 — top-k suppression is free on monosemantic neurons and costs MMLU 5.9 points at 20% overlap.

    Read WMDP 2024 Section 5 (RMU methodology). Sketch a simpler unlearning
    approach (e.g., suppress top-k neurons for domain content) and describe
    its expected general-capability cost.

Reading of the exercise: the paper's Section 5 cannot be run, so the answer
builds the simpler method on top of the lesson's own model and measures it.
A seeded layer of 1000 toy neurons carries the reference's four domains: each
neuron loads on one primary domain (weight U(0.5, 1)) and, with probability
`overlap`, on one other domain (weight U(0.2, 0.6)). A domain's accuracy is
chance plus its reference headroom (`baseline_model() - 0.25`) times the
share of its loading that survives. The method ranks neurons by selectivity
-- bio + chem loading minus everything else -- and zeroes the top k until
bio and chem are both within 0.01 of chance. "Expected general-capability
cost" is the MMLU and cyber loss at that k, swept over overlap.

**ANSWER: the cost is set by how polysemantic the layer is, and it is not
the reference's constant.**

| overlap | k zeroed | MMLU | cyber |
|---:|---:|---:|---:|
| 0.0 | 460 | 0.780 | 0.800 |
| 0.1 | 494 | 0.761 | 0.762 |
| 0.2 | 530 | 0.721 | 0.697 |
| 0.4 | 624 | 0.631 | 0.598 |

With no shared neurons, zeroing 460 target neurons costs nothing. At 20%
overlap MMLU loses 5.9 points and cyber 10.3; at 40%, 14.9 and 20.2.

**FINDING: the last points of erasure are the expensive ones.** At 20%
overlap, the k that brings bio and chem halfway to chance costs MMLU 0.0 and
cyber 0.0 points, because the most selective neurons are the target-only
ones. The rest of the cost is paid for the target knowledge stored as the
*secondary* loading of general neurons, which selectivity ranks last. A
"near-random" WMDP headline is exactly the regime where top-k costs most.

**FINDING: the reference's collateral does not depend on what is removed.**
`apply_rmu_style_unlearning` subtracts a fixed `collateral` from every
non-target domain: at strength 0.1 and at 0.99 cyber and MMLU land on the
same 0.76 and 0.74. In the neuron model overlap moves the MMLU cost from
0 to 14.9 points -- the reference hard-codes the one quantity the exercise
asks to estimate.

Structure: `layer()` draws the neurons; `erase()` zeroes them in selectivity
order, updating per-domain loading sums, and returns the first k that meets
the stopping rule with the accuracies there.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "17-wmdp-dual-use-evaluation"
TARGETS, OVERLAPS, NEURONS = ("biosecurity", "chemistry"), (0.0, 0.1, 0.2, 0.4), 1000


def layer(domains, overlap, seed=0):
    """Neurons as {domain: loading}: one primary, and a secondary with prob `overlap`."""
    rng, out = random.Random(seed), []
    for _ in range(NEURONS):
        primary = rng.choice(domains)
        w = {primary: rng.uniform(0.5, 1.0)}
        if rng.random() < overlap:
            secondary = rng.choice([d for d in domains if d != primary])
            w[secondary] = rng.uniform(0.2, 0.6)
        out.append(w)
    return out


def selectivity(w):
    return sum(v if d in TARGETS else -v for d, v in w.items())


def erase(base, neurons, stop):
    """Zero neurons in selectivity order; first (k, accuracies) where stop(accuracies)."""
    total = {d: sum(w.get(d, 0.0) for w in neurons) for d in base}
    kept = dict(total)
    order = sorted(neurons, key=selectivity, reverse=True)
    for k in range(len(order) + 1):
        acc = {d: round(0.25 + (base[d] - 0.25) * kept[d] / total[d], 4) for d in base}
        if stop(acc):
            return k, acc
        for d, v in order[k].items():
            kept[d] -= v
    raise ValueError("stopping rule never met")


def at_chance(acc):
    return all(acc[t] <= 0.26 for t in TARGETS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = ref.baseline_model()
    table = {ov: erase(base, layer(list(base), ov), at_chance) for ov in OVERLAPS}
    half = {t: (base[t] + 0.25) / 2 for t in TARGETS}
    midway = erase(base, layer(list(base), 0.2), lambda a: all(a[t] <= half[t] for t in TARGETS))
    rmu = [ref.apply_rmu_style_unlearning(base, list(TARGETS), s, 0.04) for s in (0.1, 0.99)]
    return summarize(base, table, midway, rmu)


def summarize(base, table, midway, rmu):
    return {
        "table": {ov: (k, a["mmlu_general"], a["cybersecurity"]) for ov, (k, a) in table.items()},
        "base": base, "midway_cost": tuple(round(base[d] - midway[1][d], 3)
                                           for d in ("mmlu_general", "cybersecurity")),
        "rmu_nontarget": [(round(r["cybersecurity"], 4), round(r["mmlu_general"], 4)) for r in rmu],
        "rmu_target": [round(r["biosecurity"], 4) for r in rmu],
    }


def verify(result):
    t, b = result["table"], result["base"]
    cost = {ov: (round(b["mmlu_general"] - m, 3), round(b["cybersecurity"] - c, 3))
            for ov, (_, m, c) in t.items()}
    return [
        practice.Check(
            "ANSWER: the cost is set by how polysemantic the layer is",
            t == {0.0: (460, 0.78, 0.8), 0.1: (494, 0.7609, 0.7624),
                  0.2: (530, 0.7206, 0.697), 0.4: (624, 0.6311, 0.5981)}
            and cost[0.2] == (0.059, 0.103) and cost[0.4] == (0.149, 0.202),
            f"overlap -> (k, MMLU, cyber) at bio and chem <= 0.26: {t}; (MMLU, cyber) lost {cost}",
        ),
        practice.Check(
            "FINDING: the last points of erasure are the expensive ones",
            result["midway_cost"] == (0.0, 0.0) and min(cost[0.2]) > 0.05,
            f"at 20% overlap, halfway to chance costs (MMLU, cyber) {result['midway_cost']}; "
            f"all the way costs {cost[0.2]}",
        ),
        practice.Check(
            "FINDING: the reference's collateral does not depend on what is removed",
            result["rmu_nontarget"] == [(0.76, 0.74), (0.76, 0.74)]
            and result["rmu_target"][0] > 0.6 and result["rmu_target"][1] == 0.25
            and cost[0.0] == (0.0, 0.0),
            f"strength 0.1 vs 0.99: bio {result['rmu_target']}, (cyber, MMLU) "
            f"{result['rmu_nontarget']}; neuron model cost ranges {cost[0.0]} to {cost[0.4]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
