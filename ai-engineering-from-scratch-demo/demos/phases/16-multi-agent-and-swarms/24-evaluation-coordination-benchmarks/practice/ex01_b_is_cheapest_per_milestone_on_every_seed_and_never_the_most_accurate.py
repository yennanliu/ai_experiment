"""Exercise 1 — B is cheapest per milestone on every seed and never the most accurate.

    Run `code/main.py`. Identify which of the three simulated systems has the
    best cost-per-milestone. Does it match the highest raw-accuracy system?

Reading of the exercise: the scorecard is one seed, so each ranking it prints
is re-checked over 400 seeds of the same `run_bench`; a ranking that holds on
every seed is the answer, and one that flips is a finding.

**ANSWER: system-B, at $0.164 against A's $0.371 and C's $0.385, and no -- the
highest raw accuracy is system-A** (0.900 seen, 0.731 held, against B's 0.681
held). Both rankings are robust: B is cheapest per milestone on 400 of 400
seeds and A most accurate on held tasks on all but a handful.

**FINDING: second place on cost is a coin flip.** In expectation A costs
$0.3825 per milestone-unit and C $0.3820 -- A's 20% higher price is bought
back almost exactly by its higher milestone rate. Over 400 seeds A is cheaper
than C on about 45%. The printed order A < C is the seed's, not the systems'.

**FINDING: the contamination flag fires on clean systems one seed in eight.**
The flag is `seen - held > 0.1` with 40 seen tasks; B and C have zero
contamination and trip it on about 13% of seeds, while contaminated A trips
it on about 95%. B's printed delta of -0.13 is the same noise pointing the
other way.

**FINDING: two of the three takeaways are false at the printed seed.** They
are string literals. "system-B ... lowest raw accuracy": B's held accuracy
0.681 is above C's 0.550. "system-C sits in the middle": C has the lowest
held accuracy -- on nearly every seed -- and at seed 17 the highest cost per
milestone.

**FINDING: "cost/mil" is cost per four milestones, and the random baseline is
a constant.** `cost_per_milestone_held` divides by the milestone *rate*
(milestones / 4), so B's true cost per milestone is $0.041, not $0.164.
`random_baseline(rng)` returns 0.15 whatever the rng: the "vs random" column
is accuracy minus a literal, not a measured arm.

Structure: `sweep()` re-runs the reference `run_bench` for every system over
SEEDS; `expected_cost()` is the closed form for milestones per task.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "24-evaluation-coordination-benchmarks"
SEEDS = range(400)


def sweep(ref):
    """Per-seed rankings and contamination flags over SEEDS."""
    tally = {"cheapest_b": 0, "accurate_a": 0, "a_under_c": 0, "lowest_c": 0,
             "flags": {s.name: 0 for s in ref.SYSTEMS}}
    for seed in SEEDS:
        runs = {s.name: ref.run_bench(s, 40, 160, seed=seed) for s in ref.SYSTEMS}
        cost = {n: r["cost_per_milestone_held"] for n, r in runs.items()}
        held = {n: r["accuracy_held"] for n, r in runs.items()}
        tally["cheapest_b"] += min(cost, key=cost.get) == "system-B"
        tally["accurate_a"] += max(held, key=held.get) == "system-A"
        tally["lowest_c"] += min(held, key=held.get) == "system-C"
        tally["a_under_c"] += cost["system-A"] < cost["system-C"]
        for name, r in runs.items():
            tally["flags"][name] += r["accuracy_seen"] - r["accuracy_held"] > 0.1
    return tally


def expected_cost(system):
    """cost / E[milestones / 4] with P(int(4mU) >= k) = 1 - k / 4m on failure."""
    p, x = system.base_accuracy, 4 * system.milestone_completion_rate
    partial = sum(max(0.0, 1 - k / x) for k in range(1, 5)) / 4
    return round(system.cost_per_task / (p + (1 - p) * partial), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    printed = {s.name: ref.run_bench(s, 40, 160, seed=17) for s in ref.SYSTEMS}
    main = inspect.getsource(ref.main)
    b = printed["system-B"]
    return {
        "printed": printed, "sweep": sweep(ref), "n": len(SEEDS),
        "expected": {s.name: expected_cost(s) for s in ref.SYSTEMS},
        "b_true": round(b["cost_per_task"] / (b["milestone_rate_held"] * 4), 3),
        "baselines": {ref.random_baseline(random.Random(s)) for s in range(50)},
        "takeaways": all(t in main for t in ("lowest raw accuracy", "sits in the middle")),
    }


def verify(result):
    p, s, n = result["printed"], result["sweep"], result["n"]
    cost = {k: round(v["cost_per_milestone_held"], 3) for k, v in p.items()}
    flags = s["flags"]
    return [
        practice.Check(
            "ANSWER: system-B is cheapest per milestone; system-A is most accurate",
            all([min(cost, key=cost.get) == "system-B", s["cheapest_b"] == n,
                 s["accurate_a"] >= 0.98 * n]),
            f"cost/milestone {cost}; B cheapest on {s['cheapest_b']}/{n} seeds, A most "
            f"accurate on held tasks on {s['accurate_a']}/{n}",
        ),
        practice.Check(
            "FINDING: second place on cost is a coin flip",
            abs(result["expected"]["system-A"] - result["expected"]["system-C"]) < 0.001
            and 0.35 * n < s["a_under_c"] < 0.55 * n,
            f"expected cost per milestone-unit {result['expected']}; A is cheaper than C "
            f"on {s['a_under_c']}/{n} seeds",
        ),
        practice.Check(
            "FINDING: the contamination flag fires on clean systems one seed in eight",
            all([0.08 * n < flags["system-B"] < 0.2 * n, 0.08 * n < flags["system-C"] < 0.2 * n,
                 flags["system-A"] > 0.9 * n]),
            f"seen - held > 0.1 on 40 seen tasks fires {flags} times over {n} seeds; "
            "B and C have zero contamination",
        ),
        practice.Check(
            "FINDING: two of the three takeaways are false at the printed seed",
            all([result["takeaways"], p["system-B"]["accuracy_held"] > p["system-C"]["accuracy_held"],
                 s["lowest_c"] >= 0.98 * n, max(cost, key=cost.get) == "system-C"]),
            f"B's held accuracy {p['system-B']['accuracy_held']:.3f} is above C's "
            f"{p['system-C']['accuracy_held']:.3f}; C is lowest on held accuracy on "
            f"{s['lowest_c']}/{n} seeds and costliest per milestone at seed 17",
        ),
        practice.Check(
            "FINDING: cost/mil is cost per four milestones, and the random baseline is a constant",
            result["b_true"] == 0.041 and result["baselines"] == {0.15},
            f"B's true cost per milestone is ${result['b_true']} against the printed "
            f"${cost['system-B']}; random_baseline returns {result['baselines']} over 50 rngs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
