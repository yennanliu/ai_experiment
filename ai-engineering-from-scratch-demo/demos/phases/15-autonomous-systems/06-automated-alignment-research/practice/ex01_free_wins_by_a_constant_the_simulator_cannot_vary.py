"""Exercise 1 — free wins by a constant the simulator cannot vary.

    Run `code/main.py`. Compare the "fixed-workflow" vs "free-decomposition"
    settings. Does free decomposition always win, or are there problem
    classes where the fixed workflow is competitive?

Reading of the exercise: "problem classes" asks about an interaction between
the regime and the task, so the first thing to check is whether the model has
a term for one. It does not, which turns the question into a different and
answerable one: competitive *at what*. Means are measured over 20000 draws
per task rather than the shipped 3, because the effect is smaller than the
noise on three.

**ANSWER: free wins the mean on every task by exactly the same 0.025, and
loses badly on the floor.** `solve` adds `U(0, 0.25)` under the fixed regime
and `N(0.15, 0.22)` under the free one, so the regime term never touches
`base`: the gap is **0.025** for all **5** tasks and no problem class can
differ. What does differ is spread -- **0.0722** against **0.2201**, a
**3.0x** -- so a single free draw beats a single fixed draw only **54.3%**
of the time, and **24.8%** of free draws land *below the fixed regime's
floor*.

**FINDING: the free regime produces impossible scores.** **3.7%** of free
draws fall outside [0, 1] -- **2.5%** above 1 and **1.3%** below 0 -- against
**0.0%** for fixed. Nothing clamps them, and `regime_report` averages them in,
so the free regime's reported advantage is partly made of results that cannot
exist.

**FINDING: the agent is a parameter `solve` never reads.** Its signature
takes `agent`, and the name appears **1** time in the function -- in the
signature. All three AARs are the same distribution, so the forum's "parallel
researchers" differ only in their label, and any per-agent comparison a
reviewer draws off this log is reading noise.

**FINDING: three draws cannot see the effect.** `run_regime` posts **15**
records per regime, **3** per task. Against a 0.025 gap at the measured
spreads, that is a power of **0.06** -- a coin weighted 1-in-16 against
noticing. The shipped comparison is not a weak measurement of the difference;
it is not a measurement of it.

Structure: `draws()` samples one regime and task; `power_at()` prices the
shipped sample size against the gap it is supposed to show.
"""

from __future__ import annotations

import inspect
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "06-automated-alignment-research"

DRAWS, SEED = 20000, 0
FREE_MEAN, FREE_SD, FIXED_SPAN = 0.15, 0.22, 0.25    # solve()'s own two distributions
GAP, ALPHA_Z = FREE_MEAN - FIXED_SPAN / 2, 1.959964


def draws(ref, task, regime, count=DRAWS):
    return [ref.solve("AAR-A", task, regime) for _ in range(count)]


def profile(ref, regime):
    """(mean, population sd) per task, at one regime."""
    random.seed(SEED)
    return [(round(statistics.mean(values), 4), round(statistics.pstdev(values), 4))
            for values in (draws(ref, task, regime) for task in ref.TASKS)]


def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def out_of_range(ref):
    """P(free draw outside [0, 1]), averaged over the five task baselines."""
    above = [1 - normal_cdf((1 - base - FREE_MEAN) / FREE_SD) for _name, base in ref.TASKS]
    below = [normal_cdf((0 - base - FREE_MEAN) / FREE_SD) for _name, base in ref.TASKS]
    return (round(sum(above) / len(above), 4), round(sum(below) / len(below), 4))


def sampled_out_of_range(ref, regime):
    random.seed(SEED + 1)
    values = [value for task in ref.TASKS for value in draws(ref, task, regime)]
    return round(sum(v > 1.0 or v < 0.0 for v in values) / len(values), 4)


def head_to_head(ref, count=DRAWS * 10):
    random.seed(SEED + 2)
    return round(sum(random.gauss(0.15, 0.22) > random.random() * 0.25
                     for _ in range(count)) / count, 4)


def power_at(sample, gap, spreads):
    error = math.sqrt(sum(s ** 2 for s in spreads) / sample)
    return round(0.5 * (1 + math.erf((gap / error - ALPHA_Z) / math.sqrt(2))), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fixed, free = profile(ref, "fixed"), profile(ref, "free")
    spreads = (fixed[0][1], free[0][1])
    source = inspect.getsource(ref.solve)
    measured = [round(f[0] - x[0], 3) for x, f in zip(fixed, free)]
    free_out = out_of_range(ref)
    return {
        "gap": round(GAP, 3),
        "worst_gap_error": round(max(abs(value - GAP) for value in measured), 3),
        "measured_gaps": measured,
        "tasks": len(ref.TASKS),
        "spreads": list(spreads),
        "spread_ratio": round(spreads[1] / spreads[0], 1),
        "head_to_head": head_to_head(ref),
        "below_floor": round(normal_cdf(-FREE_MEAN / FREE_SD), 4),
        "free_out": free_out,
        "free_out_sampled": sampled_out_of_range(ref, "free"),
        "fixed_out_sampled": sampled_out_of_range(ref, "fixed"),
        "agent_uses": source.count("agent"),
        "records": len(ref.TASKS) * 3,
        "per_task": 3,
        "power": power_at(len(ref.TASKS) * 3, GAP, spreads),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the same 0.025 on every task, and 24.8% below the fixed floor",
            all([result["gap"] == 0.025, result["worst_gap_error"] <= 0.005,
                 result["tasks"] == 5, result["spread_ratio"] == 3.0,
                 0.53 <= result["head_to_head"] <= 0.56,
                 result["below_floor"] == 0.2477]),
            f"the regime term never touches base, so the mean gap is {result['gap']} on "
            f"every one of {result['tasks']} tasks -- measured "
            f"{result['measured_gaps']}, worst error {result['worst_gap_error']}; "
            f"spreads are "
            f"{result['spreads']}, a {result['spread_ratio']}x, so a free draw beats a "
            f"fixed one {result['head_to_head']:.1%} of the time and falls below the "
            f"fixed floor {result['below_floor']:.1%}",
        ),
        practice.Check(
            "FINDING: the free regime produces impossible scores",
            all([result["free_out"] == (0.024, 0.0132), result["fixed_out_sampled"] == 0.0,
                 abs(result["free_out_sampled"] - sum(result["free_out"])) <= 0.004]),
            f"{sum(result['free_out']):.1%} of free draws fall outside [0, 1] in closed "
            f"form -- {result['free_out'][0]:.1%} above and {result['free_out'][1]:.1%} "
            f"below, measured {result['free_out_sampled']:.1%} -- against "
            f"{result['fixed_out_sampled']:.1%} for fixed",
        ),
        practice.Check(
            "FINDING: the agent is a parameter solve never reads",
            result["agent_uses"] == 1,
            f"`agent` appears {result['agent_uses']} time in solve, in its own "
            "signature, so the three parallel researchers are one distribution with "
            "three labels",
        ),
        practice.Check(
            "FINDING: three draws cannot see the effect",
            all([result["records"] == 15, result["per_task"] == 3,
                 result["power"] <= 0.10]),
            f"run_regime posts {result['records']} records per regime, "
            f"{result['per_task']} per task; against a {result['gap']} gap at spreads "
            f"{result['spreads']} that is a power of {result['power']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
