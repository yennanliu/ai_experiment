"""Exercise 3 — the higher gate blocks the wrong agent in one row and the right agent in another.

    Add a quality gate to AMRO-S: pheromone deposit only on runs with eval score
    > 0.7. How does this change convergence vs the un-gated version?

Reading of the exercise: the shipped router is already gated, at 0.6, so the
comparison is three-way -- ungated (0.0), shipped (0.6) and the exercise's
0.7 -- and "convergence" is the share of pheromone on each task type's best
agent after 200 tasks, which is the probability the router picks it. 40 seeds;
the gate is set by wrapping `PheromoneRouter`, so `run_amro_s` runs unmodified.

**ANSWER: against ungated, any gate converges faster; 0.7 against 0.6 helps
one row and hurts another.** Mean best-agent share, ungated / 0.6 / 0.7:
code 0.80 / 0.95 / 0.95, math 0.73 / 0.93 / 0.95, writing 0.84 / 0.95 / 0.95,
planning 0.49 / 0.74 / 0.62. Average routed quality is 0.617, 0.670 and 0.668.
Ungated, every run of every agent deposits, so wrong agents are reinforced
too. Moving to 0.7 stops the coder's occasional math deposits (range
0.35-0.65), which helps math, and stops 5 in 6 of the writer's planning
deposits, which hurts planning more than math gains.

**FINDING: a gate only acts on a row where some agent's quality range
straddles it.** `simulate_task` adds uniform noise of +/-0.15 to a fixed
affinity. The coder scores 0.75-1.0 on code, and every other agent's range
tops out below 0.6, so 0.6 and 0.7 give code and writing *identical* shares.
Planning's best agent is the writer at 0.6, range 0.45-0.75: at 0.7 only
1 in 6 of its runs deposits, so the row learns slower. Above 0.75 no
planning run can ever deposit and the row stays at exactly 1/3 each.

**FINDING: evaporation without a deposit is a no-op on routing.** `deposit`
multiplies the whole row by 0.95 whether or not the gate passes, and
`choose` normalises, so a gated-out run leaves every choice probability
unchanged to the last digit. "Decays over time" does not forget anything:
a frozen row keeps its ratios forever, and decay only matters as the weight
of old deposits against new ones.

Structure: `run()` wraps `PheromoneRouter` with a threshold and reads the
final table; `evaporate_only()` probes one gated-out deposit.
"""

from __future__ import annotations

import functools
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "19-swarm-optimization-pso-aco"
BEST = {"code": "coder", "math": "mathematician", "writing": "writer", "planning": "writer"}
GATES, SEEDS = (0.0, 0.6, 0.7), range(40)


def run(ref, gate, seed):
    """(ACO quality, best-agent share per task type) with the router gated at `gate`."""
    original = ref.PheromoneRouter
    ref.PheromoneRouter = functools.partial(original, quality_threshold=gate)
    try:
        _, quality, router = ref.run_amro_s(200, seed)
    finally:
        ref.PheromoneRouter = original
    table = router.pheromones
    return quality, {t: table[t][a] / sum(table[t].values()) for t, a in BEST.items()}


def evaporate_only(ref):
    router = ref.PheromoneRouter(["t"], ["a", "b", "c"])
    router.pheromones["t"] = {"a": 2.0, "b": 1.0, "c": 0.5}
    before = [v / 3.5 for v in router.pheromones["t"].values()]
    router.deposit("t", "a", 0.1)
    total = sum(router.pheromones["t"].values())
    return max(abs(b - v / total) for b, v in zip(before, router.pheromones["t"].values()))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    share, quality = {}, {}
    for gate in GATES + (0.76,):
        runs = [run(ref, gate, s) for s in SEEDS]
        quality[gate] = statistics.mean(q for q, _ in runs)
        share[gate] = {t: statistics.mean(s[t] for _, s in runs) for t in BEST}
    return {"share": share, "quality": quality, "drift": evaporate_only(ref),
            "ranges": {a: (round(v["planning"] - 0.15, 2), round(v["planning"] + 0.15, 2))
                       for a, v in ref.AGENT_TASK_AFFINITY.items()}}


def verify(result):
    s, q = result["share"], result["quality"]
    fmt = {t: [round(s[g][t], 2) for g in GATES] for t in BEST}
    return [
        practice.Check(
            "ANSWER: any gate beats ungated; 0.7 against 0.6 helps one row and hurts another",
            all([all(s[0.6][t] > s[0.0][t] for t in BEST), q[0.6] > q[0.0] + 0.04,
                 s[0.7]["math"] > s[0.6]["math"], q[0.7] < q[0.6],
                 s[0.7]["planning"] < s[0.6]["planning"] - 0.1]),
            f"best-agent share ungated/0.6/0.7: {fmt}; routed quality "
            f"{[round(q[g], 3) for g in GATES]}",
        ),
        practice.Check(
            "FINDING: a gate only acts on a row where some agent's range straddles it",
            all([s[0.6]["code"] == s[0.7]["code"], s[0.6]["writing"] == s[0.7]["writing"],
                 abs(s[0.76]["planning"] - 1 / 3) < 1e-12]),
            f"code and writing shares are identical at 0.6 and 0.7; planning ranges are "
            f"{result['ranges']}, so at 0.7 only 1 in 6 writer runs deposits, and at 0.76 "
            f"the planning share stays at {s[0.76]['planning']:.4f}",
        ),
        practice.Check(
            "FINDING: evaporation without a deposit is a no-op on routing",
            result["drift"] < 1e-15,
            f"a gated-out deposit scales the row by 0.95 and moves the normalised choice "
            f"probabilities by {result['drift']:.1e} -- a frozen row keeps its ratios forever",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
