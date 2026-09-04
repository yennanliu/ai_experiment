"""Exercise 3 — simulated annealing on a landscape with many local minima.

    **Implement simulated annealing** using Metropolis-Hastings. Start at high
    temperature (accept almost everything) and gradually cool down (accept only
    improvements). Use it to find the minimum of a function with many local
    minima.

Reading of the exercise: two choices decide whether this measures anything.

The score is the chain's **final** state, not the best point it ever visited.
Tracking the best-ever turns a hot chain into random search with memory, and
measured that way a chain that never cools wins 98% of the time — which says
nothing about annealing, whose whole claim is that it *settles*.

And the comparison needs baselines. Annealing is run against a chain fixed hot, a
chain fixed cold, and greedy descent, 40 seeds each, from a start 8 units from
the global minimum of f(x) = x²/20 − cos(3x). Check 5 is the one that shows the
schedule is doing the work: cold and greedy come out numerically identical.
"""

from __future__ import annotations

import math
import random

from harness import practice

SEEDS, STEPS, START, STD = 40, 4_000, 8.0, 0.25
T_HIGH, T_LOW = 3.0, 0.01
TOLERANCE = 0.6


def energy(x):
    return x * x / 20.0 - math.cos(3 * x)


def chain(seed, schedule):
    """Metropolis-Hastings; the schedule supplies the temperature per step."""
    rng = random.Random(seed)
    x = START
    for step in range(STEPS):
        candidate = x + rng.gauss(0, STD)
        delta = energy(candidate) - energy(x)
        if delta < 0 or rng.random() < math.exp(-delta / max(schedule(step), 1e-12)):
            x = candidate
    return x


def greedy(seed):
    rng = random.Random(seed)
    x = START
    for _ in range(STEPS):
        candidate = x + rng.gauss(0, STD)
        if energy(candidate) < energy(x):
            x = candidate
    return x


def solve():
    geometric = lambda step: T_HIGH * (T_LOW / T_HIGH) ** (step / STEPS)
    strategies = {
        "annealed 3.0 -> 0.01": [chain(s, geometric) for s in range(SEEDS)],
        "fixed T = 3.0": [chain(s, lambda _: T_HIGH) for s in range(SEEDS)],
        "fixed T = 0.01": [chain(s, lambda _: T_LOW) for s in range(SEEDS)],
        "greedy descent": [greedy(s) for s in range(SEEDS)],
    }
    rows = {}
    for label, finals in strategies.items():
        rows[label] = {
            "rate": sum(1 for x in finals if abs(x) < TOLERANCE) / SEEDS,
            "mean_energy": sum(energy(x) for x in finals) / SEEDS,
        }
    return {"rows": rows, "start_energy": energy(START), "global_energy": energy(0.0)}


def verify(result):
    rows = result["rows"]
    annealed = rows["annealed 3.0 -> 0.01"]
    hot, cold = rows["fixed T = 3.0"], rows["fixed T = 0.01"]
    greedy_row = rows["greedy descent"]
    best = min(rows, key=lambda k: rows[k]["mean_energy"])
    return [
        practice.Check(f"annealing has the lowest mean final energy of the four",
                       best == "annealed 3.0 -> 0.01",
                       "; ".join(f"{k}: {v['mean_energy']:+.4f}" for k, v in rows.items())
                       + f" — the global minimum is {result['global_energy']:.3f} and the "
                         f"start is {result['start_energy']:.3f}"),
        practice.Check("…and reaches the global minimum most often",
                       annealed["rate"] > max(hot["rate"], cold["rate"],
                                              greedy_row["rate"]),
                       "; ".join(f"{k}: {v['rate']:.0%}" for k, v in rows.items())
                       + f" within {TOLERANCE} of x=0"),
        practice.Check("a chain that never cools explores but does not settle",
                       hot["mean_energy"] > annealed["mean_energy"]
                       and hot["rate"] > cold["rate"],
                       f"fixed T={T_HIGH:g} ends at mean energy "
                       f"{hot['mean_energy']:+.4f} against annealing's "
                       f"{annealed['mean_energy']:+.4f} — it visits good regions and then "
                       f"leaves them, which is why the *final* state is the honest score"),
        practice.Check("a chain that starts cold never leaves the first basin it finds",
                       cold["rate"] == 0.0 and greedy_row["rate"] == 0.0,
                       f"fixed T={T_LOW:g} and greedy both reach the global minimum "
                       f"{cold['rate']:.0%} of the time, ending near "
                       f"{cold['mean_energy']:+.3f} — barely better than the start's "
                       f"{result['start_energy']:.3f}"),
        practice.Check("…and they are numerically identical, which is the whole point",
                       abs(cold["mean_energy"] - greedy_row["mean_energy"]) < 0.01,
                       f"cold {cold['mean_energy']:+.4f} against greedy "
                       f"{greedy_row['mean_energy']:+.4f}. At T=0.01 the acceptance "
                       f"probability for any uphill step is effectively zero, so the "
                       f"Metropolis chain *is* greedy descent. Neither endpoint of the "
                       f"schedule works alone — the cooling is what does the work"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
