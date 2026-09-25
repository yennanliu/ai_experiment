"""Exercise 2 — sampling four agents costs more than computing them exactly.

    Implement Shapley *sampling* (Monte Carlo over K orderings). How does K
    affect approximation accuracy? Compare to exact for N=4.

Reading of the exercise: the reference already ships `shapley_sampled`, so the
work is measuring it -- error against exact over 200 seeds at each K -- and
putting that error next to what each K costs in value-function calls, since
the only reason to sample is to spend fewer of them.

**ANSWER: error falls as 1/sqrt(K), and at N=4 no K is worth paying for.** On
a 4-agent game with weights 4, 3, 2, 1 and v(S) = (sum of weights in S)^2 / 100,
the mean worst-agent error is 0.0604 at K=10, 0.0197 at K=100 and 0.0063 at
K=1000; K=100 against K=1000 is a 3.1x ratio against sqrt(10) = 3.16. But
sampling costs K(N+1) calls: K=24 already equals exact enumeration's 120, and
still misses by 0.0402.

**FINDING: exact needs only 16 values.** `shapley_exact` walks 24 orders and
calls the value function 120 times, but there are only 2^4 = 16 coalitions.
The subset form of Shapley -- the weighted marginals over each coalition --
reads each coalition once and matches the reference to 1e-12. Sampling
already costs more than that at K=4 (20 calls).

**FINDING: 24 orders sampled with replacement is not the 24 orders.** Drawing
K=24 orders, the reference sees 15.3 distinct orders on average over 200
seeds. Drawing the same 24 without replacement is exact, error 0.

**FINDING: the efficiency check cannot see sampling error.** Every sampled
allocation sums to v(grand) = 1.0 to 1e-12 at every K, because each ordering
telescopes to v(grand) - v(empty). Exercise 1's "confirm they sum to total
value" passes for an estimate that is 0.06 off.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "21-agent-economies"
WEIGHTS = {"a": 4, "b": 3, "c": 2, "d": 1}
AGENTS, SEEDS, KS = list(WEIGHTS), 200, (4, 10, 24, 100, 1000)


def value(coalition):
    return sum(WEIGHTS[a] for a in coalition) ** 2 / 100


def subset_shapley(value_fn, agents):
    """Shapley by coalitions: each of the 2^n values is read exactly once."""
    n, cache = len(agents), {}
    for size in range(n + 1):
        for combo in itertools.combinations(agents, size):
            cache[frozenset(combo)] = value_fn(frozenset(combo))
    weight = lambda s: math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)  # noqa: E731
    return {a: sum(weight(len(s)) * (cache[s | {a}] - v) for s, v in cache.items() if a not in s)
            for a in agents}, len(cache)


def worst(estimate, exact):
    return max(abs(estimate[a] - exact[a]) for a in exact)


def sweep(ref, exact):
    errors, sums, distinct = {}, [], []
    for k in KS:
        runs = []
        for seed in range(SEEDS):
            est = ref.shapley_sampled(value, AGENTS, k, random.Random(seed))
            runs.append(worst(est, exact))
            sums.append(abs(sum(est.values()) - 1.0))
        errors[k] = statistics.mean(runs)
    for seed in range(SEEDS):
        rng = random.Random(seed)
        distinct.append(len({tuple(rng.sample(AGENTS, 4)) for _ in range(24)}))
    return errors, max(sums), statistics.mean(distinct)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    calls = {"n": 0}
    counted = lambda s: calls.__setitem__("n", calls["n"] + 1) or value(s)  # noqa: E731
    exact = ref.shapley_exact(counted, AGENTS)
    subset, coalitions = subset_shapley(value, AGENTS)
    errors, sum_err, distinct = sweep(ref, exact)
    everything = list(itertools.permutations(AGENTS))
    without = {a: 0.0 for a in AGENTS}
    for order in everything:
        for i, a in enumerate(order):
            without[a] += (value(order[:i + 1]) - value(order[:i])) / len(everything)
    return {
        "errors": errors, "exact_calls": calls["n"], "coalitions": coalitions,
        "subset_err": worst(subset, exact), "sum_err": sum_err, "distinct": distinct,
        "without_err": worst(without, exact), "cost": {k: k * 5 for k in KS},
    }


def verify(result):
    err, cost = result["errors"], result["cost"]
    ratio = err[100] / err[1000]
    return [
        practice.Check(
            "ANSWER: error falls as 1/sqrt(K), and at N=4 no K is worth paying for",
            all([err[10] > err[100] > err[1000], 2.5 < ratio < 4.0,
                 cost[24] == result["exact_calls"] == 120, err[24] > 0.02]),
            f"mean worst-agent error {err[10]:.4f}, {err[100]:.4f}, {err[1000]:.4f} at "
            f"K=10, 100, 1000 (ratio {ratio:.2f} vs sqrt(10)=3.16); K=24 costs "
            f"{cost[24]} calls, equal to exact's {result['exact_calls']}, and misses by "
            f"{err[24]:.4f}",
        ),
        practice.Check(
            "FINDING: exact needs only 16 values",
            result["coalitions"] == 16 and result["subset_err"] < 1e-12 and cost[4] > 16,
            f"the subset form reads {result['coalitions']} coalitions once each and "
            f"matches shapley_exact to {result['subset_err']:.0e}; sampling costs "
            f"{cost[4]} calls already at K=4",
        ),
        practice.Check(
            "FINDING: 24 orders sampled with replacement is not the 24 orders",
            result["distinct"] < 17 and result["without_err"] < 1e-12,
            f"K=24 draws see {result['distinct']:.1f} distinct orders on average; all 24 "
            f"without replacement is exact (error {result['without_err']:.0e})",
        ),
        practice.Check(
            "FINDING: the efficiency check cannot see sampling error",
            result["sum_err"] < 1e-12 and err[10] > 0.05,
            f"every sampled allocation sums to v(grand) within {result['sum_err']:.0e} "
            f"while the K=10 estimate is {err[10]:.4f} off -- each order telescopes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
