"""Exercise 5 — a hundred samples cost more calls than exact Shapley over ten agents.

    Read the AAMAS 2025 decentralized LaMAS paper. Implement their Shapley
    step over 10 agents on a synthetic task. How long does exact computation
    take? How close does sampling get with 100 draws?

Reading of the exercise: the paper (Yang et al., "Unlocking the Potential of
Decentralized LLM-based MAS", AAMAS 2025 Blue Sky Ideas track, 5 pages)
specifies no Shapley step -- it says once that "Attribution methods, such as
the Shapley Value, ensure profits are allocated based on each agent's
contribution" and cites Shapley 1953 -- so the step implemented is the
textbook one, on a task whose exact answer has a closed form: 10 agents
covering 20 skills, v(S) = fraction of skills covered, where agent i's
Shapley value is the sum over its skills of 1/20k, k = agents holding it.

**ANSWER: the reference's exact method makes 39,916,800 value calls at N=10;
the subset form makes 1,024; 100 samples make 1,100 and miss by 17%.**
`shapley_exact` walks N! orders and calls v N+1 times per order. Timed at
N=7 (40,320 calls, about 0.02s) and scaled by the 990x call ratio, N=10
takes about 20 seconds; the subset form reads each of the 2^10 coalitions once and takes
milliseconds, matching the closed form to 1e-12. 100 sampled orders from
`shapley_sampled` miss by a mean worst-agent error of 0.0133 on values
averaging 0.0800 over 50 seeds. They pick the right top agent on all 50,
so the ranking survives and the amounts do not -- at more calls than exact.

**FINDING: the coverage game rewards a Sybil.** Cloning the agent with the
most shared skills as an eleventh agent -- same skills, same operator -- lifts
that operator's combined Shapley value by 42% (agent-5), taken from the honest
agents who shared those skills; the clone added nothing to v. Symmetry, the
axiom that makes Shapley fair, is what pays for duplicates. The lesson names
Sybil attacks as a failure mode; its credit rule is the mechanism.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "21-agent-economies"
SKILLS, N, DRAWS, SEEDS = 20, 10, 100, 50


def make_agents(seed=7):
    rng = random.Random(seed)
    return {f"agent-{i}": frozenset(rng.sample(range(SKILLS), rng.randint(3, 6))) for i in range(N)}


def game(agents):
    return lambda coalition: len(frozenset().union(*(agents[a] for a in coalition))) / SKILLS


def closed_form(agents):
    holders = {s: sum(s in skills for skills in agents.values()) for s in range(SKILLS)}
    return {a: sum(1 / (SKILLS * holders[s]) for s in skills) for a, skills in agents.items()}


def subset_shapley(value_fn, names):
    n, cache = len(names), {}
    for size in range(n + 1):
        for combo in itertools.combinations(names, size):
            cache[frozenset(combo)] = value_fn(frozenset(combo))
    weight = [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    return {a: sum(weight[len(s)] * (cache[s | {a}] - v) for s, v in cache.items() if a not in s)
            for a in names}, len(cache)


def worst(estimate, exact):
    return max(abs(estimate[a] - exact[a]) for a in exact)


def timed_reference(ref, agents, n=7):
    names, calls = list(agents)[:n], {"k": 0}
    sub = {a: agents[a] for a in names}
    value = game(sub)
    counted = lambda s: calls.__setitem__("k", calls["k"] + 1) or value(s)  # noqa: E731
    start = time.perf_counter()
    result = ref.shapley_exact(counted, names)
    return time.perf_counter() - start, calls["k"], worst(result, closed_form(sub))


def sybil_gain(agents, truth):
    holders = {s: sum(s in k for k in agents.values()) for s in range(SKILLS)}
    target = max(agents, key=lambda a: sum(holders[s] > 1 for s in agents[a]))
    cloned = dict(agents, clone=agents[target])
    after = closed_form(cloned)
    return target, (after[target] + after["clone"]) / truth[target] - 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    agents = make_agents()
    names, value, truth = list(agents), game(agents), closed_form(agents)
    start = time.perf_counter()
    subset, reads = subset_shapley(value, names)
    subset_time = time.perf_counter() - start
    t7, calls7, err7 = timed_reference(ref, agents)
    errors, top = [], 0
    for seed in range(SEEDS):
        est = ref.shapley_sampled(value, names, DRAWS, random.Random(seed))
        errors.append(worst(est, truth))
        top += max(est, key=est.get) == max(truth, key=truth.get)
    target, gain = sybil_gain(agents, truth)
    return {
        "calls10": math.factorial(N) * (N + 1), "calls7": calls7, "err7": err7,
        "t7": t7, "t10": t7 * math.factorial(N) * (N + 1) / calls7,
        "reads": reads, "subset_time": subset_time, "subset_err": worst(subset, truth),
        "draw_calls": DRAWS * (N + 1), "mean_err": statistics.mean(errors),
        "mean_value": statistics.mean(truth.values()), "top": top, "target": target,
        "gain": gain, "sybil_check": abs(sum(closed_form(dict(agents, clone=agents[target]))
                                             .values()) - value(frozenset(names))),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 39,916,800 calls exact, 1,024 by subsets, 1,100 for 100 samples",
            all([result["calls10"] == 39_916_800, result["calls7"] == 40_320,
                 result["err7"] < 1e-12, result["reads"] == 1024,
                 result["subset_err"] < 1e-12, result["draw_calls"] > result["reads"],
                 result["t10"] > 100 * result["subset_time"],
                 0 < result["mean_err"] < 0.25 * result["mean_value"]]),
            f"shapley_exact made {result['calls7']} calls in {result['t7']:.3f}s at N=7, so "
            f"N=10's {result['calls10']:,} take ~{result['t10']:.0f}s; the subset form reads "
            f"{result['reads']} coalitions in {result['subset_time'] * 1000:.1f}ms; 100 "
            f"samples cost {result['draw_calls']} calls and miss by {result['mean_err']:.4f} "
            f"on values averaging {result['mean_value']:.4f}, top agent right on "
            f"{result['top']}/{SEEDS} seeds",
        ),
        practice.Check(
            "FINDING: the coverage game rewards a Sybil",
            0.3 <= result["gain"] <= 0.6 and result["sybil_check"] < 1e-12,
            f"cloning {result['target']} lifts its operator's combined Shapley value by "
            f"{result['gain']:.0%} while v is unchanged -- symmetry pays for duplicates",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
