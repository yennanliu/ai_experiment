"""Exercise 1 — neither, and the curve does not belong to either of them.

    **Easy.** Implement Q-learning and SARSA on the 4×4 GridWorld. Plot learning
    curves (mean return per 100 episodes) for 2,000 episodes. Who converges
    faster?

Reading of the exercise: the lesson ships both algorithms and `block_means`, so
"implement" is read as *run them and settle the comparison* -- twelve paired seeds
rather than the one the lesson's `main` uses, because "who converges faster" is a
claim about a difference and a single pair of runs cannot support one. Convergence
is timed as the first block within 1.0 of the limit that the curve never leaves
again.

**ANSWER: neither, measurably.** Both reach the limit at episode 300 on all 12
seeds, and the paired difference in final performance is -0.004 ± 0.007 -- 0.7
standard errors from zero.

**FINDING: the limit is not either algorithm's answer, it is `ε`'s.** The exact
undiscounted return of an `ε`-greedy policy over the optimal action is **-6.6087**
at `ε = 0.1`; the two land on -6.614 and -6.610. The curve measures the behaviour
policy both share.

**FINDING: they both learn the optimal policy exactly, which the curve also does
not show.** The greedy policy each recovers is worth `-5.8520` from the start, the
optimum to 0.0e+00 -- 0.76 better than the curve's plateau, and the gap is the
exploration neither algorithm is being asked to stop doing.

**FINDING: the plotted quantity is not the optimized one.** `total += r` sums raw
rewards while every update discounts at `γ = 0.99`. The curve is undiscounted
return; the algorithms minimize discounted return, and the two limits are -6.6087
and -6.42.

**FINDING: the comparison needs a board this one is not.** Q-learning learns `Q*`
off-policy and SARSA learns `Q^ε`, and here that difference is invisible because
no state punishes the `ε`-random action. Exercise 2 is where it becomes 28 points.

Structure: `ceiling` sweeps the lesson's own `step` for the exact ε-soft return;
`converged` reads the first block the curve never leaves.
"""

from __future__ import annotations

import statistics

import random

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "04-q-learning-sarsa"
GAMMA, EPSILON, EPISODES, BLOCK, SEEDS = 0.99, 0.1, 2_000, 100, 12
CELLS = [(r, c) for r in range(4) for c in range(4)]


def look(ref, state, action, values, gamma):
    nxt, reward, done = ref.step(state, action)
    return reward + (0.0 if done else gamma * values[nxt])


def fixpoint(ref, backup, tol=1e-12):
    values = {s: 0.0 for s in CELLS}
    for _ in range(500_000):
        nxt = {s: (0.0 if s == ref.TERMINAL else backup(s, values)) for s in CELLS}
        moved = max(abs(nxt[s] - values[s]) for s in values)
        values = nxt
        if moved < tol:
            break
    return values


def optimal(ref):
    """`V*` at gamma and the greedy policy it induces."""
    values = fixpoint(ref, lambda s, v: max(look(ref, s, a, v, GAMMA) for a in ref.ACTIONS))
    return values, {s: max(ref.ACTIONS, key=lambda a: look(ref, s, a, values, GAMMA))
                    for s in CELLS}


def ceiling(ref, greedy, eps, gamma):
    """Exact return of the eps-soft policy over `greedy` -- undiscounted at gamma = 1."""
    return fixpoint(ref, lambda s, v: sum(
        (eps / 4 + (1 - eps if a == greedy[s] else 0.0)) * look(ref, s, a, v, gamma)
        for a in ref.ACTIONS))


def converged(blocks, limit, tol=1.0):
    """The first block within `tol` of `limit` that the curve never leaves again."""
    for i, _value in enumerate(blocks):
        if all(abs(later - limit) < tol for later in blocks[i:]):
            return (i + 1) * BLOCK
    return None


def run(ref, learner, seed, limit):
    table, returns = learner(EPISODES, gamma=GAMMA, epsilon=EPSILON,
                             rng=random.Random(seed))
    greedy = {s: max(ref.ACTIONS, key=lambda a: table[s][a]) if s in table else "up"
              for s in CELLS}
    return {"blocks": ref.block_means(returns, BLOCK), "tail": statistics.fmean(returns[-500:]),
            "at": converged(ref.block_means(returns, BLOCK), limit), "greedy": greedy}


def learned_values(ref, runs):
    """The exact start-state value of every greedy policy the runs recovered."""
    return [fixpoint(ref, lambda s, v, g=row["greedy"]: look(ref, s, g[s], v, GAMMA))[(0, 0)]
            for rows in runs.values() for row in rows]


def summarise(runs, ref, star):
    """The per-algorithm aggregates the checks read."""
    return {"at": {n: sorted({r["at"] for r in rows}) for n, rows in runs.items()},
            "tail": {n: statistics.fmean(r["tail"] for r in rows) for n, rows in runs.items()},
            "paired": [a["tail"] - b["tail"] for a, b in zip(runs["sarsa"], runs["q"])],
            "worst": max(abs(v - star) for v in learned_values(ref, runs))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    star, best = optimal(ref)
    limit = ceiling(ref, best, EPSILON, 1.0)[(0, 0)]
    runs = {name: [run(ref, learner, seed, limit) for seed in range(SEEDS)]
            for name, learner in (("sarsa", ref.sarsa), ("q", ref.q_learning))}
    return {**summarise(runs, ref, star[(0, 0)]), "limit": limit, "star": star[(0, 0)],
            "discounted": ceiling(ref, best, EPSILON, GAMMA)[(0, 0)]}


def verify(result):
    at, tail, limit, star = result["at"], result["tail"], result["limit"], result["star"]
    mean = statistics.fmean(result["paired"])
    stderr = statistics.pstdev(result["paired"]) / SEEDS**0.5
    off = max(abs(tail["sarsa"] - limit), abs(tail["q"] - limit))
    apart = abs(tail["sarsa"] - tail["q"])
    return [
        practice.Check(
            "ANSWER: neither -- both reach the limit at episode 300 on all 12 seeds",
            at["sarsa"] == at["q"] == [300] and abs(mean) < 2 * stderr,
            f"across {SEEDS} paired seeds the first block within 1.0 of the limit that the curve "
            f"never leaves is episode {at['q'][0]} for both, on every seed. Final performance is "
            f"{tail['sarsa']:.3f} against {tail['q']:.3f}, a paired difference of {mean:+.4f} "
            f"+/- {stderr:.4f} -- {abs(mean) / stderr:.1f} standard errors from zero",
        ),
        practice.Check(
            "FINDING: the limit belongs to epsilon, not to either algorithm",
            off < 0.05,
            f"the exact undiscounted return of an eps-soft policy over the optimal action is "
            f"{limit:.4f} at eps={EPSILON}, swept from the lesson's own step(). The two curves "
            f"settle at {tail['sarsa']:.3f} and {tail['q']:.3f}, within {off:.3f} of it. The plot "
            "measures the behaviour policy the two share, and that policy is not learned",
        ),
        practice.Check(
            "FINDING: both learn the optimal policy exactly, and the curve cannot say so",
            result["worst"] < 1e-9,
            f"the greedy policy recovered on all {2 * SEEDS} runs is worth exactly {star:.4f} "
            f"from the start state -- the optimum, to {result['worst']:.1e}. The plateau the "
            f"curve shows is {abs(limit - star):.2f} worse than that, and the difference is "
            "exploration, which neither algorithm is being asked to stop",
        ),
        practice.Check(
            "FINDING: the plotted quantity is not the optimized one",
            abs(limit - result["discounted"]) > 0.15,
            f"`total += r` sums raw rewards while every update discounts at gamma={GAMMA}. The "
            f"undiscounted eps-soft return is {limit:.4f} and the discounted one "
            f"{result['discounted']:.4f}, {abs(limit - result['discounted']):.3f} apart. The "
            "curve is a quantity neither algorithm is minimising, survivable here and not on a "
            "board where the discount changes the ranking",
        ),
        practice.Check(
            "FINDING: this board cannot separate on-policy from off-policy",
            apart < 0.05,
            f"Q-learning bootstraps from max_a Q and SARSA from the action it will take, so they "
            f"converge to different Q. The curves differ by {apart:.3f} because no state here "
            f"punishes the eps-random action: the worst an exploratory step costs is one wasted "
            "move. Exercise 2 adds a cliff and the same difference becomes 29 points",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
