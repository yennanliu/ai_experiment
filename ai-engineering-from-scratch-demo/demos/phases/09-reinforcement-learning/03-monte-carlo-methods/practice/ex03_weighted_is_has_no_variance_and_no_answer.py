"""Exercise 3 — weighted IS has no variance, and 45% of the time no answer.

    **Hard.** Implement *off-policy* MC with importance sampling: collect data
    under uniform-random policy `μ`, estimate `V^π` for the deterministic optimal
    policy `π`. Compare plain IS vs per-decision IS vs weighted IS. Which has
    lowest variance?

Reading of the exercise: "which has lowest variance" is a question about the
spread of an estimator across runs, so all three are run on 20 independent batches
of 4,000 episodes from the lesson's own `rollout` under `uniform_policy`, and the
comparison is the spread of those 20 numbers. `π` is value iteration over the
lesson's own `step`; the target is `V^π(0,0) = -5.851985`, known exactly, so
"lowest variance" can be separated from "closest to right".

**ANSWER: weighted IS, and not by a margin -- by a kind.** Standard deviations
across the 20 batches are 5.22 (plain), 1.22 (per-decision) and **exactly 0**
(weighted), which returns the same -5.851985 every time.

**FINDING: that zero is degeneracy, not quality.** `π` is deterministic and so is
the board, so exactly one trajectory has nonzero weight -- the 6-step optimal
path, `ρ = 4^6 = 4096` -- and every batch that sees it sees the same return.
Weighted IS is averaging one distinct number.

**FINDING: the price is that it has no answer at all 45% of the time.** 9 of 20
batches contained no matching trajectory, against a predicted 37.6%; there the
denominator is 0 and the estimator is undefined. "Which has lowest variance"
ranks only the runs in which an estimate exists.

**FINDING: plain IS's worst failure is not noise, it is a confident 0.** Its
maximum across 20 batches is exactly 0.0000 -- outside the range of any return
this MDP can produce -- returned on the same batches where weighted IS abstains.

**MECHANISM: per-decision IS's 4.3x reduction comes from weighting rewards, not
trajectories.** `ρ_{0:t}` is `4^(t+1)` only while the prefix still matches, so a
trajectory that follows `π` for two steps and then leaves still contributes its
first two rewards: 25% of trajectories contribute something, against 0.02% for
plain IS, and the 4096x weight ever lands on one reward.

Structure: `batch` is the one pass that accumulates all three estimators from the
same trajectories, so the comparison isolates the estimator.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "03-monte-carlo-methods"
GAMMA, EPISODES, BATCHES, SEED = 0.99, 4_000, 20, 20260914


def look(ref, state, action, values):
    """One step of the lesson's own `step`, backed up."""
    nxt, reward, done = ref.step(state, action)
    return reward + (0.0 if done else GAMMA * values[nxt])


def optimal(ref):
    """`V*` and its greedy policy, from the lesson's own `step`."""
    values = {s: 0.0 for s in ref.states()}
    for _ in range(5_000):
        nxt = {s: (0.0 if s == ref.TERMINAL else
                   max(look(ref, s, a, values) for a in ref.ACTIONS)) for s in ref.states()}
        moved = max(abs(nxt[s] - values[s]) for s in values)
        values = nxt
        if moved < 1e-14:
            break
    return values, greedy(ref, values)


def greedy(ref, values):
    """The policy that is greedy with respect to `values`."""
    return {s: max(ref.ACTIONS, key=lambda a: look(ref, s, a, values)) for s in ref.states()}


def batch(ref, target, rng, episodes=EPISODES):
    """All three estimators from one set of `episodes` trajectories under mu."""
    plain = per_decision = numerator = denominator = 0.0
    touched = 0
    for _ in range(episodes):
        trajectory = ref.rollout(ref.uniform_policy, rng)
        ratio, decision, total = 1.0, 0.0, 0.0
        for step, (state, action, reward) in enumerate(trajectory):
            ratio = ratio * 4.0 if action == target[state] else 0.0
            decision += GAMMA**step * ratio * reward
            total += GAMMA**step * reward
        plain += ratio * total
        per_decision += decision
        numerator += ratio * total
        denominator += ratio
        touched += decision != 0.0
    return {"plain": plain / episodes, "per_decision": per_decision / episodes,
            "weighted": numerator / denominator if denominator else None,
            "touched": touched / episodes, "hits": denominator / 4096}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    star, target = optimal(ref)
    rows = [batch(ref, target, random.Random(SEED + i)) for i in range(BATCHES)]
    return {"rows": rows, "truth": star[(0, 0)],
            "defined": [r["weighted"] for r in rows if r["weighted"] is not None]}


def spread(rows, key):
    """(standard deviation, mean) of one estimator across the batches."""
    values = [r[key] for r in rows]
    return statistics.pstdev(values), statistics.fmean(values)


def degenerate(defined, truth):
    """Every defined estimate is one number, and that number is the answer."""
    return len(set(f"{w:.12f}" for w in defined)) == 1 and abs(defined[0] - truth) < 1e-9


def confident_zero(rows, blank):
    """Plain IS tops out at exactly 0, on exactly the batches weighted IS abstains from."""
    return (max(r["plain"] for r in rows) == 0.0
            and sum(1 for r in rows if r["plain"] == 0.0) == blank)


def verify(result):
    rows, truth, defined = result["rows"], result["truth"], result["defined"]
    plain, decision = spread(rows, "plain"), spread(rows, "per_decision")
    blank = BATCHES - len(defined)
    predicted = (1 - 4.0**-6) ** EPISODES
    touched = statistics.fmean(r["touched"] for r in rows)
    matched = statistics.fmean(r["hits"] for r in rows) / EPISODES
    return [
        practice.Check(
            "ANSWER: weighted IS, with a standard deviation of exactly zero",
            statistics.pstdev(defined) == 0.0 < decision[0] < plain[0],
            f"over {BATCHES} batches of {EPISODES:,} episodes, standard deviation across batches "
            f"is {plain[0]:.3f} for plain IS, {decision[0]:.3f} for per-decision and "
            f"{statistics.pstdev(defined):.4f} for weighted, which returns {defined[0]:.6f} every "
            f"time against a true V^pi(0,0) of {truth:.6f}. Means are {plain[1]:+.3f} and "
            f"{decision[1]:+.3f}, both consistent with unbiased at this batch count",
        ),
        practice.Check(
            "FINDING: the zero is degeneracy, not quality",
            degenerate(defined, truth),
            f"pi is deterministic and so is the board, so exactly one trajectory carries nonzero "
            f"weight -- the 6-step optimal path at rho = 4^6 = 4096 -- and every batch that sees "
            f"it sees the same return. All {len(defined)} defined estimates are the same number "
            f"to 1e-12, equal to the truth to {abs(defined[0] - truth):.1e}. Weighted IS is "
            "averaging one distinct value, which is why its spread is zero rather than small",
        ),
        practice.Check(
            "FINDING: the price is having no answer at all, 45% of the time",
            min(blank - BATCHES // 4, 0.2 - abs(blank / BATCHES - predicted)) > 0,
            f"{blank} of {BATCHES} batches contained no matching trajectory, against a predicted "
            f"(1 - 4^-6)^{EPISODES:,} = {predicted:.3f}: the denominator is 0 and the estimator "
            f"is undefined. Only {100 * matched:.3f}% of trajectories match pi at all, so 'which "
            "has lowest variance' is ranking the estimator that declines to answer nearly half "
            "the time, and the comparison is over the surviving batches",
        ),
        practice.Check(
            "FINDING: plain IS's worst failure is a confident zero, not noise",
            confident_zero(rows, blank),
            f"its maximum over the {BATCHES} batches is exactly "
            f"{max(r['plain'] for r in rows):.4f}, on the same {blank} batches where weighted IS "
            f"abstains -- an estimate outside [-100, {truth:.2f}], the range of every return this "
            f"MDP can produce. Its spread of {plain[0]:.2f} is {100 * plain[0] / abs(truth):.0f}% "
            "of the quantity being estimated, and the empty batches are most of it",
        ),
        practice.Check(
            "MECHANISM: per-decision IS weights rewards, not whole trajectories",
            0.2 < touched < 0.3 < plain[0] / decision[0],
            f"rho_0:t is 4^(t+1) only while the prefix still matches, so a trajectory that "
            f"follows pi for two steps and then leaves still contributes its first two rewards. "
            f"{100 * touched:.1f}% of trajectories contribute something, against "
            f"{100 * matched:.3f}% for plain IS, and the full 4096x weight is only ever applied "
            f"to the last reward of a complete match. That buys "
            f"{plain[0] / decision[0]:.1f}x less spread and a range that never includes 0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
