"""Exercise 1 — the average is of a censored episode, not of the MDP.

    **Easy.** Implement the 4×4 GridWorld and random-policy rollout in
    `code/main.py`. Run 10,000 episodes. Report mean and std of return.
    Compare to the optimal return (-6).

Reading of the exercise: the lesson already ships the environment and the
rollout, so "implement" is read as *re-derive and then audit* -- `my_step` is
written from the five lines the lesson page prints, checked against the lesson's
own `step` on all 64 (state, action) pairs, and the 10,000 episodes are then run
through the reference's `rollout` so every number below is the lesson's. "Compare
to the optimal return" is read as comparing against both of the exact values this
MDP has in closed form, since a 10,000-sample mean can be checked rather than
merely reported.

**ANSWER: mean -58.2, std 46.0, against an optimal -6.** The uniform policy
costs 9.7x the optimal path, and -6 is confirmed by value iteration over the
lesson's own `step`, not taken on trust.

**FINDING: that mean is not converging to E[G].** `rollout`'s `max_steps=200`
censors the episode; the estimator's limit is the censored expectation -58.32,
while the MDP's own value is -416/7 = -59.4286 exactly. The 1.11 gap is bias,
not noise: 10,000 episodes buy a standard error of 0.45, so no sample size
closes it.

**FINDING: 2.5% of episodes hit the cap, and they carry all of it.** A censored
episode is recorded as -200 when its true return is unbounded below, which is
also why the sign of the bias is *up*.

**FINDING: the lesson's own stated range holds neither value.** Step 2 says
"around -60 to -80"; the exact answers are -59.43 and -58.32, both above -60.

**FINDING: mean and std describe a shape that is neither symmetric nor wide
enough to need them.** The median is -44 against a mean of -58, and the +/- 1
sigma band spans 47% of the entire achievable range [-200, -6], so the two
numbers the exercise asks for locate the answer barely at all.

Structure: `my_step` is the re-derivation; `censored_value` is the exact
finite-horizon backup the cap actually implements; `optimal_value` is value
iteration for the -6 the exercise asks us to compare against.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "01-mdps-states-actions-rewards"
EPISODES, CAP, SEED = 10_000, 200, 20260914
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def my_step(state, action):
    """The environment re-derived from the five lines the lesson page prints."""
    if state == (3, 3):
        return state, 0.0, True
    dr, dc = MOVES[action]
    nxt = (min(max(state[0] + dr, 0), 3), min(max(state[1] + dc, 0), 3))
    return nxt, -1.0, nxt == (3, 3)


def backup(ref, values, state):
    """One uniform-policy Bellman backup at `state`, gamma = 1."""
    total = 0.0
    for action, prob in ref.uniform_policy(state).items():
        nxt, reward, done = ref.step(state, action)
        total += prob * (reward + (0.0 if done else values[nxt]))
    return total


def sweep(ref, values, optimal=False):
    """One synchronous sweep: greedy backup if `optimal`, else uniform-policy."""
    out = {}
    for state in ref.all_states():
        if state == ref.TERMINAL:
            out[state] = 0.0
        elif optimal:
            out[state] = max(reward + (0.0 if done else values[nxt])
                             for nxt, reward, done in
                             (ref.step(state, a) for a in ref.ACTIONS))
        else:
            out[state] = backup(ref, values, state)
    return out


def iterate(ref, sweeps, optimal=False):
    """`sweeps` synchronous sweeps from zero; returns V at the start state."""
    values = {s: 0.0 for s in ref.all_states()}
    for _ in range(sweeps):
        values = sweep(ref, values, optimal)
    return values[(0, 0)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(SEED)
    returns = [ref.rollout(ref.uniform_policy, rng, max_steps=CAP)[0]
               for _ in range(EPISODES)]
    exact = ref.policy_evaluation(ref.uniform_policy, gamma=1.0, tol=1e-12, max_iter=20_000)
    return {
        "mean": statistics.fmean(returns),
        "std": statistics.pstdev(returns),
        "median": statistics.median(returns),
        "censored": sum(1 for g in returns if g <= -CAP),
        "exact": exact[(0, 0)],
        "exact_censored": iterate(ref, CAP),
        "optimal": iterate(ref, 64, optimal=True),
        "agrees": all(my_step(s, a) == ref.step(s, a)
                      for s in ref.all_states() for a in ref.ACTIONS),
    }


def verify(result):
    mean, std, optimal = result["mean"], result["std"], result["optimal"]
    exact, censored = result["exact"], result["exact_censored"]
    stderr = std / EPISODES**0.5
    rate = result["censored"] / EPISODES
    return [
        practice.Check(
            "ANSWER: mean -58.2, std 46.0, which is 9.7x the optimal -6",
            -60 < mean < -56 and 40 < std < 50 and optimal == -6.0 and result["agrees"],
            f"over {EPISODES:,} episodes the uniform policy returns {mean:.2f} +/- {std:.2f} "
            f"against an optimal {optimal:.0f}, a factor of {mean / optimal:.1f}. The optimal "
            "is value iteration over the lesson's own step(), and the environment re-derived "
            "here agrees with the lesson's on all 64 (state, action) pairs",
        ),
        practice.Check(
            "FINDING: the mean converges to the censored value, not to E[G]",
            abs(mean - censored) < 4 * stderr and abs(censored - exact) > 1.0,
            f"rollout()'s max_steps={CAP} cuts the episode, so the estimator's limit is the "
            f"censored expectation {censored:.4f} and not the MDP's own value {exact:.4f} = "
            f"-416/7. The measured {mean:.2f} sits {abs(mean - censored) / stderr:.1f} standard "
            f"errors from the former and {abs(mean - exact) / stderr:.1f} from the latter",
        ),
        practice.Check(
            "FINDING: the gap is bias, and the ~2% of capped episodes carry all of it",
            0.01 < rate < 0.04 and abs(censored - exact) > 2 * stderr,
            f"{result['censored']} of {EPISODES:,} episodes ({100 * rate:.1f}%) reach the cap "
            f"and are recorded as {-CAP} when their true return is unbounded below, which is why "
            f"the bias points up. It is {abs(censored - exact):.2f}, or "
            f"{abs(censored - exact) / stderr:.1f} standard errors at this sample size, so more "
            "episodes shrink the noise and leave the bias exactly where it is",
        ),
        practice.Check(
            "FINDING: Step 2's stated range holds neither exact value",
            exact > -60 and censored > -60,
            f"the lesson says the random policy averages 'around -60 to -80 for this 4x4 board'. "
            f"The uncensored answer is {exact:.2f} and the censored one {censored:.2f}; both are "
            "above the top of that range, and the window excludes the only two numbers this "
            "MDP can produce",
        ),
        practice.Check(
            "FINDING: mean and std summarise a skewed distribution that fills its own range",
            result["median"] > mean + 10 and 2 * std > 0.4 * (CAP + optimal),
            f"the median is {result['median']:.0f} against a mean of {mean:.2f}, so the return "
            f"distribution is skewed, and the +/- 1 sigma band [{mean - std:.0f}, {mean + std:.0f}] "
            f"spans {200 * std / (CAP + optimal):.0f}% of the whole achievable range "
            f"[{-CAP}, {optimal:.0f}]. The two numbers the exercise asks for are a symmetric "
            "summary of an asymmetric, hard-clipped shape, and they locate it barely at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
