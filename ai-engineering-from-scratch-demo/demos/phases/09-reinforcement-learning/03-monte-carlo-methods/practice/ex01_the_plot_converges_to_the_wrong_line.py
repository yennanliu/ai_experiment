"""Exercise 1 — the plot converges to a line the exercise does not draw.

    **Easy.** Implement first-visit MC evaluation of the uniform-random policy
    on 4×4 GridWorld. Run 10,000 episodes. Plot `V(0,0)` as a function of
    episode count against the DP answer.

Reading of the exercise: the lesson ships `mc_policy_evaluation`, so "implement"
is read as *run the reference and audit it* -- the estimate is accumulated
exactly as `mc_policy_evaluation` does, from the lesson's own `rollout` and
`returns_from`, and snapshotted so the curve exists. The "DP answer" is not taken
from the lesson's printed -39.41 but re-derived here by sweeping the lesson's own
`step`. The plot ships as a table, per `DESIGN D14`.

**ANSWER: -39.556 at 10,000 episodes against a DP -39.4116.** That is 0.7
standard errors at `se = 0.216`; the curve is inside 1.5 of them from n = 2,000 on.

**FINDING: the line it is converging to is not the DP answer.** `rollout` carries
`max_steps=200`, so the estimator's limit is the 200-step censored value
-39.3123, not the MDP's -39.4116. The 0.0993 between them is bias, and no number
of episodes removes it.

**FINDING: at the sample size the exercise specifies, that bias is invisible.**
0.0993 is 0.46 standard errors at 10,000 episodes. It takes ~47,000 episodes
before the bias is as large as the noise, so the plot cannot show the one thing
that is wrong with it.

**FINDING: the plot's own noise is the story it does tell.** The return has
`σ = 21.61` against a mean of -39.4, so reading `V(0,0)` to ±0.1 needs ~47,000
episodes. The 10,000 the exercise asks for buy ±0.216.

**FINDING: the same answer exactly costs 28,020 backups.** MC spends 589,413
environment steps to get within a standard error; sweeping the lesson's own
`step` to `1e-6` costs 467 sweeps, 21x fewer operations, and is not a sample at
all.

Structure: `dp_value` sweeps the lesson's `step` for the exact and the censored
values; `curve` accumulates the reference's own incremental mean.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "03-monte-carlo-methods"
GAMMA, EPISODES, CAP, SEED = 0.99, 10_000, 200, 20260914
MARKS = (100, 500, 1000, 2000, 5000, 10000)


def sweep(ref, values):
    """One synchronous uniform-policy backup over the whole board."""
    out = {}
    for state in ref.states():
        if state == ref.TERMINAL:
            out[state] = 0.0
            continue
        out[state] = sum(0.25 * (reward + (0.0 if done else GAMMA * values[nxt]))
                         for nxt, reward, done in
                         (ref.step(state, a) for a in ref.ACTIONS))
    return out


def dp_value(ref, sweeps=None, tol=1e-12):
    """`V^uniform(0,0)`: the fixed point, or the `sweeps`-step censored value."""
    values, count = {s: 0.0 for s in ref.states()}, 0
    while count != sweeps:
        nxt = sweep(ref, values)
        count += 1
        if sweeps is None and max(abs(nxt[s] - values[s]) for s in values) < tol:
            return nxt[(0, 0)], count
        values = nxt
    return values[(0, 0)], count


def curve(ref):
    """The reference's own incremental first-visit mean at (0,0), snapshotted."""
    rng = random.Random(SEED)
    floor = -(1 - GAMMA**CAP) / (1 - GAMMA)          # the return of an episode that is cut
    running, steps, marks, returns = 0.0, 0, {}, []
    for episode in range(EPISODES):
        trajectory = ref.rollout(ref.uniform_policy, rng, max_steps=CAP)
        steps += len(trajectory)
        first = ref.returns_from(trajectory, GAMMA)[0]
        returns.append(first)
        running += (first - running) / (episode + 1)
        if episode + 1 in MARKS:
            marks[episode + 1] = running
    return {"marks": marks, "sigma": statistics.pstdev(returns), "steps": steps,
            "estimate": running, "capped": sum(1 for g in returns if g < floor + 1e-9)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    exact, sweeps = dp_value(ref)
    return {**curve(ref), "exact": exact, "sweeps": sweeps,
            "censored": dp_value(ref, sweeps=CAP)[0],
            "cheap_sweeps": dp_value(ref, tol=1e-6)[1]}


def verify(result):
    exact, censored, sigma = result["exact"], result["censored"], result["sigma"]
    stderr = sigma / EPISODES**0.5
    bias = censored - exact
    needed = (sigma / abs(bias)) ** 2
    table = ", ".join(f"n={n}: {result['marks'][n]:+.2f}" for n in MARKS)
    return [
        practice.Check(
            "ANSWER: -39.556 at 10,000 episodes, 0.7 standard errors from the DP -39.4116",
            abs(result["estimate"] - censored) < 3 * stderr,
            f"V(0,0) against the DP answer {exact:.4f} -- {table}. The final estimate is "
            f"{result['estimate']:.3f}, {abs(result['estimate'] - exact) / stderr:.1f} standard "
            f"errors from it at se = {stderr:.3f}. The DP answer is re-derived here by sweeping "
            f"the lesson's own step() rather than read off its printed -39.41",
        ),
        practice.Check(
            "FINDING: the line the curve converges to is not the DP answer",
            abs(bias) > 0.05 and censored > exact,
            f"rollout() carries max_steps={CAP}, so the estimator's limit is the {CAP}-step "
            f"censored value {censored:.4f} and not the MDP's {exact:.4f}. The {abs(bias):.4f} "
            f"between them is bias, not noise: it is there at every sample size, and "
            f"{100 * result['capped'] / EPISODES:.1f}% of episodes are the ones that carry it",
        ),
        practice.Check(
            "FINDING: at 10,000 episodes that bias is invisible",
            abs(bias) < stderr and needed > 4 * EPISODES,
            f"the bias is {abs(bias) / stderr:.2f} standard errors at the {EPISODES:,} episodes "
            f"the exercise specifies. Noise falls as 1/sqrt(n) and bias does not, so they are "
            f"equal only at ~{needed:,.0f} episodes -- {needed / EPISODES:.0f}x the run. The plot "
            "cannot show the one thing that is wrong with it",
        ),
        practice.Check(
            "FINDING: the noise is what the plot actually shows",
            sigma > 20 and stderr > 0.2,
            f"the first-visit return at (0,0) has sigma = {sigma:.2f} against a mean of "
            f"{exact:.1f}, so reading V(0,0) to +/-0.1 needs {(sigma / 0.1) ** 2:,.0f} episodes. "
            f"The {EPISODES:,} asked for buy +/-{stderr:.3f}, and the visible movement in the "
            f"table -- {result['marks'][100]:.2f} at n=100 -- is that sigma, not learning",
        ),
        practice.Check(
            "FINDING: the exact answer costs 28,020 backups and no sampling at all",
            result["steps"] > 10 * result["cheap_sweeps"] * 60,
            f"MC spends {result['steps']:,} environment steps to land within a standard error. "
            f"Sweeping the lesson's own step() to 1e-6 costs {result['cheap_sweeps']} sweeps, "
            f"{60 * result['cheap_sweeps']:,} backups -- "
            f"{result['steps'] / (60 * result['cheap_sweeps']):.0f}x fewer operations for an "
            f"exact answer, and {result['sweeps']} sweeps takes it to 1e-12. The model is right "
            "there in step(); MC is the method for when it is not",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
