"""Exercise 1 — it never fails, and it is four times slower.

    **Easy.** Train actor-critic with MC advantage (`G_t - V(s_t)`) on 4×4
    GridWorld. Compare sample efficiency to REINFORCE-with-running-mean-baseline
    from Lesson 06.

Reading of the exercise: no new algorithm is written. The lesson's `actor_critic`
always routes through `gae_advantages`, and GAE at `λ = 1.0` telescopes to exactly
`G_t - V(s_t)`, so "MC advantage" is the shipped function at one parameter value.
Lesson 06's `reinforce` is imported from its own lesson rather than reproduced, and
both arms are run over 30 seeds with one convergence rule, because lesson 06
established that 22% of REINFORCE runs never converge and a one-seed comparison
picks its own answer.

**ANSWER: sample efficiency says REINFORCE, by 3.6x -- and it is the wrong
question.** Median episodes-to-converge is 125 for REINFORCE and 448 for
actor-critic. But REINFORCE collapses on 8 of 30 seeds and actor-critic collapses
on **none**.

**FINDING: the two summaries disagree about which method is better.** Median final
return favours REINFORCE (-5.91 against -6.42, since its surviving runs find a
sharper policy); mean final return favours actor-critic by 14.8 points (-6.42
against -21.24), because the mean carries the failures the median throws away.

**MECHANISM: the advantage is normalised, so about half of all actions are
reinforced.** Lesson 06 measured that vanilla REINFORCE has `adv < 0` at 100% of
updates -- it can only ever punish. `normalize` forces zero mean, so **48.1%** of
the advantages the actor sees here are positive. That is the difference between an
algorithm that can be pushed onto a non-terminating policy and one that cannot.

**CONTROL: it is not the entropy bonus.** With `ent_coef = 0` the collapse rate is
still 0/30, and the median final return is slightly *better*.

Structure: `arm` runs one method at one seed; `converged_at` is the single rule
applied to both.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE = "09-reinforcement-learning"
LESSON, PRIOR = "07-actor-critic-a2c-a3c", "06-policy-gradients-reinforce"
EPISODES, SEEDS, GAMMA, TAIL = 1_500, 30, 0.99, 150
WINDOW, THRESHOLD, STEP_CAP = 50, -10.0, 100
CAP_RETURN = -(1 - GAMMA**STEP_CAP) / (1 - GAMMA)


def converged_at(log):
    """First episode after which `WINDOW` consecutive returns stay above `THRESHOLD`."""
    return next((i for i in range(WINDOW, len(log) + 1)
                 if all(g > THRESHOLD for g in log[i - WINDOW:i])), None)


def summarise(logs):
    """One method's outcome distribution across the seeds."""
    finals = [statistics.fmean(log[-TAIL:]) for log in logs]
    reached = [converged_at(log) for log in logs]
    return {"median": statistics.median(finals), "mean": statistics.fmean(finals),
            "collapsed": sum(f < -60 for f in finals),
            "converged": sum(1 for c in reached if c),
            "at": statistics.median([c for c in reached if c])}


def positive_share(ref, lam=1.0, episodes=400, seed=7):
    """The fraction of normalised advantages the actor sees that are positive."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    positive = total = 0
    for _ in range(episodes):
        traj = ref.rollout(theta, w, rng)
        advantages, returns = ref.gae_advantages(traj, gamma=GAMMA, lam=lam)
        for value in ref.normalize(advantages):
            positive += value > 0
            total += 1
        for t, node in enumerate(traj):
            error = returns[t] - ref.value(w, node["x"])
            for j in range(ref.N_FEAT):
                w[j] += 0.1 * error * node["x"][j]
    return positive / total


def telescopes(ref, seed=11, episodes=60):
    """GAE at lam=1 against `G_t - V(s_t)` computed directly: the worst disagreement."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    worst = 0.0
    for _ in range(episodes):
        traj = ref.rollout(theta, w, rng)
        advantages, _returns = ref.gae_advantages(traj, gamma=GAMMA, lam=1.0)
        future = 0.0
        for t in reversed(range(len(traj))):
            future = traj[t]["r"] + GAMMA * future
            worst = max(worst, abs(advantages[t] - (future - traj[t]["v"])))
    return worst


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prior = parity.load_reference(PHASE, PRIOR, "main")
    critic = [ref.actor_critic(EPISODES, lam=1.0, rng=random.Random(s))[2] for s in range(SEEDS)]
    plain = [prior.reinforce(EPISODES, use_baseline=True, rng=random.Random(s))[1]
             for s in range(SEEDS)]
    quiet = [ref.actor_critic(EPISODES, lam=1.0, ent_coef=0.0, rng=random.Random(s))[2]
             for s in range(SEEDS)]
    return {"critic": summarise(critic), "plain": summarise(plain),
            "quiet": summarise(quiet), "positive": positive_share(ref),
            "telescopes": telescopes(ref)}


def verify(result):
    critic, plain, quiet = result["critic"], result["plain"], result["quiet"]
    slower = critic["at"] / plain["at"]
    return [
        practice.Check(
            "ANSWER: REINFORCE converges 3.6x faster, and fails on a quarter of seeds",
            critic["at"] > 2 * plain["at"] and critic["collapsed"] == 0 < plain["collapsed"],
            f"median episodes-to-converge over {SEEDS} seeds is {plain['at']:.0f} for "
            f"REINFORCE-with-baseline and {critic['at']:.0f} for actor-critic, {slower:.1f}x "
            f"slower. But REINFORCE collapses on {plain['collapsed']}/{SEEDS} seeds "
            f"({100 * plain['collapsed'] / SEEDS:.1f}%) and actor-critic on "
            f"{critic['collapsed']}/{SEEDS}. Sample efficiency and reliability answer this "
            "exercise in opposite directions",
        ),
        practice.Check(
            "FINDING: the median and the mean disagree about which method is better",
            plain["median"] > critic["median"] and critic["mean"] > plain["mean"] + 5,
            f"median final return is {plain['median']:.2f} for REINFORCE against "
            f"{critic['median']:.2f} for actor-critic -- REINFORCE's surviving runs find the "
            f"sharper policy. Mean final return is {plain['mean']:.2f} against "
            f"{critic['mean']:.2f}, {critic['mean'] - plain['mean']:.1f} the other way, because "
            "the mean carries the failures the median throws away",
        ),
        practice.Check(
            "MECHANISM: normalising the advantage means half of all actions are reinforced",
            0.45 < result["positive"] < 0.55,
            f"lesson 06 measured adv < 0 at 100% of REINFORCE's updates -- it can only punish. "
            f"`normalize` subtracts the per-episode mean, so {100 * result['positive']:.1f}% of "
            f"the advantages this actor sees are positive. An algorithm that can reinforce "
            f"cannot be walked onto a non-terminating policy, which is why {plain['collapsed']} "
            "seeds collapse on one side and none on the other",
        ),
        practice.Check(
            "CONTROL: it is not the entropy bonus doing it",
            quiet["collapsed"] == 0,
            f"re-running the same {SEEDS} seeds with ent_coef=0 leaves the collapse rate at "
            f"{quiet['collapsed']}/{SEEDS} and moves the median final return to "
            f"{quiet['median']:.2f} against {critic['median']:.2f} with the bonus on. The "
            "stability comes from the baseline and the normalisation, not from the exploration "
            "term layered on top",
        ),
        practice.Check(
            "FINDING: no new algorithm was needed -- MC advantage is the shipped one at lam=1",
            result["telescopes"] < 1e-12,
            f"`actor_critic` always routes through `gae_advantages`, and GAE at lam=1.0 agrees "
            f"with G_t - V(s_t) computed directly to {result['telescopes']:.1e} over every step "
            f"of 60 episodes -- it telescopes exactly. So exercise 1's 'MC advantage', exercise "
            f"2's TD residual at lam=0 and exercise 3's sweep are one shipped function at three "
            f"settings, and the collapsed baseline both lessons share is {CAP_RETURN:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
