"""Exercise 2 — the race is decided at spawn, and a team reward cannot see who won.

    Implement a competitive variant: two agents, one pellet, only the first to
    reach gets reward. Which pattern handles competition cleanly? MADDPG
    historically.

Reading of the exercise: the variant reuses the lesson's own `Env`,
`step_toward` and runners, with the second pellet removed and a reward of 1
to whichever agent stands on the pellet first. "Handles cleanly" is judged by
what each pattern's value function is able to represent, since nothing in
the module learns.

**ANSWER: MADDPG, because it is the only one of the three with a critic per
agent trained on that agent's own reward.** MADDPG §4.1 says so directly:
"Since each Q_i is learned separately, agents can have arbitrary reward
structures, including conflicting rewards in a competitive setting." QMIX
mixes per-agent utilities into one Q_tot for a *team* reward, and MAPPO's
centralized V is also fit to a shared return. In this race the team return
is exactly 1 in 500 of 500 episodes, whoever wins, so its variance is 0 and
a shared value has no signal about who won. Agent 0's own return varies
(0 or 1) and carries all of it.

**FINDING: the race is decided before the first step.** A greedy walk takes
exactly the Manhattan distance, and a detour cannot shorten it. `step_toward`
never looks at the other agent, so nothing can block, and cells can be
shared. The closer agent wins 500 of 500 times, so no policy -- learned,
centralized or otherwise -- changes one outcome. The variant can tell the
patterns apart only by what they represent, never by how they play.

**FINDING: the shipped environment cannot say who won, and 85 of 500 starts
are simultaneous.** `collect_if_on_pellet` discards the pellet without
recording who stood on it, and `Env` has no field for a collector or a
per-agent reward. A "first to reach" reward therefore needs new code and a
tie rule. Reading the loop order, which checks agent 0 first, hands all 85
equal-distance starts to agent 0: 256 wins to 244 where a symmetric rule
would split them.

**FINDING: the shipped "centralized" runner is the independent one here.**
With a single pellet, `_assigned_targets` returns `(p, p)`, so both agents
chase it. `run_maddpg_style` and `run_independent` take identical step
counts on 500 of 500 single-pellet episodes.

Structure: `race()` plays one episode with the reference primitives and
reports the winner; `single()` turns the demo's two-pellet `Env` into a
one-pellet one.
"""

from __future__ import annotations

import dataclasses
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "20-marl-maddpg-qmix-mappo"
TRIALS = 500


def single(ref, seed):
    env = ref.Env.new(random.Random(seed))
    return ref.Env(env.agent0, env.agent1, env.pellet0, env.pellet0, {env.pellet0})


def race(ref, env, max_steps=50):
    """(winner by the shipped loop order, tied) for one episode of the race."""
    for _ in range(max_steps):
        env.agent0 = ref.step_toward(env.agent0, env.pellet0)
        env.agent1 = ref.step_toward(env.agent1, env.pellet0)
        on = [pos == env.pellet0 for pos in (env.agent0, env.agent1)]
        if any(on):
            return on.index(True), all(on)
    return None, False


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    winners, ties, closer, same = [], 0, 0, 0
    for seed in range(TRIALS):
        env = single(ref, seed)
        d0, d1 = ref.manhattan(env.agent0, env.pellet0), ref.manhattan(env.agent1, env.pellet0)
        winner, tied = race(ref, env)
        winners.append(winner)
        ties += tied
        closer += winner == (0 if d0 <= d1 else 1)
        same += ref.run_maddpg_style(single(ref, seed)) == ref.run_independent(single(ref, seed))
    agent0 = [int(w == 0) for w in winners]
    team = [int(w == 0) + int(w == 1) for w in winners]
    return {"wins": [winners.count(0), winners.count(1)], "ties": ties, "closer": closer,
            "same": same, "team_var": statistics.pvariance(team),
            "agent0_var": round(statistics.pvariance(agent0), 3),
            "assigned": ref._assigned_targets(single(ref, 0)),
            "fields": [f.name for f in dataclasses.fields(ref.Env)]}


def verify(result):
    wins, ties = result["wins"], result["ties"]
    return [
        practice.Check(
            "ANSWER: MADDPG, the only pattern with a per-agent critic on its own reward",
            result["team_var"] == 0 and result["agent0_var"] > 0,
            f"the team return is 1 in every episode (variance {result['team_var']}), so a "
            f"shared Q_tot or V cannot see who won; agent 0's own return has variance "
            f"{result['agent0_var']}",
        ),
        practice.Check(
            "FINDING: the race is decided before the first step",
            result["closer"] == TRIALS,
            f"the closer agent (ties to agent 0) wins {result['closer']} of {TRIALS}; "
            "step_toward never reads the other agent, so no policy changes an outcome",
        ),
        practice.Check(
            "FINDING: the environment cannot say who won, and 85 of 500 starts are simultaneous",
            ties == 85 and wins == [256, 244]
            and not any("reward" in f or "collector" in f for f in result["fields"]),
            f"Env's fields are {result['fields']}; {ties} of {TRIALS} starts are "
            f"equal-distance, and reading the loop order gives them all to agent 0: wins "
            f"{wins} where a symmetric rule would split the ties",
        ),
        practice.Check(
            "FINDING: the shipped centralized runner is the independent one here",
            result["same"] == TRIALS and len(set(result["assigned"])) == 1,
            f"_assigned_targets returns {result['assigned']} for one pellet, and "
            f"run_maddpg_style matches run_independent on {result['same']} of {TRIALS}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
