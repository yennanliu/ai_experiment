"""Exercise 2 — it converges, until the agents are actually independent.

    **Medium.** Add a "coordination" task: the goal is reached only when both agents
    step onto it on the same turn. Does independent Q still converge? What breaks?

Reading of the exercise: the shipped `step` already requires both agents to *be* at
the goal together, so the task it asks for is a strictly stronger rule -- neither
agent may already be standing there. That one clause is the only change; everything
else is the lesson's own. And because exercise 1 found that both Q-tables are indexed
by the joint state, the sweep is run twice: once as shipped, and once with each agent
seeing only its own position, which is what "independent" normally means.

**ANSWER: yes, it still converges -- because the learners are not independent.** With
the joint state, the strict rule crosses on 10 of 10 seeds at a median of 738 against
735 for the shipped rule. Almost nothing changes.

**FINDING: what breaks is stability, not convergence.** Final mean return falls from
0.30 to -0.13, and the seeds ending above zero fall from 8 of 10 to 5.

**FINDING: take the joint state away and the coordination task becomes
unsolvable.** With each agent seeing only its own position, the shipped rule is
*easier* -- 10 of 10, median 188 -- and the strict rule is **0 of 10**, final mean
-4.69, never once crossing zero in 1,500 episodes.

**MECHANISM: camping is what made the shipped rule separable.** Under it the first
agent to arrive stands still and waits, so each agent's policy is "walk to the goal"
and needs nothing about the other. The strict rule removes waiting, so the arrival
has to be timed -- and timing is exactly the thing an agent that cannot see its
partner has no way to represent.

**FINDING: so the exercise's premise is already true of the lesson.** `done = (new1
== GOAL) and (new2 == GOAL)` is a coordination condition as shipped; what makes it
coordinate-able is that the learners were given the joint state to begin with.

Structure: `strict_step` is the one-clause change; `train` runs the lesson's own
update at either observation.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "10-multi-agent-rl"
EPISODES, SEEDS, WINDOW, CAP = 1_500, 10, 100, 100
ALPHA, GAMMA, EPSILON = 0.1, 0.95, 0.15


def strict_step(ref, state, pair):
    """Both agents must step ONTO the goal together: neither may already be there."""
    old = state
    nxt = (ref.move(old[0], pair[0]), ref.move(old[1], pair[1]))
    done = (nxt[0] == ref.GOAL and nxt[1] == ref.GOAL
            and old[0] != ref.GOAL and old[1] != ref.GOAL)
    return nxt, (10.0 if done else -1.0), done


def update(ref, tables, keys, acts, reward, after, done):
    """The lesson's own Q update for both agents at one transition."""
    for i in (0, 1):
        target = reward if done else reward + GAMMA * max(tables[i][after[i]].values())
        tables[i][keys[i]][acts[i]] += ALPHA * (target - tables[i][keys[i]][acts[i]])


def rollout(ref, tables, rng, strict, joint):
    """One episode of the lesson's own update; returns the total reward."""
    state, total = ref.reset(), 0.0
    for _ in range(CAP):
        keys = (state, state) if joint else state
        acts = [ref.epsilon_greedy(tables[i], keys[i], rng, EPSILON) for i in (0, 1)]
        nxt, reward, done = (strict_step(ref, state, tuple(acts)) if strict
                             else ref.step(state, tuple(acts)))
        total += reward
        update(ref, tables, keys, acts, reward, (nxt, nxt) if joint else nxt, done)
        state = nxt
        if done:
            break
    return total


def train(ref, strict, joint, seed):
    """`EPISODES` of the lesson's own Q update, at either rule and either observation."""
    rng = random.Random(seed)
    tables = [defaultdict(ref.default_q), defaultdict(ref.default_q)]
    return [rollout(ref, tables, rng, strict, joint) for _ in range(EPISODES)]


def crossing(log):
    """First episode at which the trailing-`WINDOW` mean return exceeds zero."""
    return next((i for i in range(WINDOW, len(log) + 1)
                 if statistics.fmean(log[i - WINDOW:i]) > 0.0), None)


def cell(ref, strict, joint):
    """One (rule, observation) cell across the seeds."""
    logs = [train(ref, strict, joint, s) for s in range(SEEDS)]
    reached = [crossing(log) for log in logs]
    finals = [statistics.fmean(log[-WINDOW:]) for log in logs]
    ok = [r for r in reached if r]
    return {"crossed": len(ok), "median": statistics.median(ok) if ok else 0,
            "final": statistics.fmean(finals), "above": sum(f > 0 for f in finals)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {(strict, joint): cell(ref, strict, joint)
            for strict in (False, True) for joint in (True, False)}


def verify(result):
    jship, jstrict = result[(False, True)], result[(True, True)]
    lship, lstrict = result[(False, False)], result[(True, False)]
    return [
        practice.Check(
            "ANSWER: yes, it still converges -- because the learners are not independent",
            jstrict["crossed"] >= jship["crossed"]
            and abs(jstrict["median"] - jship["median"]) < 0.2 * jship["median"],
            f"with the joint state, the strict rule crosses on {jstrict['crossed']}/{SEEDS} "
            f"seeds at a median episode {jstrict['median']:.0f}, against "
            f"{jship['crossed']}/{SEEDS} at {jship['median']:.0f} for the shipped rule. "
            "Requiring simultaneous arrival changes almost nothing, because each agent can see "
            "where the other is standing when it decides",
        ),
        practice.Check(
            "FINDING: what breaks is stability, not convergence",
            jstrict["final"] < jship["final"] and jstrict["above"] < jship["above"],
            f"final mean return falls from {jship['final']:.2f} to {jstrict['final']:.2f}, and "
            f"the seeds ending above zero fall from {jship['above']}/{SEEDS} to "
            f"{jstrict['above']}/{SEEDS}. The runs still find the goal; they hold onto it less "
            "reliably, because a policy that has to be right about timing is easier to perturb "
            "than one that only has to walk",
        ),
        practice.Check(
            "FINDING: take the joint state away and the task becomes unsolvable",
            lstrict["crossed"] == 0 and lship["crossed"] == SEEDS,
            f"with each agent seeing only its own position, the shipped rule is *easier* -- "
            f"{lship['crossed']}/{SEEDS} crossing at a median {lship['median']:.0f} against "
            f"{jship['median']:.0f} -- and the strict rule is {lstrict['crossed']}/{SEEDS}, "
            f"final mean {lstrict['final']:.2f}, never once crossing zero in {EPISODES:,} "
            "episodes. That is the coordination failure the exercise is looking for",
        ),
        practice.Check(
            "MECHANISM: camping is what made the shipped rule separable",
            lship["final"] > lstrict["final"] + 4,
            f"under the shipped rule the first agent to arrive stands still and waits -- `move` "
            f"at the goal returns the goal -- so each policy is 'walk to the goal' and needs "
            f"nothing about the other, which is why local observation scores "
            f"{lship['final']:.2f}. The strict rule removes waiting, so arrival has to be timed, "
            f"and an agent that cannot see its partner has no way to represent timing: "
            f"{lstrict['final']:.2f}",
        ),
        practice.Check(
            "FINDING: the exercise's premise is already true of the lesson",
            jship["crossed"] > 0,
            "`done = (new1 == GOAL) and (new2 == GOAL)` is already a coordination condition as "
            "shipped -- both agents must be there at once. The exercise asks to add one, and "
            "what it actually adds is the removal of waiting. What made the original "
            "coordinate-able was never the rule; it was that both learners were handed the "
            "joint state to index on",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
