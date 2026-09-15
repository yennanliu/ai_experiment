"""Exercise 1 — the joint state costs four times the episodes, and buys nothing here.

    **Easy.** Train independent Q-learning on the 2-agent cooperative GridWorld. How
    many episodes until mean return > 0? Plot the joint learning curve.

Reading of the exercise: "mean return > 0" needs a window, so it is read as the
first episode at which a trailing 100 crosses, applied to 10 seeds because one seed
of `independent_q` fails to cross at all inside its budget. The lesson's own
`independent_q` is run unchanged; the curve ships as a table of its own `block_mean`.

**ANSWER: a median of 735 episodes, and one seed of ten never gets there.** The
final mean return is 0.30 against a best-possible 3.0.

**FINDING: the best possible return is 3.0, not 10.** `step` pays `+10` on the
joint arrival and `-1` on every other turn. Agent 1 starts 8 moves from the goal and
agent 2 starts 4, so the fastest episode is 8 turns and scores `-7 + 10`. The
exercise's "> 0" bar sits at 10% of the achievable range, not near the reward the
environment advertises.

**FINDING: these learners are not independent in the sense the name implies.** Both
index their Q-tables by `s = (pos1, pos2)` -- the *joint* state. They are
independent in their actions only; each sees exactly where the other is.

**FINDING: and that observation costs 4x the episodes while buying nothing.** Giving
each agent only its own position -- 25 states instead of 625 -- crosses at a median
of **188** against 735 and ends at **1.27** against 0.30. On this task the joint
state is 25x more to learn and there is nothing in it to learn.

**MECHANISM: the shipped task is separable.** `done` requires both agents to *be* at
the goal, not to arrive together, so an agent that reaches it first can stand still
and wait. Each agent's optimal behaviour is "walk to the goal", which does not
depend on the other at all.

Structure: `crossing` reads the first trailing-100 window above zero; `local_q` is
the lesson's own update with the joint state replaced by each agent's own position.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "10-multi-agent-rl"
EPISODES, SEEDS, WINDOW, CAP = 1_500, 10, 100, 100
ALPHA, GAMMA, EPSILON = 0.1, 0.95, 0.15


def crossing(log, window=WINDOW):
    """First episode at which the trailing-`window` mean return exceeds zero."""
    return next((i for i in range(window, len(log) + 1)
                 if statistics.fmean(log[i - window:i]) > 0.0), None)


def local_q(ref, seed):
    """The lesson's own update, with each agent indexing only its own position."""
    rng = random.Random(seed)
    tables = [defaultdict(ref.default_q), defaultdict(ref.default_q)]
    log = []
    for _ in range(EPISODES):
        state, total = ref.reset(), 0.0
        for _ in range(CAP):
            acts = [ref.epsilon_greedy(tables[i], state[i], rng, EPSILON) for i in (0, 1)]
            nxt, reward, done = ref.step(state, tuple(acts))
            total += reward
            for i in (0, 1):
                target = reward if done else reward + GAMMA * max(tables[i][nxt[i]].values())
                tables[i][state[i]][acts[i]] += ALPHA * (target - tables[i][state[i]][acts[i]])
            state = nxt
            if done:
                break
        log.append(total)
    return log


def distance(start, goal):
    """Manhattan moves from `start` to `goal` on the grid."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def summarise(logs):
    """Crossing episodes and final performance across the seeds."""
    reached = [crossing(log) for log in logs]
    finals = [statistics.fmean(log[-WINDOW:]) for log in logs]
    return {"crossed": sum(1 for r in reached if r), "final": statistics.fmean(finals),
            "median": statistics.median([r for r in reached if r]), "above": sum(f > 0 for f in finals)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    joint = [ref.independent_q(EPISODES, rng=random.Random(s))[2] for s in range(SEEDS)]
    start = ref.reset()
    far, near = distance(start[0], ref.GOAL), distance(start[1], ref.GOAL)
    return {"joint": summarise(joint), "local": summarise([local_q(ref, s) for s in range(SEEDS)]),
            "blocks": ref.block_mean(joint[0], 150), "best": 10.0 - (max(far, near) - 1),
            "far": far, "near": near, "cells": ref.GRID ** 2}


def verify(result):
    joint, local = result["joint"], result["local"]
    return [
        practice.Check(
            "ANSWER: a median of 735 episodes, and one seed of ten never gets there",
            joint["crossed"] < SEEDS and 500 < joint["median"] < 1000,
            f"the first trailing-{WINDOW} window above zero arrives at a median episode "
            f"{joint['median']:.0f}, on the {joint['crossed']} of {SEEDS} seeds that reach it at "
            f"all. Final mean return is {joint['final']:.2f}, above zero on {joint['above']} of "
            f"{SEEDS}. The curve by 150-episode block: "
            + ", ".join(f"{b:.1f}" for b in result["blocks"]),
        ),
        practice.Check(
            "FINDING: the best possible return is 3.0, not 10",
            result["best"] == 3.0,
            f"`step` pays +10 on the joint arrival and -1 on every other turn. Agent 1 starts "
            f"{result['far']} moves from the goal and agent 2 starts {result['near']}, so the "
            f"fastest episode is {max(result['far'], result['near'])} turns and scores "
            f"{result['best']:.1f}. The exercise's '> 0' bar is therefore satisfied by any "
            f"episode averaging under {int(10)} turns, and the whole achievable band is "
            f"[{-CAP + 10}, {result['best']:.1f}] -- nowhere near the +10 the reward advertises",
        ),
        practice.Check(
            "FINDING: these learners are not independent in the sense the name implies",
            result["cells"] ** 2 > 20 * result["cells"],
            f"both Q-tables are indexed by s = (pos1, pos2), the joint state -- "
            f"{result['cells']}^2 = {result['cells'] ** 2} entries against {result['cells']} for "
            f"a single agent's own position. They are independent in their actions only; each "
            "one sees exactly where the other is at every step, which is the information "
            "independent learners are usually defined by not having",
        ),
        practice.Check(
            "FINDING: and that observation costs 4x the episodes while buying nothing",
            local["median"] < joint["median"] / 2 and local["final"] > joint["final"],
            f"giving each agent only its own position -- {result['cells']} states instead of "
            f"{result['cells'] ** 2} -- crosses at a median {local['median']:.0f} against "
            f"{joint['median']:.0f}, a factor of {joint['median'] / local['median']:.1f}, and "
            f"ends at {local['final']:.2f} against {joint['final']:.2f}, above zero on "
            f"{local['above']} of {SEEDS} seeds against {joint['above']}",
        ),
        practice.Check(
            "MECHANISM: the shipped task is separable, so there is nothing to coordinate",
            local["above"] >= joint["above"],
            f"`done` requires both agents to *be* at the goal, not to arrive together, so "
            f"whichever gets there first can stand still and wait -- `move` at the goal returns "
            f"the goal. Each agent's optimal behaviour is 'walk to the goal', which does not "
            "depend on the other at all, so the joint state is 25x more to learn with nothing in "
            "it to learn. Exercise 2 makes arrival simultaneous and the picture inverts",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
