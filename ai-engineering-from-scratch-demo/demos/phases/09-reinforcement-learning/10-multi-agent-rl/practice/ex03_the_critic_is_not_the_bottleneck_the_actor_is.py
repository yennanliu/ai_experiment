"""Exercise 3 — the centralised critic is not the bottleneck; the actor's eyes are.

    **Hard.** Implement a centralized critic for MAPPO-style training and compare
    convergence speed to independent PPO on the coordination task.

Reading of the exercise: MAPPO is centralised training with decentralised execution,
so the two arms differ in the critic's input only -- both keep actors over each
agent's own position, which is the setting exercise 2 showed the coordination task
defeats. The comparison is therefore run on both tasks, and a third pair of arms
gives the *actor* the joint state, because otherwise the result reads as a verdict
on centralised critics rather than on where the information has to arrive.

**ANSWER: on the coordination task the centralised critic makes it worse.** 3 of 10
seeds finish above zero against 6 for independent PPO, at a final mean of -69.1
against -39.1. Neither arm solves the task.

**FINDING: on the separable task the same critic is a clear win.** 10 of 10 seeds
above zero against 8, and a final mean of **+2.97** against **-17.7**. The
centralised critic works exactly where coordination is not required.

**MECHANISM: the actor cannot represent what the critic has learned.** Giving the
*actor* the joint state solves the coordination task outright -- 10 of 10 seeds,
final **+2.36** with a local critic and **+2.75** with a central one. The critic's
input barely matters once the actor can see its partner; the actor's input decides
everything.

**FINDING: and the joint-state actor pays for it in speed.** It crosses at a median
near 1,400 episodes against 263 for a local actor on the separable task -- the same
625-against-25 cost exercises 1 and 2 measured. Centralised execution is not free;
it is simply the only thing that works here.

Structure: `run` is one actor-critic arm parameterised by which of the two
components sees the joint state; `strict_step` is exercise 2's rule.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "10-multi-agent-rl"
EPISODES, SEEDS, WINDOW, CAP = 3_000, 10, 100, 100
LR_A, LR_V, GAMMA = 0.1, 0.2, 0.95
ARMS = ((False, False, True), (False, True, True), (False, False, False),
        (False, True, False), (True, False, True), (True, True, True))


def strict_step(ref, state, pair):
    """Exercise 2's rule: both agents must step onto the goal together."""
    nxt = (ref.move(state[0], pair[0]), ref.move(state[1], pair[1]))
    done = nxt == (ref.GOAL, ref.GOAL) and ref.GOAL not in state
    return nxt, (10.0 if done else -1.0), done


def act(logits, rng):
    """(softmax of `logits`, one action index sampled from it)."""
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    probs = [e / sum(exps) for e in exps]
    draw = rng.random()
    for i, p in enumerate(probs):
        draw -= p
        if draw <= 0:
            return probs, i
    return probs, len(probs) - 1


def episode(ref, theta, rng, actor_joint, strict):
    """One episode; returns the trajectory and its total reward."""
    state, traj, total = ref.reset(), [], 0.0
    for _ in range(CAP):
        keys = (state, state) if actor_joint else state
        probs, acts = zip(*[act(theta[i][keys[i]], rng) for i in (0, 1)])
        pair = tuple(ref.ACTIONS[a] for a in acts)
        nxt, reward, done = strict_step(ref, state, pair) if strict else ref.step(state, pair)
        total += reward
        traj.append((keys, acts, probs, state, reward))
        state = nxt
        if done:
            break
    return traj, total


def discounted(traj):
    """Return-to-go for each step of the episode."""
    returns, carry = [], 0.0
    for record in reversed(traj):
        carry = record[4] + GAMMA * carry
        returns.append(carry)
    return returns[::-1]


def baseline(critic, keys, state, central):
    """The critic's estimate for one step: the joint state, or the mean of the two local ones."""
    return critic[state] if central else (critic[keys[0]] + critic[keys[1]]) / 2


def learn(ref, theta, critic, traj, central):
    """One advantage-normalised update over the episode."""
    returns = discounted(traj)
    raw = [g - baseline(critic, k, st, central)
           for (k, _a, _p, st, _r), g in zip(traj, returns)]
    mean, sd = statistics.fmean(raw), statistics.pstdev(raw) + 1e-8
    for (keys, acts, probs, state, _r), total, adv in zip(traj, returns, raw):
        scaled = (adv - mean) / sd
        for key in ([state] if central else list(keys)):
            critic[key] += LR_V * (total - critic[key])
        for i in (0, 1):
            for k in range(len(ref.ACTIONS)):
                theta[i][keys[i]][k] += LR_A * scaled * (
                    (1.0 if k == acts[i] else 0.0) - probs[i][k])


def cell(ref, actor_joint, central, strict):
    """One arm across the seeds: how often it ends above zero, and how fast it gets there."""
    finals, reached = [], []
    for seed in range(SEEDS):
        rng = random.Random(seed)
        theta = [defaultdict(lambda: [0.0] * len(ref.ACTIONS)) for _ in (0, 1)]
        critic, log = defaultdict(float), []
        for _ in range(EPISODES):
            traj, total = episode(ref, theta, rng, actor_joint, strict)
            learn(ref, theta, critic, traj, central)
            log.append(total)
        finals.append(statistics.fmean(log[-WINDOW:]))
        reached.append(next((i for i in range(WINDOW, len(log) + 1)
                             if statistics.fmean(log[i - WINDOW:i]) > 0.0), None))
    ok = [r for r in reached if r]
    return {"final": statistics.fmean(finals), "above": sum(f > 0 for f in finals),
            "median": statistics.median(ok) if ok else 0}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {arm: cell(ref, *arm) for arm in ARMS}


def verify(result):
    ind, mappo = result[(False, False, True)], result[(False, True, True)]
    sep_ind, sep_mappo = result[(False, False, False)], result[(False, True, False)]
    local_critic, central_critic = result[(True, False, True)], result[(True, True, True)]
    return [
        practice.Check(
            "ANSWER: on the coordination task the centralised critic makes it worse",
            mappo["above"] <= ind["above"] and mappo["final"] < ind["final"],
            f"over {SEEDS} seeds of {EPISODES:,} episodes on exercise 2's strict rule, MAPPO is "
            f"above zero on {mappo['above']}/{SEEDS} at a final mean {mappo['final']:.1f}, "
            f"against {ind['above']}/{SEEDS} and {ind['final']:.1f} for independent PPO -- "
            "neither solves it, and centralising the critic costs a little",
        ),
        practice.Check(
            "FINDING: on the separable task the same critic is a clear win",
            sep_mappo["above"] > sep_ind["above"] and sep_mappo["final"] > sep_ind["final"] + 10,
            f"the identical pair on the lesson's shipped rule: {sep_mappo['above']}/{SEEDS} "
            f"above zero against {sep_ind['above']}/{SEEDS}, final mean "
            f"{sep_mappo['final']:+.2f} against {sep_ind['final']:+.2f}. It works exactly where "
            "coordination is not required -- the opposite of the setting proposed for it",
        ),
        practice.Check(
            "MECHANISM: the actor cannot represent what the critic has learned",
            local_critic["above"] == central_critic["above"] == SEEDS,
            f"giving the *actor* the joint state solves the coordination task outright: "
            f"{local_critic['above']}/{SEEDS} above zero at {local_critic['final']:+.2f} with a "
            f"local critic, {central_critic['above']}/{SEEDS} at {central_critic['final']:+.2f} "
            f"with a central one. The critic's input barely separates those two; the actor's "
            f"separates everything -- MAPPO executes locally, and a local policy cannot express "
            f"the timing this task needs. Not free either: the joint-state actor crosses at a "
            f"median {central_critic['median']:.0f} against {sep_ind['median']:.0f}, the same "
            "625-against-25 cost exercises 1 and 2 measured",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
