"""Exercise 1 — the running mean never crosses it, and four other readings do.

    **Easy.** Run `code/main.py`. Plot the per-episode return curve. How many
    episodes until the running mean exceeds -10?

Reading of the exercise: "running mean" is not defined anywhere in the lesson, so
rather than pick a window and report one number, all five readings a learner could
reasonably take are computed on the same run -- the cumulative mean from episode 0
and trailing windows of 10, 50 and 100, plus the 50-episode blocks the lesson's
own `main` prints. The loop is the one `main` writes, rebuilt from the lesson's own
`init_net`, `clone`, `epsilon_greedy`, `step` and `train_step` at its own seed,
because `main` returns nothing to measure. The plot ships as a table, per `D14`.

**ANSWER: it depends on a window the exercise never states -- and the most literal
reading never crosses at all.** The cumulative running mean ends the 400 episodes
at -10.65. Trailing-10 crosses at 115, trailing-50 at 154, trailing-100 at 179,
and the lesson's own 50-blocks at 200.

**FINDING: the first 50 episodes are heavy enough to sink the cumulative mean for
the whole run.** They average about -28 against a -6 optimum, and no later episode
can be better than -6, so the average of everything is still under -10 at episode
400.

**MECHANISM: the crossing episode measures the `ε` schedule, not the learning.**
`ε = max(0.05, 1 - ep/200)`, and the exact 50-step-capped return of an `ε`-soft
policy over the *optimal* policy crosses -10 within a few episodes of where the
measured curve does. An agent that was optimal from episode 0 would produce almost
the same answer.

**FINDING: the network is done long before the curve is.** `max_a Q(0,0,a)` lands
within about 0.02 of `V*(0,0) = -5.8520`, so what the last 200 episodes of the
curve are showing is exploration being switched off.

**FINDING: the cap is what makes the early episodes that heavy.** The loop stops
at 50 steps, so the worst episode is exactly -50, and the first block sits near
that rather than near the -6 optimum.

Structure: `train` is `main`'s own loop over the lesson's primitives; `ceiling`
sweeps the lesson's `step` for the exact capped `ε`-soft return, and `trailing`
reads the same crossing off both the measured curve and that exact one.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "05-dqn"
EPISODES, CAP, BATCH, CAPACITY, SYNC = 400, 50, 32, 2000, 200
GAMMA, LR, SEED, TARGET = 0.99, 0.05, 0, -10.0
CELLS = [(r, c) for r in range(4) for c in range(4)]


def train(ref, episodes=EPISODES, seed=SEED):
    """`main`'s loop, rebuilt from the lesson's own primitives because `main` returns nothing."""
    rng = random.Random(seed)
    online = ref.init_net(16, 32, len(ref.ACTIONS), rng)
    target, buffer, log, count = ref.clone(online), [], [], 0
    for episode in range(episodes):
        state, total = ref.reset(), 0.0
        for _ in range(CAP):
            action = ref.epsilon_greedy(online, state, rng, max(0.05, 1.0 - episode / 200))
            nxt, reward, done = ref.step(state, ref.ACTIONS[action])
            total += reward
            buffer.append((state, action, reward, nxt, done))
            del buffer[:-CAPACITY]
            if len(buffer) >= BATCH:
                ref.train_step(online, target, rng.sample(buffer, BATCH), GAMMA, LR)
            count += 1
            if count % SYNC == 0:
                target = ref.clone(online)
            if done:
                break
            state = nxt
        log.append(total)
    return log, ref.forward(online, ref.state_features((0, 0)))[0]


def look(ref, state, action, values, gamma):
    """One step of the lesson's own `step`, backed up."""
    nxt, reward, done = ref.step(state, action)
    return reward + (0.0 if done else gamma * values[nxt])


def optimal(ref):
    """`V*` and its greedy policy, swept from the lesson's own `step`."""
    values = {s: 0.0 for s in CELLS}
    for _ in range(40):
        values = {s: (0.0 if s == ref.TERMINAL else
                      max(look(ref, s, a, values, GAMMA) for a in ref.ACTIONS)) for s in CELLS}
    return values, {s: max(ref.ACTIONS, key=lambda a: look(ref, s, a, values, GAMMA))
                    for s in CELLS}


def ceiling(ref, greedy, eps):
    """Exact undiscounted return of the eps-soft policy over `greedy`, truncated at `CAP` steps."""
    values = {s: 0.0 for s in CELLS}
    for _ in range(CAP):
        values = {s: (0.0 if s == ref.TERMINAL else sum(
            (eps / 4 + (1 - eps if a == greedy[s] else 0.0)) * look(ref, s, a, values, 1.0)
            for a in ref.ACTIONS)) for s in CELLS}
    return values[(0, 0)]


def trailing(log, window):
    """First episode at which the trailing-`window` mean exceeds the target."""
    return next((i for i in range(window, len(log) + 1)
                 if statistics.fmean(log[i - window:i]) > TARGET), None)


def crossings(log):
    """First episode at which each reading of 'running mean' exceeds the target."""
    running, cumulative = 0.0, None
    for i, value in enumerate(log):
        running += (value - running) / (i + 1)
        if running > TARGET and cumulative is None:
            cumulative = i + 1
    blocks = [statistics.fmean(log[i:i + 50]) for i in range(0, len(log), 50)]
    return {"cumulative": cumulative, "final_cumulative": running, "blocks": blocks,
            "block": next((50 * (i + 1) for i, b in enumerate(blocks) if b > TARGET), None),
            **{w: trailing(log, w) for w in (10, 50, 100)}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log, q0 = train(ref)
    star, greedy = optimal(ref)
    ideal = [ceiling(ref, greedy, max(0.05, 1.0 - ep / 200)) for ep in range(EPISODES)]
    return {**crossings(log), "log": log, "q0": q0, "star": star[(0, 0)],
            "ideal_cross": trailing(ideal, 50)}


def verify(result):
    blocks, log = result["blocks"], result["log"]
    worst, best = min(log), max(log)
    return [
        practice.Check(
            "ANSWER: never, 115, 154, 179 or 200 -- the exercise does not say which",
            result["cumulative"] is None and result[10] < result[50] < result[100],
            f"the first episode at which each reading of 'running mean' exceeds {TARGET:.0f}: "
            f"cumulative from episode 0 -- never, ending at {result['final_cumulative']:.2f}; "
            f"trailing-10 at {result[10]}; trailing-50 at {result[50]}; trailing-100 at "
            f"{result[100]}; the lesson's 50-blocks at {result['block']}. Five readings, four "
            "numbers and one that does not exist",
        ),
        practice.Check(
            "FINDING: the first 50 episodes sink the cumulative mean for the whole run",
            blocks[0] < -20 and result["final_cumulative"] < TARGET,
            f"the first block averages {blocks[0]:.2f} while the last averages {blocks[-1]:.2f}, "
            f"and no episode can beat the {best:.0f} optimum. Averaging everything from episode 0 "
            f"leaves {result['final_cumulative']:.2f} after {EPISODES} episodes. At the "
            f"converged rate it would take about "
            f"{EPISODES * (TARGET - result['final_cumulative']) / (blocks[-1] - TARGET):.0f} "
            "further episodes to drag the cumulative mean over the line",
        ),
        practice.Check(
            "MECHANISM: the crossing episode measures the epsilon schedule, not the learning",
            abs(result["ideal_cross"] - result[50]) <= 25,
            f"epsilon is max(0.05, 1 - ep/200), and the exact {CAP}-step-capped return of an "
            f"eps-soft policy over the *optimal* policy -- swept from the lesson's own step(), no "
            f"learning in it -- crosses {TARGET:.0f} on a trailing-50 window at episode "
            f"{result['ideal_cross']} against the measured {result[50]}. An agent optimal from "
            "episode 0 answers this exercise with nearly the same number",
        ),
        practice.Check(
            "FINDING: the network is finished long before the curve is",
            abs(max(result["q0"]) - result["star"]) < 0.1,
            f"max_a Q(0,0,a) = {max(result['q0']):.4f} against V*(0,0) = {result['star']:.4f}, a "
            f"gap of {abs(max(result['q0']) - result['star']):.4f}. The blocks from episode 150 "
            f"on read {', '.join(f'{b:.2f}' for b in blocks[3:])} -- that movement is epsilon "
            "being switched off, not the value function still arriving",
        ),
        practice.Check(
            "FINDING: the 50-step cap is what makes the early episodes that heavy",
            worst == -float(CAP),
            f"the loop stops at {CAP} steps, so the worst episode is exactly {worst:.0f} and the "
            f"first block's {blocks[0]:.2f} sits nearer that floor than the {best:.0f} optimum. "
            f"Most of the curve's visible {best:.0f} to {worst:.0f} range is a truncation "
            "constant rather than anything the agent did",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
