"""Exercise 3 — the curriculum wins without ever reaching the target range.

    **Hard.** Implement a curriculum: start with slip=0.0, widen the DR range every
    time the policy hits 90% of optimal. Measure total environment steps to reach
    slip=0.3 zero-shot vs. a fixed DR baseline.

Reading of the exercise: "90% of optimal" needs care when returns are negative --
`value >= 0.9 * optimal` demands a policy *better* than optimal and can never fire,
so the criterion used here is `value >= optimal / 0.9`, a return no worse than
1/0.9 of the optimum's magnitude. Both arms are stopped by the same rule applied to
the same target, checked every 25 episodes, and environment steps are counted inside
the loop rather than inferred from the episode count.

**ANSWER: 5,652 steps against 7,067 -- the curriculum needs 20% fewer.** Both arms
reach the target on 8 of 8 seeds.

**FINDING: it wins on step count while barely winning on episodes.** 300 episodes
against 325. The saving is in episode *length*: an episode at slip 0 is 18.8 steps
where one drawn from `Uniform[0, 0.3]` is 21.7, so the curriculum buys its lead by
practising in a cheaper environment rather than a faster one.

**FINDING: and it passes while still six times narrower than the target.** At the
moment the curriculum clears 90% of optimal at slip=0.3, its own slip ceiling is
**0.05**. It has never trained above a sixth of the slip it is being scored on, which
makes "zero-shot" the right word for the whole comparison and not just its endpoint.

**FINDING: the widening rule almost never fires.** Reaching 0.3 in steps of 0.05
would take six promotions; the run ends after one. What the exercise describes as a
curriculum is, here, a warm-up at slip 0 followed by an early exit.

Structure: `race` runs one arm to the stopping rule, counting steps; the sweeps are
exercise 1's, used for the rule rather than for the answer.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "11-sim-to-real-transfer"
TARGET, FRACTION, SEEDS, CHECK = 0.3, 0.90, 8, 25
WIDEN, CAP, MAX_EPISODES = 0.05, 100, 20_000
ALPHA, GAMMA, EPSILON = 0.1, 0.95, 0.15
CELLS = [(r, c) for r in range(5) for c in range(5)]


def outcomes(ref, state, action, slip):
    """(next state, probability) for one action under `slip`, from the lesson's own rule."""
    perp = ("left", "right") if action in ("up", "down") else ("up", "down")
    for act, prob in ((action, 1 - slip), (perp[0], slip / 2), (perp[1], slip / 2)):
        dr, dc = ref.DELTAS[act]
        yield (min(max(state[0] + dr, 0), 4), min(max(state[1] + dc, 0), 4)), prob


def backup(ref, state, actions, values, slip):
    """The best expected undiscounted return at `state` over `actions`."""
    return max(sum(prob * (-1.0 + (0.0 if nxt == ref.TERMINAL else values[nxt]))
                   for nxt, prob in outcomes(ref, state, a, slip)) for a in actions)


def sweep_values(ref, choose, slip, tol=1e-11):
    """Expected return from (0,0), by sweeping to the fixed point."""
    values = {s: 0.0 for s in CELLS}
    for _ in range(20_000):
        nxt = {s: (0.0 if s == ref.TERMINAL else backup(ref, s, choose(s), values, slip))
               for s in CELLS}
        moved = max(abs(nxt[s] - values[s]) for s in values)
        values = nxt
        if moved < tol:
            break
    return values[(0, 0)]


def exact_value(ref, table, slip):
    """Expected return of the greedy policy read off `table`."""
    best = {s: max(ref.ACTIONS, key=lambda a: table[s][a]) for s in CELLS}
    return sweep_values(ref, lambda s: [best[s]], slip)


def good_enough(ref, table, slip, ceiling):
    """`FRACTION` of optimal, written so it can fire when returns are negative."""
    return exact_value(ref, table, slip) >= ceiling / FRACTION


def episode(ref, table, rng, slip):
    """One Q-learning episode at `slip`; returns the environment steps it consumed."""
    state, steps = (0, 0), 0
    for steps in range(1, CAP + 1):
        action = ref.epsilon_greedy(table, state, rng, EPSILON)
        nxt, reward, done = ref.step(state, action, slip, rng)
        target = reward if done else reward + GAMMA * max(table[nxt].values())
        table[state][action] += ALPHA * (target - table[state][action])
        state = nxt
        if done:
            break
    return steps


def promote(ref, table, high, ceilings):
    """Widen the curriculum's range if the policy has cleared the rule at its current top."""
    ready = high < TARGET and good_enough(ref, table, high, ceilings[round(high, 2)])
    return round(min(TARGET, high + WIDEN), 2) if ready else high


def draw(rng, curriculum, high):
    """The slip for one episode: the curriculum's current range, or the full one."""
    return (rng.uniform(0.0, high) if high else 0.0) if curriculum else rng.uniform(0.0, TARGET)


def race(ref, curriculum, seed, ceilings):
    """Run one arm until it clears the rule at `TARGET`; return steps, episodes and reach."""
    rng, table = random.Random(seed), defaultdict(ref.default_q)
    steps, high = 0, 0.0
    for number in range(MAX_EPISODES):
        if curriculum and number % CHECK == 0 and number:
            high = promote(ref, table, high, ceilings)
        steps += episode(ref, table, rng, draw(rng, curriculum, high))
        if (number + 1) % CHECK == 0 and good_enough(ref, table, TARGET, ceilings[TARGET]):
            return {"steps": steps, "episodes": number + 1, "reach": high}
    return {"steps": None, "episodes": None, "reach": high}


LEVELS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ceilings = {s: sweep_values(ref, lambda _x: ref.ACTIONS, s) for s in LEVELS}
    return {name: [race(ref, name == "curriculum", s, ceilings) for s in range(SEEDS)]
            for name in ("fixed", "curriculum")}


def middle(rows, key):
    """Median of one field across the seeds that finished."""
    return statistics.median([r[key] for r in rows if r["steps"]])


def verify(result):
    fixed, curri = result["fixed"], result["curriculum"]
    steps = (middle(fixed, "steps"), middle(curri, "steps"))
    eps, reach = (middle(fixed, "episodes"), middle(curri, "episodes")), middle(curri, "reach")
    return [
        practice.Check(
            "ANSWER: 5,652 steps against 7,067 -- the curriculum needs 20% fewer",
            steps[1] < steps[0] and all(r["steps"] for r in fixed + curri),
            f"median environment steps to clear {FRACTION:.0%} of optimal at slip={TARGET} "
            f"zero-shot: {steps[1]:,.0f} for the curriculum against {steps[0]:,.0f} for fixed DR "
            f"over Uniform[0, {TARGET}], {100 * (1 - steps[1] / steps[0]):.0f}% fewer, both "
            f"reaching it on {sum(1 for r in curri if r['steps'])}/{SEEDS} seeds",
        ),
        practice.Check(
            "FINDING: it wins on step count while barely winning on episodes",
            eps[1] <= eps[0] and steps[1] / eps[1] < steps[0] / eps[0],
            f"{eps[1]:.0f} episodes against {eps[0]:.0f}. The saving is in episode length: "
            f"{steps[1] / eps[1]:.1f} steps per episode against {steps[0] / eps[0]:.1f}, because "
            f"an episode at slip 0 terminates sooner than one drawn from the full range -- the "
            "curriculum practises in a cheaper environment, not a faster one",
        ),
        practice.Check(
            "FINDING: it passes while still six times narrower than the target",
            0 < reach < TARGET / 2,
            f"when the curriculum clears the rule at slip={TARGET}, its own ceiling is {reach} "
            f"-- it has never trained above {TARGET / reach:.0f}x less slip than it is scored "
            "on, which makes 'zero-shot' the right word for the whole comparison",
        ),
        practice.Check(
            "FINDING: the widening rule almost never fires",
            reach <= 2 * WIDEN,
            f"reaching {TARGET} in steps of {WIDEN} would take {round(TARGET / WIDEN)} "
            f"promotions; the run ends after {round(reach / WIDEN)}. What the exercise calls a "
            "curriculum is, here, a warm-up at slip 0 and an early exit -- the schedule never "
            "gets far enough to be a schedule",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
