"""Exercise 1 — most of the curve is the environment, not the agent.

    **Easy.** Train a Q-learning agent on the fixed-slip GridWorld (slip=0.0).
    Evaluate on slip ∈ {0.0, 0.1, 0.3, 0.5}. Plot return vs slip.

Reading of the exercise: the lesson's `evaluate` samples 200 episodes, which adds
noise to a quantity that can be computed exactly -- the greedy policy and the slip
together define a Markov chain, so its expected return is a linear solve. Both are
reported, and the sweep is placed against the *optimal* return at each slip, because
a curve that falls with slip proves nothing until you know how much of the fall was
unavoidable. The plot ships as a table, per `D14`.

**ANSWER: -8.00, -8.90, -11.57, -16.60 across the sweep.** The agent loses 8.60
return between slip 0 and slip 0.5.

**FINDING: 7.67 of that 8.60 is the environment getting harder.** The best any
policy can do at those slips is -8.00, -8.79, -11.15, -15.67. Only **0.93** of the
drop is the agent being wrong -- 11% of the decline the plot shows.

**FINDING: the degradation is entirely out-of-distribution, by construction.** At
the slip it trained on the agent is exactly optimal (-8.000 against -8.000), so
every point of the gap appears only at slips it never saw.

**FINDING: sampling 200 episodes is not needed here.** The greedy policy induces a
Markov chain whose expected return solves exactly; the lesson's sampled `evaluate`
agrees to within 0.2 but carries noise that is the same size as the whole
optimality gap at slip 0.1.

Structure: `exact_value` sweeps the induced chain to convergence; `optimal` does the
same with a max over actions, giving the ceiling each column is measured against.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "11-sim-to-real-transfer"
SLIPS, SEEDS, EPISODES = (0.0, 0.1, 0.3, 0.5), 8, 3_000
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
    return sweep_values(ref, lambda s: [max(ref.ACTIONS, key=lambda a: table[s][a])], slip)


def optimal(ref, slip):
    """The best expected return any policy can achieve at `slip`."""
    return sweep_values(ref, lambda _s: ref.ACTIONS, slip)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tables = [ref.train_fixed(0.0, episodes=EPISODES, rng=random.Random(s)) for s in range(SEEDS)]
    exact = {sl: statistics.fmean(exact_value(ref, t, sl) for t in tables) for sl in SLIPS}
    sampled = {sl: statistics.fmean(ref.evaluate(t, sl, rng=random.Random(1000 + i))
                                    for i, t in enumerate(tables)) for sl in SLIPS}
    return {"exact": exact, "sampled": sampled, "best": {sl: optimal(ref, sl) for sl in SLIPS}}


def verify(result):
    exact, best, sampled = result["exact"], result["best"], result["sampled"]
    drop = exact[SLIPS[0]] - exact[SLIPS[-1]]
    unavoidable = best[SLIPS[0]] - best[SLIPS[-1]]
    gaps = [best[sl] - exact[sl] for sl in SLIPS]
    noise = max(abs(sampled[sl] - exact[sl]) for sl in SLIPS)
    return [
        practice.Check(
            "ANSWER: -8.00, -8.90, -11.57, -16.60 across the sweep",
            all(exact[a] > exact[b] for a, b in zip(SLIPS, SLIPS[1:])),
            "expected return of the trained greedy policy, exactly, at slip = "
            + f"{SLIPS}: " + ", ".join(f"{exact[sl]:.2f}" for sl in SLIPS)
            + f". The agent loses {drop:.2f} between slip {SLIPS[0]} and slip {SLIPS[-1]}, "
            f"averaged over {SEEDS} seeds of {EPISODES:,} training episodes",
        ),
        practice.Check(
            "FINDING: 7.67 of that 8.60 is the environment getting harder",
            unavoidable > 0.8 * drop,
            "the best any policy can do at those slips is "
            + ", ".join(f"{best[sl]:.2f}" for sl in SLIPS)
            + f", so {unavoidable:.2f} of the {drop:.2f} decline is unavoidable and only "
            f"{drop - unavoidable:.2f} of it -- {100 * (drop - unavoidable) / drop:.0f}% -- is "
            "the agent being wrong. A curve that falls with slip proves nothing until it is "
            "placed against the curve that has to fall",
        ),
        practice.Check(
            "FINDING: the degradation is entirely out-of-distribution, by construction",
            abs(gaps[0]) < 1e-9 and max(gaps) > 0.5,
            f"at the slip it trained on the agent is exactly optimal: {exact[SLIPS[0]]:.3f} "
            f"against {best[SLIPS[0]]:.3f}, a gap of {gaps[0]:.1e}. The optimality gap by slip is "
            + ", ".join(f"{g:.3f}" for g in gaps)
            + " -- every point of it appears only at slips the agent never saw, which is what "
            "makes this a sim-to-real measurement rather than a training-quality one",
        ),
        practice.Check(
            "FINDING: sampling 200 episodes is not needed here",
            noise > gaps[1],
            f"the greedy policy and the slip define a Markov chain, so the expected return is a "
            f"fixed point rather than a sample mean. The lesson's `evaluate` agrees with the "
            f"exact value to within {noise:.2f} -- but that disagreement is larger than the "
            f"whole optimality gap at slip {SLIPS[1]} ({gaps[1]:.3f}), so at the low end of the "
            "sweep the measurement noise exceeds the effect being measured",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
