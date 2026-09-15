"""Exercise 2 — it buys 3% of the return and 60% of the gap.

    **Medium.** Train a DR Q-learning agent sampling `slip ~ Uniform[0, 0.3]`.
    Evaluate the same sweep. How much does DR buy at slip=0.5 (out-of-distribution)?

Reading of the exercise: "how much" has two defensible denominators and they disagree
by a factor of seventeen, so both are reported -- the change in return, and the share
of the optimality gap closed. Values are computed exactly rather than sampled, for
the reason exercise 1 gives, and the same eight seeds train both arms so the
comparison is paired.

**ANSWER: 0.56 of return, which is 60% of the gap.** At slip=0.5 the fixed-slip
agent returns -16.60 and the DR agent -16.04, against an optimum of -15.67. As a
fraction of the return that is **3.4%**; as a fraction of the distance to optimal it
is **60%**.

**FINDING: DR closes about 55-60% of the gap at every slip, including inside its own
range.** Gaps fall 0.111 -> 0.066 at slip 0.1, 0.428 -> 0.196 at 0.3 and
0.931 -> 0.371 at 0.5. Nothing special happens at the edge of the training
distribution.

**FINDING: the two agents are indistinguishable where they were trained.** Both
return exactly -8.000 at slip=0 -- both are shortest paths. DR's entire benefit is a
different *choice* among equally optimal policies, and no measurement taken in the
training environment can see it.

**MECHANISM: they disagree on 6 of 24 cells, and the disagreement is the route.**
The fixed-slip agent walks the boundary -- straight down the left column, then right
along the bottom. The DR agent takes a staircase through the interior. Both are
8 steps when nothing slips.

Structure: `exact_value` and `optimal` are exercise 1's sweeps; `route` reads the
greedy policy off a table so the two can be compared cell by cell.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "11-sim-to-real-transfer"
SLIPS, SEEDS, EPISODES = (0.0, 0.1, 0.3, 0.5), 8, 3_000
LOW, HIGH = 0.0, 0.3
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


def route(ref, table):
    """The greedy action at every cell."""
    return {s: max(ref.ACTIONS, key=lambda a: table[s][a]) for s in CELLS}


def exact_value(ref, table, slip):
    """Expected return of the greedy policy read off `table`."""
    policy = route(ref, table)
    return sweep_values(ref, lambda s: [policy[s]], slip)


def grid(ref, policy):
    """The policy as five rows of arrows."""
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    return [" ".join(arrows[policy[(r, c)]] if (r, c) != ref.TERMINAL else "."
                     for c in range(5)) for r in range(5)]


def curve(ref, tables):
    """One arm's exact expected return at every slip in the sweep."""
    return {sl: statistics.fmean(exact_value(ref, t, sl) for t in tables) for sl in SLIPS}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fixed = [ref.train_fixed(0.0, episodes=EPISODES, rng=random.Random(s)) for s in range(SEEDS)]
    wide = [ref.train_dr(LOW, HIGH, episodes=EPISODES, rng=random.Random(s)) for s in range(SEEDS)]
    routes = (route(ref, fixed[0]), route(ref, wide[0]))
    return {"fixed": curve(ref, fixed), "dr": curve(ref, wide), "cells": len(CELLS) - 1,
            "best": {sl: sweep_values(ref, lambda _s: ref.ACTIONS, sl) for sl in SLIPS},
            "differ": sum(1 for s in CELLS if s != ref.TERMINAL and routes[0][s] != routes[1][s]),
            "grids": (grid(ref, routes[0]), grid(ref, routes[1]))}


def gaps(result):
    """(fixed-arm gaps, DR-arm gaps, share of each closed) across the sweep."""
    fixed, dr, best = result["fixed"], result["dr"], result["best"]
    one = [best[sl] - fixed[sl] for sl in SLIPS]
    two = [best[sl] - dr[sl] for sl in SLIPS]
    return one, two, [1 - d / f if f else 0.0 for f, d in zip(one, two)]


def verify(result):
    fixed, dr, best = result["fixed"], result["dr"], result["best"]
    edge = SLIPS[-1]
    bought = dr[edge] - fixed[edge]
    gaps_f, gaps_d, closed = gaps(result)
    return [
        practice.Check(
            "ANSWER: 0.56 of return, which is 60% of the gap",
            bought > 0 and 0.02 < bought / abs(fixed[edge]) < 0.06 < closed[-1],
            f"at slip={edge} the fixed-slip agent returns {fixed[edge]:.2f} and the DR agent "
            f"{dr[edge]:.2f}, against an optimum of {best[edge]:.2f}. DR buys {bought:.2f} -- "
            f"{100 * bought / abs(fixed[edge]):.1f}% of the return, and "
            f"{100 * closed[-1]:.0f}% of the distance to optimal. The two denominators disagree "
            f"by a factor of {closed[-1] / (bought / abs(fixed[edge])):.0f}",
        ),
        practice.Check(
            "FINDING: DR closes 55-60% of the gap at every slip, inside its range and out",
            min(closed[1:]) > 0.4 and max(closed[1:]) < 0.75,
            "optimality gap, fixed against DR, by slip: "
            + ", ".join(f"{sl}: {f:.3f} -> {d:.3f}" for sl, f, d in zip(SLIPS, gaps_f, gaps_d))
            + " -- a reduction of " + ", ".join(f"{100 * c:.0f}%" for c in closed[1:])
            + f". slip={HIGH} is the edge of the training range and slip={edge} is well outside "
            "it, and nothing distinguishes them: whatever DR learned generalises at the same "
            "rate in and out of distribution",
        ),
        practice.Check(
            "FINDING: the two agents are indistinguishable where they were trained",
            abs(fixed[0.0] - dr[0.0]) < 1e-9,
            f"both return exactly {fixed[0.0]:.3f} at slip=0, matching to "
            f"{abs(fixed[0.0] - dr[0.0]):.1e} -- both are shortest paths. DR's entire benefit is "
            "a different choice among equally optimal policies, so no measurement taken in the "
            "training environment can see it. The exercise's own sweep only separates them at "
            "slips neither agent was asked about",
        ),
        practice.Check(
            "MECHANISM: they disagree on 6 of 24 cells, and the disagreement is the route",
            0 < result["differ"] < result["cells"] / 2,
            f"the greedy policies differ on {result['differ']} of {result['cells']} non-terminal "
            f"cells. Fixed-slip walks the boundary, straight down the left column then right "
            f"along the bottom; DR takes a staircase through the interior:\n"
            + "\n".join(f"      fixed  {a}      DR  {b}"
                        for a, b in zip(*result["grids"])),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
