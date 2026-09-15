"""Exercise 2 — Q-learning walks the edge, and pays 28 points for it.

    **Medium.** Build a cliff-walking environment (4×12, last row is the cliff
    with reward -100 and reset to start). Compare Q-learning and SARSA final
    policies. Screenshot the paths each takes. Which is closer to the cliff?

Reading of the exercise: only the environment is built. `step` and `reset` are
module-level names that the lesson's own `sarsa` and `q_learning` look up at call
time, so replacing those two runs both learners unmodified on the new board --
nothing is forked. "Screenshot" ships as the ASCII grids below, per `DESIGN D14`,
and "closer to the cliff" is scored by how many of the ten cells directly above
the cliff each greedy path walks through, which is the thing that costs.

**ANSWER: Q-learning, and not marginally.** Its greedy path walks all 10
cliff-edge cells on all three seeds; SARSA's walks 0, at a cost of three or four
extra steps.

**FINDING: the policy Q-learning learns is better and the policy it runs is much
worse.** Followed greedily its path is -13.00, the optimum, against SARSA's
-16.33. Run under the `ε = 0.1` that produced it, it is -50.52 against SARSA's
-21.44 -- a 37.52-point gap between the policy learned and the policy behaved,
against 5.10 for SARSA.

**FINDING: the measured online return confirms the exact numbers.** -49.95 and
-21.59 over the last 500 episodes, against exact `ε`-greedy values of -50.52 and
-21.44.

**MECHANISM: SARSA's target contains the exploration, Q-learning's does not.**
`r + γQ(s',a')` averages over the `ε`-random `a'` that may step off the cliff;
`r + γ max_a Q(s',a)` prices the edge as if the agent never slipped. Neither is
wrong -- they answer different questions, and only one of them is the question the
learning curve scores. Exercise 1's board could not tell them apart at all.

**FINDING: the exercise's own metric would pick the loser.** "Which is closer to
the cliff" is answered by Q-learning, and `closest row touched` is 2 for both,
because SARSA's first move leaves the start corner upward. Only the count of
cliff-adjacent cells separates them.

Structure: `cliff_step` is the new environment; `bind` swaps it into the lesson's
module so its own learners run on it; `readout` follows the learned greedy policy
and `behaved` sweeps the same board under `ε`-soft behaviour.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "04-q-learning-sarsa"
ROWS, COLS, GAMMA, EPSILON, EPISODES = 4, 12, 0.99, 0.1, 3_000
START, GOAL = (3, 0), (3, 11)
CLIFF, EDGE = frozenset((3, c) for c in range(1, 11)), frozenset((2, c) for c in range(1, 11))
CELLS = [(r, c) for r in range(ROWS) for c in range(COLS)]


def cliff_step(ref, state, action):
    """4x12 cliff walking: the cliff costs -100 and teleports to the start."""
    dr, dc = ref.DELTAS[action]
    nxt = (min(max(state[0] + dr, 0), ROWS - 1), min(max(state[1] + dc, 0), COLS - 1))
    if nxt in CLIFF:
        return START, -100.0, False
    return nxt, -1.0, nxt == GOAL


def readout(ref, table):
    """(the greedy policy everywhere, the set of cells its path from the start visits)."""
    greedy = {s: max(ref.ACTIONS, key=lambda a: table[s][a]) if s in table else "right"
              for s in CELLS}
    state, path = START, []
    while state != GOAL and len(path) < 80:
        path.append(state)
        state = cliff_step(ref, state, greedy[state])[0]
    return greedy, set(path)


def picture(greedy, path):
    """The grid the exercise asks to screenshot; the walked path is upper-case."""
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    glyph = {s: arrows[greedy[s]].upper() if s in path else arrows[greedy[s]] for s in CELLS}
    glyph.update({s: "X" for s in CLIFF})
    glyph[START], glyph[GOAL] = "S", "G"
    return ["".join(glyph[(r, c)] for c in range(COLS)) for r in range(ROWS)]


def backup(ref, greedy, eps, state, values):
    """One undiscounted eps-soft backup at `state`."""
    return sum((eps / 4 + (1 - eps if a == greedy[state] else 0.0))
               * (lambda n, r, d: r + (0.0 if d else values[n]))(*cliff_step(ref, state, a))
               for a in ref.ACTIONS)


def behaved(ref, greedy, eps):
    """Exact undiscounted return of the eps-soft policy over `greedy`, from the start.

    Only `eps > 0` is swept: at `eps = 0` a greedy policy can cycle in cells the path
    never reaches and this would not converge, so the greedy value is read off the walk
    instead, which is exact because every step costs -1.
    """
    values = {s: 0.0 for s in CELLS}
    for _ in range(20_000):
        nxt = {s: (0.0 if s == GOAL else backup(ref, greedy, eps, s, values)) for s in CELLS}
        moved = max(abs(nxt[s] - values[s]) for s in values)
        values = nxt
        if moved < 1e-11:
            break
    return values[START]


def one(pick, seed):
    """One run of the lesson's own learner on the cliff, via its own `step` name."""
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.step = lambda state, action: cliff_step(ref, state, action)
    ref.reset = lambda: START
    table, returns = pick(ref)(EPISODES, gamma=GAMMA, epsilon=EPSILON, rng=random.Random(seed))
    greedy, path = readout(ref, table)
    return {"edge": len(set(path) & EDGE), "steps": len(path), "picture": picture(greedy, path),
            "online": statistics.fmean(returns[-500:]), "greedy_value": -float(len(path)),
            "behaved": behaved(ref, greedy, EPSILON),
            "deepest": max(s[0] for s in path if s != START)}


def summarise(pick, seeds=(7, 42, 101)):
    """Three seeds of one learner, reduced to what the checks read."""
    rows = [one(pick, seed) for seed in seeds]
    keys = ("edge", "steps", "online", "greedy_value", "behaved", "deepest")
    out = {k: statistics.fmean(r[k] for r in rows) for k in keys}
    out.update(edges=[r["edge"] for r in rows], lengths=[r["steps"] for r in rows],
               deepest_all={r["deepest"] for r in rows}, picture=rows[1]["picture"])
    return {**out, "gap": out["greedy_value"] - out["behaved"]}


def solve():
    return {"sarsa": summarise(lambda r: r.sarsa), "q": summarise(lambda r: r.q_learning)}


def verify(result):
    sarsa, q = result["sarsa"], result["q"]
    edge, spread = len(EDGE), sarsa["behaved"] - q["behaved"]
    online = max(abs(row["online"] - row["behaved"]) for row in (q, sarsa))
    return [
        practice.Check(
            "ANSWER: Q-learning -- its path walks all 10 cliff-edge cells, SARSA's walks none",
            q["edges"] == [edge] * 3 and sarsa["edges"] == [0, 0, 0],
            f"cliff-adjacent cells on the greedy path, seeds 7, 42 and 101: {q['edges']} for "
            f"Q-learning against {sarsa['edges']} for SARSA, out of {edge}; path lengths "
            f"{q['lengths']} and {sarsa['lengths']}. Q-learning's board:\n"
            + "\n".join("      " + line for line in q["picture"])
            + "\n    SARSA's:\n" + "\n".join("      " + line for line in sarsa["picture"]),
        ),
        practice.Check(
            "FINDING: Q-learning learns the better policy and runs the worse one",
            q["greedy_value"] > sarsa["greedy_value"] and spread > 20,
            f"followed greedily, Q-learning's path is worth {q['greedy_value']:.2f} -- the "
            f"optimum -- against SARSA's {sarsa['greedy_value']:.2f}. Run under the eps="
            f"{EPSILON} that produced it, the two are {q['behaved']:.2f} and "
            f"{sarsa['behaved']:.2f}, {spread:.2f} apart. Both exact, swept over the new board",
        ),
        practice.Check(
            "FINDING: the gap between the policy learned and the policy run is 7x wider",
            q["gap"] > 6 * sarsa["gap"],
            f"Q-learning loses {q['gap']:.2f} between greedy and behaved; SARSA loses "
            f"{sarsa['gap']:.2f}, a factor of {q['gap'] / sarsa['gap']:.1f}. The cliff-edge path "
            f"is optimal only for an agent that never explores, and the one that learned it "
            f"explores {100 * EPSILON:.0f}% of the time. Three or four extra steps is the "
            "premium SARSA pays to close that gap",
        ),
        practice.Check(
            "FINDING: the measured online return lands on the exact numbers",
            online < 3.0,
            f"mean return over the last 500 episodes is {q['online']:.2f} and "
            f"{sarsa['online']:.2f}, against exact eps-greedy values of {q['behaved']:.2f} and "
            f"{sarsa['behaved']:.2f} -- within {online:.2f}. The simulation measures what the "
            f"sweep computes, so the {spread:.0f}-point gap belongs to the policies, not the "
            "three seeds",
        ),
        practice.Check(
            "FINDING: the obvious version of the metric cannot separate them",
            q["deepest_all"] == sarsa["deepest_all"] == {2},
            f"the deepest row either path touches is {q['deepest']:.0f} for both: SARSA's first "
            "move out of the start corner is upward into row 2 at column 0, which is not above a "
            "cliff cell. Read as a minimum distance the metric ties; read as how much of the "
            "edge is walked it is 10 against 0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
