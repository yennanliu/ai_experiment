"""Exercise 1 — three CTDE rows are one policy, and it splits pellets one agent should take.

    Run `code/main.py`. Measure the steps-to-goal gap between independent and
    MAPPO-style agents. Does the gap grow or shrink on a 6x6 grid?

Reading of the exercise: the gap is measured on the demo's own 500 seeded
episodes at GRID=4 and GRID=6 (the module global every function reads), in
steps and as a share of the independent baseline, and both rows are scored
against the true joint optimum found by breadth-first search.

**ANSWER: the gap grows in steps and holds in share.** 4x4: independent 3.212,
MAPPO-style 2.726, a gap of 0.486 steps (15.1%). 6x6: 4.654 against 4.014, a
gap of 0.640 steps (13.8%). Longer walks give duplicated effort more steps to
waste; the fraction of steps wasted stays roughly the same.

**FINDING: the lesson's expected output is off by nearly 2x.** It promises
"independent agents take ~6 steps on average; CTDE variants converge toward
~3.5 steps (optimal for the 4x4 grid is 3)". The runs give 3.212 and 2.726,
and the BFS optimum averages 2.698 -- below the stated optimum of 3, and
reached exactly in only 204 of 500 episodes.

**FINDING: MADDPG, QMIX and MAPPO are one policy.** `run_mappo_style` returns
`run_maddpg_style(env)`, and `run_qmix_style`'s one-pellet branch returns the
same `(only, only)` that `_assigned_targets` does. Nothing is learned in any
of them, and the three rows agree on 500 of 500 episodes at both sizes.

**FINDING: the "centralized critic" is not optimal and sometimes loses.**
It gives each agent a *distinct* pellet, choosing the split with the smaller
*total* distance, but steps-to-goal is the *longer* of the two walks. It is
worse than the optimum in 13 episodes at 4x4 and 31 at 6x6.
- In 12 and 26 of those, a split with a shorter longest walk existed.
- In the rest (1 and 5), the best plan has one agent take both pellets,
  which the critic never allows.

It ties independent in 378 of 500 episodes. On seed 262 independent wins,
2 steps to 3: agent 0 sits next to both pellets and collects them alone.

**FINDING: the takeaways describe mechanics the code lacks.** "Only the closer
agent moves per step": `move_or_wait` is defined and never called, so both
agents move every step. `Env`'s docstring charges collisions an extra step,
and no line implements it.

Structure: `bench()` re-runs the demo's seeds at a grid size; `optimum()` is
BFS over joint positions, where each agent can stay or move one cell.
"""

from __future__ import annotations

import collections
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "20-marl-maddpg-qmix-mappo"
TRIALS = 500


def optimum(ref, env):
    """Fewest steps for both pellets over all joint plans (BFS)."""
    grid = ref.GRID

    def moves(p):
        steps = [(p[0] + dx, p[1] + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
        return {(min(grid - 1, max(0, x)), min(grid - 1, max(0, y))) for x, y in steps}

    start = (env.agent0, env.agent1, frozenset((env.pellet0, env.pellet1)))
    seen, frontier = {start}, collections.deque([(start, 0)])
    while frontier:
        (a, b, left), depth = frontier.popleft()
        if not left:
            return depth
        for state in {(na, nb, left - {na, nb}) for na in moves(a) for nb in moves(b)} - seen:
            seen.add(state)
            frontier.append((state, depth + 1))
    return None


def best_split(ref, env):
    """Steps for the distinct-pellet split with the shorter longest walk."""
    m, a, b, p, q = ref.manhattan, env.agent0, env.agent1, env.pellet0, env.pellet1
    return min(max(m(a, p), m(b, q)), max(m(a, q), m(b, p)))


def bench(ref, grid):
    ref.GRID = grid
    runners = {"independent": ref.run_independent, "maddpg": ref.run_maddpg_style,
               "qmix": ref.run_qmix_style, "mappo": ref.run_mappo_style,
               "optimum": lambda env: optimum(ref, env), "split": lambda env: best_split(ref, env)}
    return {name: [run(ref.Env.new(random.Random(i))) for i in range(TRIALS)]
            for name, run in runners.items()}


def summary(runs):
    ind, mappo, best = runs["independent"], runs["mappo"], runs["optimum"]
    return {
        "mean": {k: round(sum(v) / TRIALS, 3) for k, v in runs.items()},
        "identical": sum(a == b == c for a, b, c in zip(runs["maddpg"], runs["qmix"], mappo)),
        "above_optimum": sum(m > o for m, o in zip(mappo, best)),
        "ties": sum(i == m for i, m in zip(ind, mappo)),
        "ind_wins": [s for s, (i, m) in enumerate(zip(ind, mappo)) if i < m],
        "optimal_three": best.count(3),
        "split_better": sum(m > o and s < m for m, o, s in zip(mappo, best, runs["split"])),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    original = ref.GRID
    try:
        sizes = {grid: summary(bench(ref, grid)) for grid in (4, 6)}
    finally:
        ref.GRID = original
    source = inspect.getsource(ref)
    return {"sizes": sizes, "doc": parity.doc_text(PHASE, LESSON),
            "move_or_wait_calls": source.count("move_or_wait(") - 1,
            "collision_code": source.count("collision") - 1}


def gap(size):
    mean = size["mean"]
    return round(mean["independent"] - mean["mappo"], 3), mean["independent"]


def verify(result):
    four, six = result["sizes"][4], result["sizes"][6]
    (g4, i4), (g6, i6) = gap(four), gap(six)
    return [
        practice.Check(
            "ANSWER: the gap grows in steps and holds in share",
            g4 == 0.486 and g6 == 0.64 and abs(g4 / i4 - g6 / i6) < 0.02,
            f"4x4 {four['mean']['independent']} vs {four['mean']['mappo']} = {g4} steps "
            f"({g4 / i4:.1%}); 6x6 {six['mean']['independent']} vs {six['mean']['mappo']} "
            f"= {g6} steps ({g6 / i6:.1%})",
        ),
        practice.Check(
            "FINDING: the lesson's expected output is off by nearly 2x",
            "~6 steps" in result["doc"] and four["mean"]["optimum"] < 3,
            f"the lesson promises ~6, ~3.5 and an optimum of 3; the runs give "
            f"{four['mean']['independent']} and {four['mean']['mappo']}, the BFS optimum "
            f"averages {four['mean']['optimum']} and equals 3 in {four['optimal_three']} of 500",
        ),
        practice.Check(
            "FINDING: MADDPG, QMIX and MAPPO are one policy",
            four["identical"] == six["identical"] == TRIALS,
            f"the three CTDE rows agree on {four['identical']} of {TRIALS} episodes at 4x4 "
            f"and {six['identical']} at 6x6 -- run_mappo_style returns run_maddpg_style",
        ),
        practice.Check(
            "FINDING: the centralized critic is not optimal and sometimes loses",
            four["above_optimum"] == 13 and six["above_optimum"] == 31
            and (four["split_better"], six["split_better"]) == (12, 26)
            and 262 in four["ind_wins"],
            f"it exceeds the optimum in {four['above_optimum']} episodes at 4x4 and "
            f"{six['above_optimum']} at 6x6 -- {four['split_better']} and "
            f"{six['split_better']} of them because it minimises the sum, not the longer "
            f"walk; it ties independent in {four['ties']} of 500 and loses on seeds "
            f"{four['ind_wins']}",
        ),
        practice.Check(
            "FINDING: the takeaways describe mechanics the code lacks",
            result["move_or_wait_calls"] == 0 and result["collision_code"] == 0,
            "move_or_wait is defined and called 0 times, so both agents move every step; "
            "the collision cost appears only in Env's docstring",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
