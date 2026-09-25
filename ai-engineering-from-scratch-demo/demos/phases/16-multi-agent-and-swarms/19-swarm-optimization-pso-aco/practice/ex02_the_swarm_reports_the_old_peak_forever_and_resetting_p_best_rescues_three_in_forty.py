"""Exercise 2 — the swarm reports the old peak forever, and resetting p_best rescues 3 in 40.

    Implement a "catastrophic drift" experiment: after iteration 30, change the
    fitness function. How fast does PSO adapt? Does resetting `p_best` help?

Reading of the exercise: the drift moves the fitness landscape's centre from
(0.72, 0.40) to (0.30, 0.75) at iteration 30 of 60; "adapted" means the
*true* fitness of g_best under the new function is within 1e-3 of its
maximum. `run_lmpso` hardcodes `fitness` and returns only its history, so a
port is used for the variants and is checked to reproduce the reference
history exactly before any drift.

**ANSWER: without a reset it never adapts, and resetting `p_best` alone
barely helps.** Over 40 seeds at 20 particles: no reset adapts in 0 of 40,
re-evaluating every `p_best` under the new function in 3 of 40, doing that
and re-drawing velocities in 14 of 40, and re-seeding positions -- a restart
-- in 40 of 40, 6.2 iterations after the drift on average.

**FINDING: the reported fitness never drops.** On the reference `run_lmpso`,
with `fitness` swapped mid-run, the history reads 1.00945 from iteration 30
to 60 while g_best's true fitness is 0.0. A particle only replaces its
`p_best` when a new evaluation beats the *stored* value, and the new
function's maximum equals the old stored one, so `g_best` can never move --
a strict `>` against a number measured on a function that no longer exists.

**FINDING: the reset fails because the swarm sits on a plateau.** `fitness`
is clamped at 0, and 58% of the unit square scores exactly 0 under the moved
function -- including the old peak. A converged swarm has collapsed there:
re-evaluated, every `p_best` is 0, the velocities are near zero, and there is
no signal to follow. Resetting memory does not restore diversity; the 14 of
40 rescued by fresh velocities, and the 40 of 40 by fresh positions, say
diversity is what adaptation needs.

Structure: `swarm()` / `step()` port `run_lmpso` with the fitness passed in;
`drift()` applies one of four responses at the drift point.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "19-swarm-optimization-pso-aco"
TOP, AT, ITERS, SEEDS, N = 1.0094489833890408, 30, 60, range(40), 20
MODES = ("none", "reset_p_best", "reset_and_kick", "restart")


def moved(ref):
    """The landscape recentred on (0.30, 0.75); binds fitness now, not at call time."""
    base = ref.fitness
    return lambda x: base([x[0] + 0.42, x[1] - 0.35])


def fresh(rng, fit):
    x = [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
    return {"x": x, "v": [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)],
            "pb": list(x), "pf": fit(x)}


def best(swarm):
    top = max(swarm, key=lambda p: p["pf"])
    return list(top["pb"]), top["pf"]


def step(swarm, g, rng, fit):
    """One run_lmpso iteration, same draw order, same asynchronous g_best."""
    gx, gf = g
    for p in swarm:
        r1, r2 = rng.random(), rng.random()
        for d in range(2):
            p["v"][d] = 0.6 * p["v"][d] + 1.2 * r1 * (p["pb"][d] - p["x"][d]) \
                + 1.2 * r2 * (gx[d] - p["x"][d])
            p["x"][d] = max(0.0, min(1.0, p["x"][d] + p["v"][d]))
        f = fit(p["x"])
        if f > p["pf"]:
            p["pb"], p["pf"] = list(p["x"]), f
            gx, gf = (list(p["x"]), f) if f > gf else (gx, gf)
    return gx, gf


def drift(swarm, mode, rng, fit):
    for i, p in enumerate(swarm):
        if mode == "restart":
            swarm[i] = fresh(rng, fit)
        elif mode != "none":
            p["pf"] = fit(p["pb"])
            p["v"] = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)] \
                if mode == "reset_and_kick" else p["v"]


def run(ref, seed, mode=None, n=N, iters=ITERS):
    """(reported history, true fitness of g_best per iteration)."""
    rng, fit = random.Random(seed), ref.fitness
    swarm = [fresh(rng, fit) for _ in range(n)]
    g = best(swarm)
    reported, true = [g[1]], [g[1]]
    for it in range(iters):
        if mode and it == AT:
            fit = moved(ref)
            drift(swarm, mode, rng, fit)
            g = best(swarm) if mode != "none" else g
        g = step(swarm, g, rng, fit)
        reported.append(g[1])
        true.append(fit(g[0]))
    return reported, true


def patched_reference(ref):
    """The reference run_lmpso, with fitness swapped after iteration 30's calls."""
    original, calls, new = ref.fitness, [0], moved(ref)

    def switching(x):
        calls[0] += 1
        return original(x) if calls[0] <= N * (1 + AT) else new(x)

    ref.fitness = switching
    try:
        return ref.run_lmpso(N, ITERS, 0)
    finally:
        ref.fitness = original


def adaptation(ref):
    """{mode: iterations after the drift until g_best's true fitness tops out}."""
    waits = {m: [next((i - AT for i in range(AT + 1, ITERS + 1) if t[i] >= TOP - 1e-3), None)
                 for t in (run(ref, s, m)[1] for s in SEEDS)] for m in MODES}
    return {m: [w for w in ws if w is not None] for m, ws in waits.items()}


def plateau(ref):
    grid = [[i / 200, j / 200] for i in range(201) for j in range(201)]
    return sum(moved(ref)(p) == 0 for p in grid) / len(grid)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    adapted = adaptation(ref)
    return {
        "parity": all(run(ref, s, None, n, 30)[0] == ref.run_lmpso(n, 30, s)
                      for n in (5, 20) for s in range(5)),
        "adapted": {m: len(w) for m, w in adapted.items()},
        "restart_wait": statistics.mean(adapted["restart"]),
        "ref_tail": sorted({round(v, 5) for v in patched_reference(ref)[AT:]}),
        "true_after": run(ref, 0, "none")[1][-1],
        "plateau": plateau(ref),
        "old_peak": moved(ref)([0.72, 0.40]),
    }


def verify(result):
    a = result["adapted"]
    return [
        practice.Check(
            "ANSWER: without a reset it never adapts; resetting p_best alone barely helps",
            all([result["parity"], a == {"none": 0, "reset_p_best": 3,
                                         "reset_and_kick": 14, "restart": 40}]),
            f"adapted within 30 iterations, of 40 seeds: {a}; a restart takes "
            f"{result['restart_wait']:.1f} iterations; the port reproduces run_lmpso "
            "exactly before the drift",
        ),
        practice.Check(
            "FINDING: the reported fitness never drops",
            result["ref_tail"] == [1.00945] and result["true_after"] == 0.0,
            f"the reference history reads {result['ref_tail']} from iteration 30 to 60 "
            f"while g_best's true fitness is {result['true_after']} -- updates need a "
            "strict > against a value measured on the old function",
        ),
        practice.Check(
            "FINDING: the reset fails because the swarm sits on a plateau",
            result["plateau"] > 0.55 and result["old_peak"] == 0.0,
            f"{result['plateau']:.0%} of the unit square scores exactly 0 under the moved "
            f"function, the old peak included ({result['old_peak']}); re-evaluated "
            "memory is all zeros and the collapsed swarm has nothing to follow",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
