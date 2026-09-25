"""Exercise 1 — iterations saturate at ten, and the peak they converge on is not the one named.

    Run `code/main.py`. Observe LMPSO convergence. Vary population size 5, 10,
    20, 50. At what size does time-to-converge saturate?

Reading of the exercise: "converged" is g_best within 1e-3 of the true
maximum of `fitness`, found by maximising it rather than trusting the demo's
printed optimum; time is counted both in iterations and in fitness
evaluations, since in LMPSO every evaluation is an LLM call. 40 seeds per size.

**ANSWER: iterations saturate at about 10 particles, and evaluations are
cheapest at 5-10.** Mean iterations to converge are 14.2, 7.3, 5.7 and 3.9 at
5, 10, 20 and 50 particles: doubling from 5 to 10 saves 6.9 iterations,
doubling again saves 1.6, and 2.5x more particles after that saves 1.9. In
evaluations -- n x (iterations + 1) -- the same runs cost 76, 83, 134 and 242,
so past 10 every extra particle buys fewer iterations than it costs calls.
The demo's own 20 particles converge at iteration 6 of its 30.

**FINDING: the printed optimum is a local minimum.** The demo prints
"optimum = 1.0000 at (0.72, 0.40)"; `fitness` there is 0.84, because the
ripple term subtracts 0.16 at its own centre. Per axis the curvature at the
centre is -12 + 0.08 x 64 pi^2 = +38.5, so (0.72, 0.40) is a minimum of the
separable fitness along both axes. The four true maxima sit at
(0.72 +/- 0.0997, 0.40 +/- 0.0997) with value 1.00945 -- which is why the
demo's own "final g_best" prints above the optimum it names.

**FINDING: the docstring's "narrow peak" is a ring of four.** Every seed and
size converges to 1.00945, not 0.84, so no run in 160 ever reports the named
point; the swarm is climbing out of the very spot the lesson says it finds.

Structure: `true_max()` maximises one axis of the separable fitness on a
1e-5 grid; `converge()` scores one reference `run_lmpso` history.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "19-swarm-optimization-pso-aco"
SIZES, SEEDS, ITERS, TOL = (5, 10, 20, 50), range(40), 60, 1e-3


def axis(d):
    """One axis of the separable fitness: -6 d^2 - 0.08 cos(8 pi d)."""
    return -6 * d * d - 0.08 * math.cos(8 * math.pi * d)


def true_max():
    best = max((i / 1e5 for i in range(-30000, 30001)), key=axis)
    return best, 1 + 2 * axis(best)


def converge(history, top):
    """First iteration whose g_best is within TOL of the true maximum."""
    return next((i for i, v in enumerate(history) if v >= top - TOL), None)


def sweep(ref, top):
    """{size: [iteration converged per seed]}, and every final g_best seen."""
    runs, finals = {}, set()
    for n in SIZES:
        histories = [ref.run_lmpso(n, ITERS, s) for s in SEEDS]
        runs[n] = [converge(h, top) for h in histories]
        finals |= {round(h[-1], 5) for h in histories}
    return runs, finals


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    offset, top = true_max()
    runs, finals = sweep(ref, top)
    done = {n: [t for t in r if t is not None] for n, r in runs.items()}
    return {
        "offset": offset, "top": top,
        "iters": {n: statistics.mean(d) for n, d in done.items()},
        "evals": {n: statistics.mean(n * (t + 1) for t in d) for n, d in done.items()},
        "missed": sum(len(r) - len(done[n]) for n, r in runs.items()),
        "demo": converge(ref.run_lmpso(), top),
        "centre": ref.fitness([0.72, 0.40]), "finals": finals,
        "curvature": -12 + 0.08 * 64 * math.pi ** 2,
    }


def verify(result):
    it, ev = result["iters"], result["evals"]
    gains = [it[5] - it[10], it[10] - it[20], it[20] - it[50]]
    return [
        practice.Check(
            "ANSWER: iterations saturate at about 10; evaluations are cheapest at 5-10",
            all([result["missed"] == 0, gains[0] > 3 * max(gains[1:]),
                 ev[10] < 1.2 * ev[5], ev[50] > 2.5 * ev[10], result["demo"] == 6]),
            f"mean iterations {[round(it[n], 1) for n in SIZES]} at {list(SIZES)}; "
            f"doubling 5->10 saves {gains[0]:.1f}, then {gains[1]:.1f} and {gains[2]:.1f}; "
            f"evaluations {[round(ev[n]) for n in SIZES]}; the demo's 20 particles "
            f"converge at iteration {result['demo']} of 30",
        ),
        practice.Check(
            "FINDING: the printed optimum is a local minimum",
            all([abs(result["centre"] - 0.84) < 1e-12, result["curvature"] > 0,
                 abs(abs(result["offset"]) - 0.0997) < 1e-4, result["top"] > 1.0]),
            f"fitness(0.72, 0.40) = {result['centre']:.2f}, per-axis curvature there "
            f"+{result['curvature']:.1f}; the maxima sit at offsets +/-"
            f"{abs(result['offset']):.4f} with value {result['top']:.5f}",
        ),
        practice.Check(
            "FINDING: the docstring's narrow peak is a ring of four",
            result["finals"] == {round(result["top"], 5)},
            f"all {len(SIZES) * len(SEEDS)} runs end at {sorted(result['finals'])}, "
            "none at 0.84 -- the swarm climbs out of the point the lesson says it finds",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
