"""Exercise 4 — Langevin in a double well: find the mixing temperature.

    **Compare Langevin dynamics at different temperatures.** Sample from a
    double-well potential U(x) = (x^2 - 1)^2. At low temperature, samples cluster
    in one well. At high temperature, they spread across both. Find the critical
    temperature where the chain mixes between wells.

Reading of the exercise: "the critical temperature" is not a single number but a
crossover, so it needs a definition to be findable. The one used here is the
lowest tested temperature at which the chain **crosses the barrier at least
once** — measured by counting sign changes of x, which is unambiguous for a
symmetric double well with minima at ±1. The barrier height is U(0) − U(±1) = 1,
so the crossing rate should turn over around T ≈ 1; check 4 reports where it
actually does.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "22-stochastic-processes"
SEED, DT, STEPS = 20260904, 0.005, 40_000
TEMPERATURES = (0.02, 0.1, 0.3, 1.0, 3.0)
BARRIER = 1.0                       # U(0) − U(±1) = 1 − 0 = 1


def grad_U(x):
    return 4 * x * (x * x - 1)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "stochastic")
    rows = {}
    for temperature in TEMPERATURES:
        trajectory = ref.langevin_dynamics(grad_U, numpy.array([1.0]), DT,
                                           temperature, STEPS, seed=SEED)
        xs = [float(v[0]) for v in trajectory]
        crossings = sum(1 for a, b in zip(xs, xs[1:]) if a * b < 0)
        right = sum(1 for v in xs if v > 0) / len(xs)
        rows[temperature] = {"crossings": crossings, "right_fraction": right,
                             "mean": sum(xs) / len(xs),
                             "spread": max(xs) - min(xs),
                             "balance": abs(right - 0.5)}
    mixing = next((t for t in TEMPERATURES if rows[t]["crossings"] > 0), None)
    balanced = next((t for t in TEMPERATURES if rows[t]["balance"] < 0.1), None)
    return {"rows": rows, "mixing": mixing, "balanced": balanced}


def verify(result):
    rows = result["rows"]
    cold, hot = rows[TEMPERATURES[0]], rows[TEMPERATURES[-1]]
    crossings = [rows[t]["crossings"] for t in TEMPERATURES]
    return [
        practice.Check(f"at T={TEMPERATURES[0]:g} the chain never leaves its well",
                       cold["crossings"] == 0 and cold["right_fraction"] == 1.0,
                       f"0 sign changes over {STEPS:,} steps, {cold['right_fraction']:.0%} "
                       f"of samples on the right, spread {cold['spread']:.3f} — started at "
                       f"x=+1 and stayed there, as the exercise predicts"),
        practice.Check(f"at T={TEMPERATURES[-1]:g} it crosses freely",
                       hot["crossings"] > 100,
                       f"{hot['crossings']:,} sign changes, {hot['right_fraction']:.0%} on "
                       f"the right, spread {hot['spread']:.2f}"),
        practice.Check("crossings rise monotonically with temperature",
                       all(a <= b for a, b in zip(crossings, crossings[1:])),
                       ", ".join(f"T={t:g}: {rows[t]['crossings']:,}"
                                 for t in TEMPERATURES)),
        practice.Check(f"ANSWER: mixing starts at T = {result['mixing']:g}, near the "
                       f"barrier height of {BARRIER:g}",
                       result["mixing"] is not None and 0.05 <= result["mixing"] <= 1.0,
                       f"the first temperature with any barrier crossing is "
                       f"{result['mixing']:g}. The barrier is U(0) − U(±1) = {BARRIER:g}, "
                       f"and Kramers' law makes the crossing rate ∝ exp(−ΔU/T) — so "
                       f"crossings become common once T approaches ΔU, not at some "
                       f"sharp threshold"),
        practice.Check("…and equal occupancy needs a higher temperature than first crossing",
                       result["balanced"] is not None
                       and result["balanced"] > result["mixing"],
                       f"the wells are only balanced (within 10% of 50/50) from "
                       f"T={result['balanced']:g}, above the T={result['mixing']:g} where "
                       f"crossing begins. 'Mixes between wells' and 'samples the "
                       f"distribution correctly' are different bars, and only the second "
                       f"is what a sampler needs"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
