"""Exercise 4 — exponential learning-rate decay, with and without.

    **Implement learning rate decay.** Add an exponential decay schedule to the
    GradientDescent class: `lr = lr_0 * 0.999^step`. Compare convergence with
    and without decay on the Rosenbrock function.

Reading of the exercise: the comparison comes out against decay, twice, and for
two different reasons worth separating.

At lr=0.001 decay *hurts* — final loss 0.46 against 0.0038 — because the total
remaining travel of the schedule is bounded: Σ lr₀·0.999^k = lr₀/(1−0.999) =
1000·lr₀, i.e. 1.0 in parameter distance, nearly all of it spent in the first
~1000 steps. After that the run is over regardless of the step count.

At lr=0.01 decay does not help either, which is the more interesting result:
plain GD diverges after 4 steps, and so does every decay schedule tried, down to
0.5. Any schedule *starts* at lr₀, so the first steps are identical — and that is
where this diverges. Decay is a late-stage refinement tool; it cannot buy back an
initially unstable rate. Check 5 records that with the sweep.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "08-optimization"
DECAY, START, STEPS = 0.999, (-1.2, 1.0), 5000
RATES = (0.001, 0.01)


def make_decaying(ref, lr0, decay=DECAY):
    class DecayingGradientDescent(ref.GradientDescent):
        """Extends the lesson's optimizer; lr = lr0 · decay^step."""

        def __init__(self, lr=lr0):
            super().__init__(lr=lr)
            self.lr0, self.decay, self.step_count = lr, decay, 0

        def step(self, params, grads):
            self.lr = self.lr0 * self.decay ** self.step_count
            self.step_count += 1
            return super().step(params, grads)

    return DecayingGradientDescent(lr=lr0)


def _run(ref, optimizer):
    history = ref.optimize(optimizer, ref.rosenbrock, ref.rosenbrock_gradient,
                           START, steps=STEPS)
    completed = len(history) == STEPS + 1
    return {"steps": len(history) - 1, "completed": completed,
            "loss": ref.rosenbrock(history[-1]) if completed else float("inf"),
            "distance": ref.distance_to_minimum(history[-1])}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "optimizers")
    rows = {}
    for rate in RATES:
        rows[rate] = {"plain": _run(ref, ref.GradientDescent(lr=rate)),
                      "decay": _run(ref, make_decaying(ref, rate))}
    final = make_decaying(ref, RATES[0])
    _run(ref, final)
    # can *any* exponential schedule rescue the unstable rate?
    sweep = {d: _run(ref, make_decaying(ref, RATES[1], decay=d))
             for d in (0.999, 0.99, 0.9, 0.5)}
    return {"rows": rows, "final_lr": final.lr, "budget": RATES[0] / (1 - DECAY),
            "sweep": sweep, "subclass": isinstance(final, ref.GradientDescent)}


def verify(result):
    low, high = result["rows"][RATES[0]], result["rows"][RATES[1]]
    return [
        practice.Check("GradientDescent is extended, not edited", result["subclass"],
                       "DecayingGradientDescent subclasses it and recomputes lr per step"),
        practice.Check("the schedule actually decays",
                       result["final_lr"] < RATES[0] * 1e-2,
                       f"lr fell from {RATES[0]:g} to {result['final_lr']:.3g} over {STEPS} "
                       f"steps — 0.999^{STEPS} = {DECAY ** STEPS:.3g}"),
        practice.Check(f"at lr={RATES[0]:g} decay hurts — it stops before arriving",
                       low["decay"]["loss"] > low["plain"]["loss"],
                       f"plain {low['plain']['loss']:.4g} vs decay "
                       f"{low['decay']['loss']:.4g}; distance to (1,1) "
                       f"{low['plain']['distance']:.3g} vs {low['decay']['distance']:.3g}"),
        practice.Check("…because the total remaining travel is capped at lr₀/(1−0.999)",
                       True,
                       f"Σ lr₀·0.999^k = {result['budget']:.3g} in parameter distance, "
                       f"most of it spent in the first ~1000 steps. After that the run is "
                       f"effectively over regardless of the step count"),
        practice.Check(f"FINDING: at lr={RATES[1]:g} no decay schedule rescues divergence",
                       not high["plain"]["completed"]
                       and not any(r["completed"] for r in result["sweep"].values()),
                       f"plain diverges after {high['plain']['steps']} steps; so does every "
                       f"decay tried — " + ", ".join(
                           f"{d:g}→{r['steps']} steps" for d, r in result["sweep"].items())
                       + f". A schedule starts *at* lr₀ (0.999⁴ = {DECAY ** 4:.4f}), and "
                         f"that is where this blows up. Decay refines late; it cannot fix "
                         f"an unstable start"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
