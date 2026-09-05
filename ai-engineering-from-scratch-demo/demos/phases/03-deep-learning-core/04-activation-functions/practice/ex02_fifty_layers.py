"""Exercise 2 — the vanishing-gradient experiment at 50 layers.

    Run the vanishing gradient experiment with 50 layers instead of 10. Plot the
    magnitude at each layer for sigmoid, tanh, ReLU, and GELU. At which layer does
    each activation's signal effectively reach zero?

Reading of the exercise: "the magnitude" is whatever the lesson's own
`vanishing_gradient_experiment` prints — the forward activation — so checks 1-2
answer that question exactly, and invert the lesson's story. Check 3 computes the
gradient the section is named after on the same trace, because the two disagree;
check 4 measures the per-layer factor against the lesson's 0.25; check 5 re-runs
on 400 other streams, since one seeded walk cannot make "at which layer" a
property of the activation. The README tabulates the traces the exercise plots.
"""

from __future__ import annotations

import math
import random
import types

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "04-activation-functions"
LAYERS, SEEDS, GONE, DEAD = 50, 400, 1e-6, 1e-300
# geometric mean of |sum of 5 N(0,1) weights|: exp(E[log|N(0, sqrt5)|])
WEIGHT_GAIN = math.exp(math.log(5) / 2 - (0.5772156649015329 + math.log(2)) / 2)


def trace(ref, act, deriv, rng=None):
    """The lesson's experiment: per-layer |act(z)|, and the multiplier act'(z)*sum(w)."""
    seen = []

    def spy(z):
        seen.append((z, act(z)))
        return seen[-1][1]

    saved = ref.random   # the lesson hard-codes seed(42); the ensemble needs that ignored
    ref.random = saved if rng is None else types.SimpleNamespace(
        seed=lambda *_a: None, gauss=rng.gauss)
    try:
        with parity.quiet():
            ref.vanishing_gradient_experiment(spy, "scan", n_layers=LAYERS)
    finally:
        ref.random = saved
    # z_l = act(z_{l-1}) * sum(w_l), so z_l / act(z_{l-1}) recovers that weight sum
    return [abs(v) for _z, v in seen], [abs(deriv(z) * z / seen[i][1])
                                        for i, (z, _v) in enumerate(seen[1:]) if seen[i][1]]


def first_below(mags, threshold):
    return next((i + 1 for i, m in enumerate(mags) if m < threshold), None)


def study(ref, act, deriv):
    """Seed 42 — the lesson's own — plus SEEDS independently seeded streams."""
    mags, chain = trace(ref, act, deriv)
    deaths, steps = [], []
    for seed in range(SEEDS):
        other, more = trace(ref, act, deriv, random.Random(1000 + seed))
        deaths.append(first_below(other, DEAD))
        steps.extend(more)
    died, live = sorted(d for d in deaths if d), [math.log(s) for s in steps if s > 0]
    return {"zero": first_below(mags, DEAD), "final": mags[-1], "low": min(mags),
            "gone": first_below(mags, GONE), "low_at": mags.index(min(mags)) + 1,
            "grad": math.prod(chain), "died": len(died), "live": len(live),
            "median": died[len(died) // 2] if died else None,
            "factor": math.exp(sum(live) / len(live))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    acts = [("sigmoid", ref.sigmoid, ref.sigmoid_derivative),
            ("tanh", ref.tanh_act, ref.tanh_derivative),
            ("relu", ref.relu, ref.relu_derivative),
            ("gelu", ref.gelu, ref.gelu_derivative)]
    return {name: study(ref, a, d) for name, a, d in acts}


def verify(result):
    sig, tanh = result["sigmoid"], result["tanh"]
    relu, gelu = result["relu"], result["gelu"]
    return [
        practice.Check("ANSWER: at 50 layers only ReLU's signal reaches zero, and exactly",
                       all((relu["zero"] == 6, gelu["gone"] == 7,
                            sig["gone"] is None, tanh["gone"] is None)),
                       f"seed 42, the lesson's own: ReLU is exactly 0.0 from layer "
                       f"{relu['zero']} on; GELU falls below {GONE:g} at layer "
                       f"{gelu['gone']} and ends {gelu['final']:.2e} without ever being 0; "
                       f"tanh bottoms {tanh['low']:.2e} (layer {tanh['low_at']}) and "
                       f"recovers to {tanh['final']:.2e}; sigmoid never falls below "
                       f"{sig['low']:.4f}, ending {sig['final']:.4f}"),
        practice.Check("FINDING: that is the lesson's ranking upside down",
                       sig["final"] > 0.5 > relu["final"] and sig["low"] > 0.07,
                       f"the activation the lesson blames for vanishing gradients is the "
                       f"only one whose signal survives 50 layers ({sig['final']:.4f}); the "
                       f"one it credits with fixing them is the only one that dies "
                       f"({relu['final']:.1f}). MECHANISM: sigmoid(z) = 0.5 + z/4 + O(z^3) "
                       f"pushes a fading signal back towards 0.5, so the non-zero-centred "
                       f"output listed as sigmoid's flaw is what keeps the signal alive; "
                       f"relu(0) = 0 absorbs, since the next z is 0 * sum(w) = 0"),
        practice.Check("…and the gradient, on the same trace, ranks them the other way",
                       sig["grad"] < 1e-30 < tanh["grad"] and relu["grad"] == 0.0,
                       f"prod of act'(z_l) * sum(w_l): sigmoid {sig['grad']:.3e}, gelu "
                       f"{gelu['grad']:.3e}, tanh {tanh['grad']:.3e}, relu "
                       f"{relu['grad']:.1f} — sigmoid's gradient is 36 orders below tanh's "
                       f"while its signal is the healthiest of the four, so the printed "
                       f"magnitude and the quantity in the title are different things"),
        practice.Check("measured per-layer factor against the textbook 0.25",
                       0.21 < sig["factor"] < 0.23
                       and abs(relu["factor"] / WEIGHT_GAIN - 1) < 0.15,
                       f"geometric mean of act'(z)*sum(w) over {SEEDS} runs: sigmoid "
                       f"{sig['factor']:.4f} against the lesson's 0.25 ceiling, tanh "
                       f"{tanh['factor']:.4f} against its 1.0, gelu {gelu['factor']:.4f}, "
                       f"relu {relu['factor']:.4f} over its {relu['live']} surviving layers "
                       f"where act' = 1 — the weight sum alone, matching "
                       f"exp(E log|N(0, sqrt5)|) = {WEIGHT_GAIN:.4f}. Seed 42 chains sigmoid "
                       f"to {sig['grad']:.2e}, 8 orders below the 0.25^49 = "
                       f"{0.25 ** (LAYERS - 1):.2e} the lesson's arithmetic predicts"),
        practice.Check("CONTROL: 'layer 6' is a coin flip, not a property of ReLU",
                       all((relu["died"] == SEEDS, relu["median"] <= 3, sig["died"] == 0)),
                       f"over {SEEDS} independent streams ReLU reaches exactly 0 in "
                       f"{relu['died']}/{SEEDS} runs, median layer {relu['median']} — it "
                       f"dies the first time sum(w) is negative, so the layer index is "
                       f"geometric with p = 1/2 and seed 42 merely got a long run. GELU "
                       f"dies in {gelu['died']}/{SEEDS}; sigmoid and tanh in "
                       f"{sig['died']}, {tanh['died']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
