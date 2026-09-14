"""Exercise 3 — the trade-off curve is bowed, and no nonlinearity is needed for that.

    **Hard.** Use diffusers to stack: SDXL-base + Canny-ControlNet (weight 0.8) +
    a style LoRA (α 0.8) + IP-Adapter (weight 0.6). Measure FID-vs-prompt-adherence
    trade-off as the stack weights vary.

Reading of the exercise: `diffusers` is absent and SDXL is ~7 GB of weights this
repo does not ship, so the four-part stack cannot be assembled and FID cannot be
computed -- FID needs an Inception network, which is also absent. What *can* be
run is the structure the exercise is really about (`DESIGN D11`): two adapters
pulling one frozen layer toward two different targets, with a weight each, and
the two competing objectives measured across the weight grid. The lesson's own
`lora_forward`, `train_lora`'s gradient and `controlnet_toy`'s gated side signal
supply both halves.

**ANSWER: there is a real trade-off, and its curve is bowed.** Sweeping the two
weights across GRID settings, style adherence runs **-1.315 -> 0.000** while
control adherence runs **0.000 -> -1.315** -- opposite directions, which is the
trade-off. The front sits **0.465** off the straight line through its endpoints.

**FINDING: the bow needs no nonlinearity, only a squared error.** The stack is
exactly linear in its weights (Exercise 2), which is what made me expect a
straight front -- wrongly. Adherence is a *squared* error, so each objective is
**quadratic** in the weights even though the model is not. A bowed
FID-versus-adherence curve is therefore evidence about the *metric*, not about
any interaction inside the network.

**FINDING: and the curve is exactly a conic.** Fitting an exact quadratic in the
sweep parameter to each coordinate leaves a worst residual of **4.4e-16**, so the
shape is determined rather than measured: two quadratics traced against each
other are a parabola.

**FINDING: orthogonal targets bow it *more*, not less.** Replacing the control
target with one orthogonal to the style target raises the bow from **0.465** to
**0.760**. Overlap is not the source of the tension -- the tension is one layer
being asked to satisfy two targets at once, and targets that ask for unrelated
things are harder to satisfy together, not easier. That was the second thing this
measurement corrected.

**CONTROL: the ingredients are absent, and that is checked rather than assumed.**
`diffusers`, `torch` and `transformers` all return `None` from `find_spec`, so
neither the pipeline nor the FID network can be constructed here.

Structure: `delta_from` builds a target; `blend` is the two-weight stack;
`objectives` scores one setting; `front` sweeps the grid.
"""

from __future__ import annotations

import importlib.util
import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "08-controlnet-lora-conditioning"
DIM, GRID, PROBES, NEEDED = 6, 5, 200, ("diffusers", "torch", "transformers")


def delta_from(rng, tilt):
    """A rank-1 target delta, `tilt` controlling how far it leans from the first axis."""
    left = [math.cos(tilt)] + [0.0] * (DIM - 2) + [math.sin(tilt)]
    right = [rng.gauss(0, 1) for _ in range(DIM)]
    norm = math.sqrt(sum(v * v for v in right))
    return [[left[i] * right[j] / norm for j in range(DIM)] for i in range(DIM)]


def apply_delta(delta, x):
    """delta @ x."""
    return [sum(delta[i][j] * x[j] for j in range(DIM)) for i in range(DIM)]


def blend(frozen, style, control, x, style_w, control_w):
    """The stacked layer: frozen output plus each adapter scaled by its own weight."""
    base = [sum(frozen[i][j] * x[j] for j in range(DIM)) for i in range(DIM)]
    a, b = apply_delta(style, x), apply_delta(control, x)
    return [base[i] + style_w * a[i] + control_w * b[i] for i in range(DIM)]


def objectives(frozen, style, control, probes, style_w, control_w):
    """(style adherence, control adherence) -- negative mean squared error to each target."""
    scores = [0.0, 0.0]
    for x in probes:
        got = blend(frozen, style, control, x, style_w, control_w)
        base = [sum(frozen[i][j] * x[j] for j in range(DIM)) for i in range(DIM)]
        for idx, delta in enumerate((style, control)):
            want = [b + d for b, d in zip(base, apply_delta(delta, x))]
            scores[idx] -= sum((p - q) ** 2 for p, q in zip(got, want))
    return scores[0] / len(probes), scores[1] / len(probes)


def front(frozen, style, control, probes):
    """The achievable (style, control) pairs across the weight grid."""
    return [objectives(frozen, style, control, probes, i / (GRID - 1), 1 - i / (GRID - 1))
            for i in range(GRID)]


def bow(points):
    """Worst distance of a point from the straight line through the two endpoints."""
    (x0, y0), (x1, y1) = points[0], points[-1]
    span = math.hypot(x1 - x0, y1 - y0) or 1.0
    return max(abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / span for x, y in points)


def summarise(points):
    """(bow from the chord, worst residual from an exact quadratic in the sweep parameter)."""
    steps = [i / (len(points) - 1) for i in range(len(points))]
    worst = 0.0
    for axis in (0, 1):
        values = [p[axis] for p in points]
        picks = (0, len(points) // 2, len(points) - 1)
        coefficients = quadratic_through([(steps[i], values[i]) for i in picks])
        worst = max(worst, max(abs(coefficients[0] * t * t + coefficients[1] * t
                                   + coefficients[2] - y) for t, y in zip(steps, values)))
    return bow(points), worst


def quadratic_through(points):
    """The (a, b, c) of the parabola through three (t, y) points."""
    (t0, y0), (t1, y1), (t2, y2) = points
    den = (t0 - t1) * (t0 - t2) * (t1 - t2)
    a = (t2 * (y1 - y0) + t1 * (y0 - y2) + t0 * (y2 - y1)) / den
    b = (t2 * t2 * (y0 - y1) + t1 * t1 * (y2 - y0) + t0 * t0 * (y1 - y2)) / den
    return a, b, y0 - a * t0 * t0 - b * t0


def solve():
    import random
    parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(21)
    frozen = [[rng.gauss(0, 0.5) for _ in range(DIM)] for _ in range(DIM)]
    style, control = delta_from(rng, 0.0), delta_from(rng, 0.6)
    orthogonal = delta_from(rng, math.pi / 2)
    probes = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(PROBES)]
    points = front(frozen, style, control, probes)
    curved, residual = summarise(points)
    return {
        "front": points, "bow": curved, "residual": residual,
        "orthogonal_bow": summarise(front(frozen, style, orthogonal, probes))[0],
        "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None],
    }


def verify(result):
    first, last = result["front"][0], result["front"][-1]
    return [
        practice.Check(
            "ANSWER: there is a real trade-off, and its curve is bowed",
            first[0] < last[0] and first[1] > last[1] and result["bow"] > 0.1,
            f"across {GRID} weight settings style adherence runs {first[0]:.3f} to {last[0]:.3f} "
            f"while control adherence runs {first[1]:.3f} to {last[1]:.3f} -- opposite "
            f"directions, which is the trade-off -- and the front sits {result['bow']:.3f} off "
            "the straight line through its endpoints",
        ),
        practice.Check(
            "FINDING: the bow needs no nonlinearity, only a squared error",
            result["bow"] > 0.1,
            "the stack is exactly linear in its weights, which is what makes a straight front the "
            "natural guess -- and it is wrong. Adherence is a squared error, so each objective is "
            "quadratic in the weights even though the model is not. A bowed FID-versus-adherence "
            "curve is evidence about the metric, not about any interaction inside the network",
        ),
        practice.Check(
            "FINDING: and the curve is exactly a conic",
            result["residual"] < 1e-12,
            f"fitting an exact quadratic in the sweep parameter to each coordinate leaves a worst "
            f"residual of {result['residual']:.1e}, so the shape is determined rather than "
            "measured: two quadratics traced against one another are a parabola, and that is the "
            "whole content of the front",
        ),
        practice.Check(
            "FINDING: orthogonal targets bow it more, not less",
            result["orthogonal_bow"] > result["bow"],
            f"replacing the control target with one orthogonal to the style target raises the bow "
            f"from {result['bow']:.3f} to {result['orthogonal_bow']:.3f}. Overlap is not the "
            "source of the tension: the tension is one layer satisfying two targets at once, and "
            "targets asking for unrelated things are harder to satisfy together, not easier",
        ),
        practice.Check(
            "CONTROL: the ingredients are absent, and that is checked rather than assumed",
            result["absent"] == list(NEEDED),
            f"{result['absent']} all return None from find_spec, so neither the four-part SDXL "
            "pipeline nor the Inception network FID needs can be built here. What runs instead is "
            "the structure the exercise is about: two weighted adapters on one frozen layer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
