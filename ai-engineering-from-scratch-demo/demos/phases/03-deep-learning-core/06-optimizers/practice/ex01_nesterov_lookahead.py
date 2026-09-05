"""Exercise 1 — Nesterov momentum, against the lesson's own SGDMomentum.

    Implement Nesterov momentum, where you compute the gradient at the
    "lookahead" position (w - lr * beta * v) instead of the current position.
    Compare convergence to standard momentum on the circle dataset.

Reading of the exercise: Nesterov changes *where the gradient is sampled*, not the
update rule, so this reuses `SGDMomentum` untouched. "Compare convergence" is answered
twice — on the circle dataset at lr = 0.05 (check 1), and in closed form on the
quadratic the lesson's own `__main__` minimises, where both methods are exact two-term
recurrences with computable rates (2-3). Check 4 keeps check 1 honest.
"""

from __future__ import annotations

import cmath
import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "06-optimizers"
LR, BETA, EPOCHS = 0.05, 0.9, 300


def train(ref, data, opt, lookahead, epochs=EPOCHS):
    """The lesson's own loop, with the gradient site as the single free choice."""
    net = ref.OptimizerTestNetwork(opt, hidden_size=8)
    history = []
    for _epoch in range(epochs):
        total, correct = 0.0, 0
        for point, label in data:
            params = net.get_params()
            if lookahead and opt.velocities is not None:
                net.set_params([p - opt.lr * opt.beta * v
                                for p, v in zip(params, opt.velocities)])
            net.forward(point)
            grads = net.compute_grads(label)
            net.set_params(params)
            pred = net.forward(point)
            opt.step(params, grads)
            net.set_params(params)
            p = max(1e-15, min(1 - 1e-15, pred))
            total -= label * math.log(p) + (1 - label) * math.log(1 - p)
            correct += (pred >= 0.5) == (label >= 0.5)
        history.append((total / len(data), 100.0 * correct / len(data)))
    return history


def reached(history, threshold):
    return next((i for i, (_l, a) in enumerate(history) if a >= threshold), None)


def quadratic(ref, kind, steps=40):
    """f(x) = (x-3)^2 from x = 10 — the lesson's Step 1 demo, error per step."""
    x, errors = [10.0], [7.0]
    opt = ref.SGD(lr=0.1) if kind == "sgd" else ref.SGDMomentum(lr=0.1, beta=BETA)
    for _step in range(steps):
        velocity = getattr(opt, "velocities", None)
        site = x[0] - 0.1 * BETA * velocity[0] if kind == "nesterov" and velocity else x[0]
        opt.step(x, [2.0 * (site - 3.0)])
        errors.append(x[0] - 3.0)
    return errors


def analyse(errors, a, b):
    """Worst violation of the hand-derived e[t+1] = a*e[t] - b*e[t-1], and its rate."""
    root = cmath.sqrt(a * a - 4 * b)
    return (max(abs(errors[t + 1] - a * errors[t] + b * errors[t - 1])
                for t in range(1, len(errors) - 1)),
            max(abs((a + root) / 2), abs((a - root) / 2)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data()
    runs = {name: train(ref, data, ref.SGDMomentum(lr=lr, beta=BETA), look)
            for name, lr, look in [("momentum", LR, False), ("nesterov", LR, True),
                                   ("tuned", 0.01, False)]}
    return {"runs": runs, "rec": {"momentum": (1.7, BETA), "nesterov": (1.52, 0.72)},
            "errors": {k: quadratic(ref, k) for k in ("sgd", "momentum", "nesterov")}}


def verify(result):
    runs, errors, rec = result["runs"], result["errors"], result["rec"]
    end = {k: f"{v[-1][0]:.4f} / {v[-1][1]:.1f}%" for k, v in runs.items()}
    loss, best = {k: v[-1][0] for k, v in runs.items()}, max(a for _l, a in runs["momentum"])
    fit = {k: analyse(errors[k], *rec[k]) for k in rec}
    rate, worst = {k: v[1] for k, v in fit.items()}, max(v[0] for v in fit.values())
    exact = max(abs(e - 7.0 * 0.8 ** t) for t, e in enumerate(errors["sgd"]))
    size = ", ".join(f"{abs(errors[k][40]):.2e}" for k in ("sgd", "nesterov", "momentum"))
    return [
        practice.Check("ANSWER: the lookahead converges faster at the lesson's own lr = 0.05",
                       reached(runs["nesterov"], 90) == 3 and reached(runs["momentum"], 90) == 6
                       and loss["nesterov"] < 0.5 * loss["momentum"],
                       f"90% at epoch 3 against 6; after {EPOCHS} epochs {end['nesterov']} "
                       f"against {end['momentum']}. Momentum never reaches 95% (best "
                       f"{best:.1f}%); Nesterov does at epoch {reached(runs['nesterov'], 95)}"),
        practice.Check("MECHANISM: 1-2*lr factors out of both coefficients", worst < 1e-13,
                       f"on f(x) = (x-3)^2 from x = 10 at lr = 0.1: momentum obeys e[t+1] = 1.70"
                       f"*e[t] - 0.90*e[t-1], Nesterov 0.8*(1.90*e[t] - 0.90*e[t-1]); worst "
                       f"residual over 40 steps {worst:.2e}"),
        practice.Check("FINDING: on that quadratic plain SGD beats both",
                       abs(rate["momentum"] - 0.9487) < 5e-4 and exact < 1e-13
                       and abs(rate["nesterov"] - 0.8485) < 5e-4,
                       f"per-step rates: SGD 0.8000 (exactly 7*0.8^t, worst deviation "
                       f"{exact:.1e}), Nesterov {rate['nesterov']:.4f}, momentum "
                       f"{rate['momentum']:.4f} = sqrt(beta); |e40| = {size}, momentum 873x "
                       f"behind. A 1-D bowl has no narrow valley to smooth"),
        practice.Check("CONTROL: the circle-data win is a step-size effect, not a free lunch",
                       loss["tuned"] < 0.5 * loss["nesterov"] and runs["tuned"][-1][1] >= 99.0,
                       f"standard momentum at lr = 0.01 ends at {end['tuned']}, better than "
                       f"Nesterov at lr = 0.05 ({end['nesterov']}) — 0.05 is past momentum's "
                       f"stability edge, and the lookahead damps rather than accelerates"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
