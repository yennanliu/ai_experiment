"""Exercise 5 — one optimizer step per batch instead of one per sample.

    Replace the per-sample training loop with proper mini-batch gradient
    accumulation: accumulate gradients across all samples in a batch, then divide
    by batch size and take one optimizer step. Measure whether this changes
    convergence speed.

Reading of the exercise: "convergence speed" has two units here and they disagree, so check 1
reports both — epochs to a loss threshold, and where each arm ends. The framework already
accumulates (check 3), so the edit is where `zero_grad` and `step` sit. Checks 4-5 test the
"divide by batch size" clause on its own, by running the same loop with and without it under
each of the lesson's two optimizers.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "10-mini-framework"
EPOCHS, BATCH, LR, SPLIT, TARGET = 40, 16, 0.01, 400, 0.10


def build(ref, seed=42):
    """The lesson's own 2-16-16-8-1 stack from `train_framework`."""
    random.seed(seed)
    return ref.Sequential(ref.Linear(2, 16), ref.ReLU(), ref.Linear(16, 16), ref.ReLU(),
                          ref.Linear(16, 8), ref.ReLU(), ref.Linear(8, 1), ref.Sigmoid())


def scale(optimizer, factor) -> None:
    """The 'divide by batch size' the exercise asks for, applied to the accumulated grads."""
    for container, i, j, grads in optimizer.params:
        if j is not None:
            grads[i][j] *= factor
        else:
            grads[i] *= factor


def score(ref, model, held) -> tuple:
    model.eval()
    crit = ref.BCELoss()
    loss = sum(crit(model.forward(x), t) for x, t in held) / len(held)
    right = sum((model.forward(x)[0] >= 0.5) == (t[0] >= 0.5) for x, t in held)
    model.train()
    return loss, 100.0 * right / len(held)


def epoch(ref, model, crit, optimizer, loader, per_sample, divide) -> int:
    """One pass. `per_sample` is the lesson's loop; the other is the exercise's."""
    steps = 0
    for inputs, targets in loader:
        if not per_sample:
            optimizer.zero_grad()
        for x, target in zip(inputs, targets):
            if per_sample:
                optimizer.zero_grad()
            crit(model.forward(x), target)
            model.backward(crit.backward())
            if per_sample:
                optimizer.step()
                steps += 1
        if not per_sample:
            if divide:
                scale(optimizer, 1.0 / len(inputs))
            optimizer.step()
            steps += 1
    return steps


def train(ref, per_sample, divide=True, optimizer_cls=None) -> dict:
    model, crit = build(ref), ref.BCELoss()
    optimizer = (optimizer_cls or ref.Adam)(model.parameters(), lr=LR)
    data = ref.make_circle_data(500)
    random.seed(1)
    loader = ref.DataLoader(data[:SPLIT], batch_size=BATCH, shuffle=True)
    curve, steps = [], 0
    for _ in range(EPOCHS):
        steps += epoch(ref, model, crit, optimizer, loader, per_sample, divide)
        curve.append(score(ref, model, data[SPLIT:]))
    return {"curve": curve, "steps": steps, "end": curve[-1],
            "first": next((i + 1 for i, (loss, _a) in enumerate(curve) if loss < TARGET), None)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    accumulates = "weight_grads[i][j] +=" in inspect.getsource(ref.Linear.backward)
    return {"sample": train(ref, True), "batch": train(ref, False),
            "undivided": train(ref, False, divide=False),
            "sgd": {d: train(ref, False, divide=d, optimizer_cls=ref.SGD) for d in (True, False)},
            "accumulates": accumulates}


def verify(result):
    one, many = result["sample"], result["batch"]
    plain, sgd = result["undivided"], result["sgd"]
    return [
        practice.Check("ANSWER: it changes convergence, and in opposite directions on the two "
                       "clocks",
                       one["first"] < many["first"] and one["end"][0] > 2 * many["end"][0],
                       f"epochs to test loss < {TARGET}: {one['first']} per sample against "
                       f"{many['first']} per batch — but after {EPOCHS} epochs per-sample ends at "
                       f"{one['end'][0]:.4f}/{one['end'][1]:.1f}% and per-batch at "
                       f"{many['end'][0]:.4f}/{many['end'][1]:.1f}%. The per-sample loop arrives "
                       f"first and then walks away"),
        practice.Check("MECHANISM: it is 16x fewer optimizer steps for the same data",
                       one["steps"] == BATCH * many["steps"],
                       f"{one['steps']:,} steps against {many['steps']:,} over the same "
                       f"{EPOCHS} epochs. `train_framework` iterates a DataLoader of batch "
                       f"{BATCH} and then calls zero_grad/step *inside* the inner loop, so its "
                       f"batching only decides the shuffling order"),
        practice.Check("MECHANISM: the framework was already built for this",
                       result["accumulates"],
                       "`Linear.backward` writes `self.weight_grads[i][j] += ...`, so gradients "
                       "already accumulate across calls. The whole change is moving `zero_grad` "
                       "and `step` out of the inner loop and scaling once — no new state, no "
                       "new method"),
        practice.Check("MECHANISM: 'divide by batch size' is a no-op under Adam",
                       abs(plain["end"][0] - many["end"][0]) < 1e-3,
                       f"the same loop with and without the 1/{BATCH}: "
                       f"{many['end'][0]:.4f}/{many['end'][1]:.1f}% divided against "
                       f"{plain['end'][0]:.4f}/{plain['end'][1]:.1f}% undivided. Adam divides "
                       f"the update by sqrt(v_hat), so a constant factor on every gradient "
                       f"cancels out of it"),
        practice.Check("CONTROL: under the lesson's SGD the same division decides the run",
                       abs(sgd[True]["end"][0] - sgd[False]["end"][0]) > 0.2
                       and sgd[False]["end"][1] - sgd[True]["end"][1] > 5,
                       f"SGD at the same lr = {LR}: {sgd[True]['end'][0]:.4f}/"
                       f"{sgd[True]['end'][1]:.1f}% divided against "
                       f"{sgd[False]['end'][0]:.4f}/{sgd[False]['end'][1]:.1f}% undivided — "
                       f"dividing by {BATCH} is a {BATCH}x learning-rate cut, and nothing "
                       f"downstream undoes it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
