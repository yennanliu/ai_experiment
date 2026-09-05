"""Exercise 4 — weight decay in the framework's Adam.

    Implement weight decay (L2 regularization) in the Adam optimizer. Add a
    `weight_decay` parameter that shrinks weights toward zero each step. Compare
    training with decay=0 vs decay=0.01.

Reading of the exercise: "shrinks weights toward zero each step" is the decoupled rule, so
that is the one measured against decay=0 in check 1 and pinned to its closed form in check 2.
The other reading — L2 folded into the gradient, which the title names — is check 3, and it
shrinks more than twice as hard at the same 0.01. Check 4 is the parameter list the rule is
applied over, which does not distinguish weights from biases.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "10-mini-framework"
EPOCHS, BATCH, LR, SPLIT, DECAY = 40, 16, 0.01, 400, 0.01


def build(ref, seed=42):
    """The lesson's own 2-16-16-8-1 stack from `train_framework`."""
    random.seed(seed)
    return ref.Sequential(ref.Linear(2, 16), ref.ReLU(), ref.Linear(16, 16), ref.ReLU(),
                          ref.Linear(16, 8), ref.ReLU(), ref.Linear(8, 1), ref.Sigmoid())


def slots(entry) -> tuple:
    """(weights, grads, index) for one of `parameters()`'s tuples, matrix or bias alike."""
    container, i, j, grads = entry
    return ((container[i], grads[i], j) if j is not None else (container, grads, i))


def apply_decay(optimizer, decay, mode, biases=True) -> None:
    """decoupled: w -= lr*wd*w, straight off the weight. coupled: wd*w into the gradient."""
    for entry in optimizer.params:
        if not decay or (entry[2] is None and not biases):
            continue
        weights, grads, index = slots(entry)
        if mode == "decoupled":
            weights[index] -= LR * decay * weights[index]
        else:
            grads[index] += decay * weights[index]


def mean_grads(optimizer, size) -> None:
    """Exercise 5's convention: the accumulated gradients are a batch mean before the step."""
    for entry in optimizer.params:
        _weights, grads, index = slots(entry)
        grads[index] /= size


def norm(model) -> float:
    """||theta|| over every scalar `parameters()` enumerates."""
    return math.sqrt(sum(slots(e)[0][slots(e)[2]] ** 2 for e in model.parameters()))


def score(ref, model, held) -> tuple:
    model.eval()
    crit = ref.BCELoss()
    loss = sum(crit(model.forward(x), t) for x, t in held) / len(held)
    right = sum((model.forward(x)[0] >= 0.5) == (t[0] >= 0.5) for x, t in held)
    model.train()
    return loss, 100.0 * right / len(held)


def train(ref, decay, mode="decoupled", biases=True) -> dict:
    """The exercise's mini-batch loop, with the decay applied just before `step`."""
    model, crit = build(ref), ref.BCELoss()
    optimizer = ref.Adam(model.parameters(), lr=LR)
    data = ref.make_circle_data(500)
    random.seed(1)
    loader = ref.DataLoader(data[:SPLIT], batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        for inputs, targets in loader:
            optimizer.zero_grad()
            for x, target in zip(inputs, targets):
                crit(model.forward(x), target)
                model.backward(crit.backward())
            mean_grads(optimizer, len(inputs))
            apply_decay(optimizer, decay, mode, biases)
            optimizer.step()
    return {"end": score(ref, model, data[SPLIT:]), "norm": norm(model),
            "steps": EPOCHS * len(loader)}


def frozen(ref, decay, steps) -> float:
    """The decoupled rule with no gradient at all — what its closed form predicts."""
    model = build(ref)
    optimizer = ref.Adam(model.parameters(), lr=LR)
    start = norm(model)
    for _ in range(steps):
        apply_decay(optimizer, decay, "decoupled")
    return norm(model) / (start * (1 - LR * decay) ** steps)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    n_bias = sum(1 for _c, _i, j, _g in build(ref).parameters() if j is None)
    return {"off": train(ref, 0.0), "on": train(ref, DECAY),
            "coupled": train(ref, DECAY, mode="coupled"),
            "no_bias": train(ref, DECAY, biases=False), "hard": train(ref, 10 * DECAY),
            "law": frozen(ref, DECAY, 1000), "n_bias": n_bias,
            "n_param": len(build(ref).parameters())}


def verify(result):
    off, on, coupled = result["off"], result["on"], result["coupled"]
    plain, hard = result["no_bias"], result["hard"]
    return [
        practice.Check("ANSWER: at decay 0.01 the weights shrink 6% and nothing else moves",
                       0.02 < 1 - on["norm"] / off["norm"] < 0.15
                       and abs(on["end"][0] - off["end"][0]) < 0.01,
                       f"{EPOCHS} epochs, {on['steps']} Adam steps: decay 0 ends at "
                       f"{off['end'][0]:.4f}/{off['end'][1]:.1f}% with ||theta|| "
                       f"{off['norm']:.3f}; decay {DECAY} ends at {on['end'][0]:.4f}/"
                       f"{on['end'][1]:.1f}% with {on['norm']:.3f} — "
                       f"{100 * (1 - on['norm'] / off['norm']):.1f}% smaller, one test point of "
                       f"accuracy apart on 100 held-out points"),
        practice.Check("MECHANISM: the decoupled rule is exactly a geometric shrink",
                       abs(result["law"] - 1.0) < 1e-9,
                       f"with every gradient zeroed, {on['steps']} steps of `w -= lr*wd*w` land "
                       f"on ||theta|| * (1 - lr*wd)^n to {abs(result['law'] - 1):.1e} relative. "
                       f"That predicts a {100 * (1 - (1 - LR * DECAY) ** on['steps']):.1f}% shrink "
                       f"over this run against the {100 * (1 - on['norm'] / off['norm']):.1f}% "
                       f"measured — the gradient pushes the rest back"),
        practice.Check("FINDING: the other reading of 'L2' shrinks more than twice as hard at "
                       "the same 0.01",
                       coupled["norm"] < 0.6 * on["norm"],
                       f"folding wd*w into the gradient before the step ends at ||theta|| "
                       f"{coupled['norm']:.3f} against the decoupled rule's {on['norm']:.3f} — "
                       f"{on['norm'] / coupled['norm']:.1f}x — at {coupled['end'][0]:.4f}/"
                       f"{coupled['end'][1]:.1f}%. Adam divides the coupled term by sqrt(v_hat), "
                       f"so it survives as a pull of order lr however small w is, while the "
                       f"decoupled term vanishes with w. The exercise's own wording picks the "
                       f"weaker one"),
        practice.Check("FINDING: `parameters()` does not distinguish weights from biases",
                       abs(plain["norm"] - on["norm"]) < 0.5 and result["n_bias"] == 41,
                       f"the list holds {result['n_param']} entries of which {result['n_bias']} "
                       f"are biases, told apart only by `j is None`. A loop over it decays them "
                       f"too — excluding them gives ||theta|| {plain['norm']:.3f} against "
                       f"{on['norm']:.3f}, because the biases start at 0.0 and stay small"),
        practice.Check("CONTROL: ten times the decay does move the norm, and not the accuracy",
                       hard["norm"] < 0.85 * off["norm"] and abs(hard["end"][1] - off["end"][1]) < 3,
                       f"decay {10 * DECAY} ends at ||theta|| {hard['norm']:.3f}, "
                       f"{100 * (1 - hard['norm'] / off['norm']):.0f}% below the undecayed run, "
                       f"at {hard['end'][0]:.4f}/{hard['end'][1]:.1f}%. On 400 training points a "
                       f"465-parameter net has nothing to over-fit, so the whole sweep buys "
                       f"smaller weights and no generalisation"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
