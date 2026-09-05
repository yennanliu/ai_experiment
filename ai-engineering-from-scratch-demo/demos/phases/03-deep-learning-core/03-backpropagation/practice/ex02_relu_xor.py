"""Exercise 2 — relu on XOR, and the training it is supposed to speed up.

    Add a `relu` method to Value (output max(0, x), derivative is 1 if x > 0,
    else 0). Replace sigmoid with relu in the hidden layers and train on XOR
    again. Compare convergence speed. You should see faster training -- this
    previews Lesson 04.

Reading of the exercise: "you should see faster training" is a prediction, so it
gets measured. `relu` is grafted onto the lesson's own `Value` and `Neuron`
(hidden layer only, output stays sigmoid) and one loop runs both activations over
eight seeds, "converged" meaning total XOR loss under 0.04. Check 3 is what the
failing seeds share, check 4 prices the comparison itself."""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "03-backpropagation"
XOR = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
SEEDS, HIDDEN, EPOCHS, LR, TARGET = range(8), 4, 600, 1.0, 0.04


def install(ref) -> None:
    """The exercise's edit: `Value.relu`, plus a Neuron that picks its activation."""
    value = ref.Value
    def relu(self):
        out = value(max(0.0, self.data), (self,), "relu")
        def _backward(): self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
    def call(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.relu() if getattr(self, "act", "") == "relu" else act.sigmoid()
    value.relu, ref.Neuron.__call__ = relu, call


def dead(net) -> int:
    """Hidden units with pre-activation <= 0 on all four rows — relu sends them nothing."""
    return sum(max(sum(w.data * x for w, x in zip(n.weights, r)) + n.bias.data
                   for r, _t in XOR) <= 0 for n in net.layers[0].neurons)


def run(ref, seed, act, lr) -> tuple:
    random.seed(seed)
    net, hist = ref.Network([2, HIDDEN, 1]), []
    for neuron in net.layers[0].neurons: neuron.act = act
    born = dead(net)
    ref.mse_loss(net([ref.Value(0.0), ref.Value(0.0)]), 0.0).backward()
    zero = sum(abs(p.grad) for p in net.layers[0].parameters())
    for _epoch in range(EPOCHS):
        total = ref.Value(0.0)
        for inputs, target in XOR:
            total = total + ref.mse_loss(net([ref.Value(i) for i in inputs]), target)
        net.zero_grad()
        total.backward()
        for param in net.parameters(): param.data -= lr * param.grad
        hist.append(total.data)
    return (next((i + 1 for i, v in enumerate(hist) if v < TARGET), None),
            hist[-1], born, dead(net), zero)


def sweep(ref, act, lr) -> dict:
    """One configuration over every seed, transposed into columns."""
    ep, loss, born, died, zero = zip(*(run(ref, s, act, lr) for s in SEEDS))
    good = [e for e in ep if e]
    return {"ep": ep, "conv": len(good), "born": list(born), "died": list(died),
            "span": (min(good, default=0), max(good, default=0)),
            "high": max(zero), "low": min(zero),
            "grew": sum(d > b for b, d in zip(born, died)),
            "stuck": sorted({round(v, 4) for e, v in zip(ep, loss) if e is None})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    install(ref)
    sig, rel = sweep(ref, "sigmoid", LR), sweep(ref, "relu", LR)
    pairs = [(a, b) for a, b in zip(sig["ep"], rel["ep"]) if a and b]
    return {"seeds": len(SEEDS), "pairs": pairs, "sig": sig, "rel": rel,
            "half": sweep(ref, "sigmoid", LR / 2)["conv"]}


def verify(result):
    seeds, pairs, sig, rel = result["seeds"], result["pairs"], result["sig"], result["rel"]
    fast = ", ".join(f"{r} vs {s} ({s / r:.1f}x)" for s, r in pairs)
    return [
        practice.Check("ANSWER: where relu converges at all it is several times faster",
                       min(s / r for s, r in pairs) > 3.0 and rel["conv"] > 0,
                       f"on the {len(pairs)} of {seeds} seeds where both reach loss < "
                       f"{TARGET}, relu epochs vs sigmoid epochs: {fast}"),
        practice.Check("FINDING: that is 2 seeds in 8 — the usual relu outcome is no "
                       "training at all",
                       rel["conv"] < sig["conv"] == seeds,
                       f"sigmoid converges on {sig['conv']}/{seeds} ({sig['span'][0]}-"
                       f"{sig['span'][1]} epochs), relu on {rel['conv']}; the rest park at "
                       f"constant-output losses {rel['stuck']} for good"),
        practice.Check("MECHANISM: relu'(x <= 0) = 0 kills units, and `bias = Value(0.0)` "
                       "kills the (0,0) row at birth",
                       min(rel["died"]) > 0 and rel["grew"] > seeds // 2
                       and rel["high"] == 0.0 < sig["low"],
                       f"units dead at init vs after {EPOCHS} epochs: {rel['born']} -> "
                       f"{rel['died']} of {HIDDEN}, {rel['grew']}/{seeds} seeds losing units "
                       f"mid-training and none recovering. Input (0, 0) also sits on the "
                       f"kink at init on every seed: |grad| into layer 0 from that row peaks "
                       f"at {rel['high']:.4f} for relu against {sig['low']:.4f} at worst for "
                       f"sigmoid"),
        practice.Check("CONTROL: halve the learning rate and sigmoid stops converging too",
                       result["half"] < sig["conv"],
                       f"the same sigmoid net at lr = {LR / 2} converges on "
                       f"{result['half']}/{seeds}, down from {sig['conv']}: relu's "
                       f"derivative of 1 against sigmoid's 0.25 ceiling rescales the step as "
                       f"much as it reshapes it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
