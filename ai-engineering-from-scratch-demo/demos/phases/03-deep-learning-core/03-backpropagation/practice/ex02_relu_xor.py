"""Exercise 2 — relu on XOR, and the training it is supposed to speed up.

    Add a `relu` method to Value (output max(0, x), derivative is 1 if x > 0,
    else 0). Replace sigmoid with relu in the hidden layers and train on XOR
    again. Compare convergence speed. You should see faster training -- this
    previews Lesson 04.

Reading of the exercise: "you should see faster training" is a prediction, so it
gets measured. `relu` is grafted onto the lesson's own `Value` and `Neuron`
(hidden layer only, output stays sigmoid) and one loop runs both activations over
eight seeds, "converged" meaning total XOR loss under 0.04. Check 3 is what the
failing seeds share; check 4 prices the comparison itself.
"""

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


def fresh(ref, seed, act):
    random.seed(seed)
    net = ref.Network([2, HIDDEN, 1])
    for neuron in net.layers[0].neurons: neuron.act = act
    return net


def dead(net) -> int:
    """Hidden units with pre-activation <= 0 on all four rows: relu passes them nothing."""
    return sum(max(sum(w.data * x for w, x in zip(n.weights, r)) + n.bias.data
                   for r, _t in XOR) <= 0 for n in net.layers[0].neurons)


def run(ref, seed, act, lr) -> dict:
    net, hist = fresh(ref, seed, act), []
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
    return {"epochs": next((i + 1 for i, v in enumerate(hist) if v < TARGET), None),
            "loss": hist[-1], "born": born, "died": dead(net), "zero": zero}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    install(ref)
    sig = [run(ref, s, "sigmoid", LR) for s in SEEDS]
    rel = [run(ref, s, "relu", LR) for s in SEEDS]
    half = [run(ref, s, "sigmoid", LR / 2) for s in SEEDS]
    pairs = [(a["epochs"], b["epochs"]) for a, b in zip(sig, rel) if a["epochs"] and b["epochs"]]
    born, died = [r["born"] for r in rel], [r["died"] for r in rel]
    return {
        "sig_conv": sum(r["epochs"] is not None for r in sig), "seeds": len(SEEDS),
        "rel_conv": sum(r["epochs"] is not None for r in rel), "pairs": pairs, "born": born,
        "half_conv": sum(r["epochs"] is not None for r in half), "died": died,
        "speedup": [s / r for s, r in pairs], "grew": sum(d > b for b, d in zip(born, died)),
        "span": (min(r["epochs"] for r in sig), max(r["epochs"] for r in sig)),
        "stuck": sorted({round(r["loss"], 4) for r in rel if r["epochs"] is None}),
        "zero": (max(r["zero"] for r in rel), min(r["zero"] for r in sig)),
    }


def verify(result):
    seeds, span, pairs = result["seeds"], result["span"], result["pairs"]
    fast = ", ".join(f"{r} vs {s} ({s / r:.1f}x)" for s, r in pairs)
    return [
        practice.Check("ANSWER: where relu converges at all it is several times faster",
                       min(result["speedup"]) > 3.0 and result["rel_conv"] > 0,
                       f"on the {len(pairs)} of {seeds} seeds where both reach loss < "
                       f"{TARGET}, relu epochs vs sigmoid epochs: {fast}"),
        practice.Check("FINDING: that is 2 seeds in 8 — the usual relu outcome is no "
                       "training at all",
                       result["rel_conv"] < result["sig_conv"] == seeds,
                       f"sigmoid converges on {result['sig_conv']}/{seeds} ({span[0]}-"
                       f"{span[1]} epochs), relu on {result['rel_conv']}; the rest park at "
                       f"constant-output losses {result['stuck']} for good"),
        practice.Check("MECHANISM: relu'(x <= 0) = 0 kills units, and `bias = Value(0.0)` "
                       "kills the (0,0) row at birth",
                       min(result["died"]) > 0 and result["grew"] > seeds // 2
                       and result["zero"][0] == 0.0 < result["zero"][1],
                       f"units dead at init vs after {EPOCHS} epochs: {result['born']} -> "
                       f"{result['died']} of {HIDDEN}, {result['grew']}/{seeds} seeds losing "
                       f"units mid-training and none recovering. Input (0, 0) also sits on "
                       f"the kink at init, for every seed: |grad| into layer 0 from that row "
                       f"peaks at {result['zero'][0]:.4f} for relu, {result['zero'][1]:.4f} "
                       f"at worst for sigmoid"),
        practice.Check("CONTROL: halve the learning rate and sigmoid stops converging too",
                       result["half_conv"] < result["sig_conv"],
                       f"the same sigmoid net at lr = {LR / 2} converges on "
                       f"{result['half_conv']}/{seeds}, down from {result['sig_conv']}: "
                       f"relu's derivative of 1 against sigmoid's 0.25 ceiling rescales the "
                       f"step as much as it reshapes it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
