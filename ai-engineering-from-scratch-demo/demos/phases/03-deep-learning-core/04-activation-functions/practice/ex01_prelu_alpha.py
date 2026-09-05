"""Exercise 1 — PReLU with a learnable alpha, against fixed Leaky ReLU.

    Implement Parametric ReLU (PReLU) where the negative slope alpha is a
    learnable parameter. Train it on the circle dataset and compare to fixed
    Leaky ReLU.

Reading of the exercise: "learnable" is the claim to test, so check 1 audits the
alpha gradient against a central difference before any training number is read
off. "Compare to fixed Leaky ReLU" is then two comparisons — against alpha = 0.01
(check 3), and against the value alpha *reaches*, used fixed from epoch 0 (check 4).
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "04-activation-functions"
EPOCHS, TARGET, BASE_RATE = 200, 0.05, 71.5   # 143 of 200 circle points lie outside


def make(ref, alpha):
    """The lesson's own network, its slope read from a cell we can train."""
    box = [alpha]
    net = ref.ActivationNetwork(lambda x: ref.leaky_relu(x, box[0]),
                                lambda x: ref.leaky_relu_derivative(x, box[0]), 8, 0.1)
    net.alpha = box
    return net


def alpha_grad(net, target):
    """dL/dalpha = sum_i dL/dz_i * min(z_i, 0). Must be read before w2 updates."""
    d_out = (net.out - target) * net.out * (1 - net.out)
    return sum(d_out * net.w2[i] * min(net.z1[i], 0.0) for i in range(net.hidden_size))


def train(ref, alpha, learn, data):
    net, losses = make(ref, alpha), []
    for _epoch in range(EPOCHS):
        total, correct = 0.0, 0
        for point, label in data:
            pred = net.forward(point)
            grad = alpha_grad(net, label)
            net.backward(label)
            net.alpha[0] -= net.lr * grad * learn
            total += (pred - label) ** 2
            correct += (pred >= 0.5) == (label >= 0.5)
        losses.append(total / len(data))
    return {"loss": losses[-1], "acc": correct / len(data) * 100, "alpha": net.alpha[0],
            "epochs": next((i + 1 for i, v in enumerate(losses) if v < TARGET), None)}


def finite_diff(ref, net, point, label, step=1e-6):
    """Central difference of the sample loss in alpha, at this net's own weights."""
    def loss_at(alpha):
        probe = make(ref, alpha)
        probe.w1 = [row[:] for row in net.w1]
        probe.b1, probe.w2, probe.b2 = list(net.b1), list(net.w2), net.b2
        return 0.5 * (probe.forward(point) - label) ** 2
    return (loss_at(net.alpha[0] + step) - loss_at(net.alpha[0] - step)) / (2 * step)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data()
    probe = make(ref, 0.3)
    probe.forward(data[0][0])
    learned = train(ref, 0.01, 1, data)
    return {"analytic": alpha_grad(probe, data[0][1]),
            "numeric": finite_diff(ref, probe, *data[0]), "learned": learned,
            "fixed": train(ref, 0.01, 0, data), "identity": train(ref, 1.0, 0, data),
            "frozen": train(ref, learned["alpha"], 0, data)}


def verify(result):
    learned, fixed, frozen, ident = (result[k] for k in
                                     ("learned", "fixed", "frozen", "identity"))
    rel = abs(result["analytic"] - result["numeric"]) / abs(result["numeric"])
    return [
        practice.Check("MECHANISM: alpha really is trained, not drifting", rel < 1e-6,
                       f"sum_i dL/dz_i * min(z_i, 0) = {result['analytic']:.8e} against a "
                       f"central difference of the sample loss {result['numeric']:.8e} — "
                       f"relative error {rel:.2e}"),
        practice.Check("ANSWER: alpha learns, and leaves the range Leaky ReLU lives in",
                       learned["alpha"] < -2.0,
                       f"alpha runs 0.0100 -> {learned['alpha']:+.4f} over {EPOCHS} epochs, "
                       f"crossing 0 inside the first. A negative slope makes the unit a V — "
                       f"x for x > 0, {-learned['alpha']:.2f}|x| for x <= 0 — non-monotone and "
                       f"positive on both sides, and the circle label is even in each "
                       f"coordinate, so one V does what two ReLUs would have to"),
        practice.Check("…and against fixed Leaky ReLU it buys almost nothing",
                       abs(learned["loss"] - fixed["loss"]) < 0.002
                       and learned["acc"] <= fixed["acc"],
                       f"loss {learned['loss']:.4f} against {fixed['loss']:.4f} at fixed "
                       f"alpha = 0.01, accuracy {learned['acc']:.1f}% against "
                       f"{fixed['acc']:.1f}%, {learned['epochs']} epochs to loss < {TARGET} "
                       f"against {fixed['epochs']} — a 2% edge and half a point of accuracy "
                       f"the wrong way, for a parameter that moved by 2.6"),
        practice.Check("FINDING: the alpha it finds is worth far more than learning it",
                       frozen["epochs"] * 2 < fixed["epochs"]
                       and frozen["loss"] < fixed["loss"],
                       f"frozen at the learned {learned['alpha']:+.4f} from epoch 0, the same "
                       f"net reaches loss < {TARGET} in {frozen['epochs']} epochs not "
                       f"{fixed['epochs']} ({fixed['epochs'] / frozen['epochs']:.1f}x) and "
                       f"ends at {frozen['loss']:.4f} against {fixed['loss']:.4f}. MECHANISM: "
                       f"PReLU spends the run travelling there, training under a slope that "
                       f"is only good at the end"),
        practice.Check("CONTROL: alpha = 1 is the ridge the descent runs away from",
                       abs(ident["acc"] - BASE_RATE) < 1e-9 and ident["epochs"] is None,
                       f"at alpha = 1.0 leaky_relu is the identity, the hidden layer is linear "
                       f"and the net is one logistic regression: loss stuck at "
                       f"{ident['loss']:.4f}, accuracy {ident['acc']:.1f}% — exactly the "
                       f"143/200 base rate of calling every point 'outside'. alpha descends "
                       f"away from that ridge, so the learned slope goes negative"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
