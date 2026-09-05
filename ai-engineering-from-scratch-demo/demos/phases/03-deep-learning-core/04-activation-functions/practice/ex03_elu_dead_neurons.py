"""Exercise 3 — ELU, and what the lesson's "dead neuron rate" actually measures.

    Implement the ELU (Exponential Linear Unit): elu(x) = x if x > 0, alpha *
    (e^x - 1) if x <= 0. Compare its dead neuron rate to ReLU on the same network.

Reading of the exercise: "the same network" is taken twice — the lesson's own
`dead_neuron_detector` with its `relu` swapped for ELU (check 2), and its
`ActivationNetwork` trained on circles at five learning rates (check 1), because
the detector turns out blind to the difference. Check 3 is the mechanism, check 4
asks what dead units cost, check 5 controls the ELU against the lesson's own.
"""

import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "04-activation-functions"
EPOCHS, RATES, HIDDEN = 200, (0.1, 0.3, 0.5, 1.0, 2.0), 8


def elu(x, alpha=1.0):
    """ELU exactly as the exercise writes it; expm1 is e^x - 1 without cancellation."""
    return x if x > 0 else alpha * math.expm1(x)


def elu_derivative(x, alpha=1.0):
    return 1.0 if x > 0 else alpha * math.exp(x)


def census(ref, act, hidden=20):
    """The lesson's own dead-neuron detector, its `relu` swapped for `act`."""
    seen = []

    def spy(z):
        seen.append((z, act(z)))
        return seen[-1][1]
    saved, ref.relu = ref.relu, spy
    with parity.quiet():
        ref.dead_neuron_detector()
    ref.relu = saved
    return [z for z, _v in seen], [sum(v > 0 for _z, v in seen[i::hidden])
                                   for i in range(hidden)]


def train(ref, act, deriv, data, lr):
    """The lesson's own train loop, then a census of units that fire for no input."""
    net = ref.ActivationNetwork(act, deriv, HIDDEN, lr)
    with parity.quiet():
        losses = net.train(data, epochs=EPOCHS)
    alive, correct = [0] * HIDDEN, 0
    for point, label in data:
        correct += (net.forward(point) >= 0.5) == (label >= 0.5)
        for i in range(HIDDEN):
            alive[i] += net.h[i] > 0
    return losses[-1], correct / len(data) * 100, alive.count(0)


def sweep(ref, act, deriv, data):
    """Loss, accuracy and dead-unit count after training at each learning rate."""
    runs = zip(*(train(ref, act, deriv, data, lr) for lr in RATES))
    return dict(zip(("loss", "acc", "dead"), (list(r) for r in runs)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data()
    zs, counts = census(ref, ref.relu)
    return {"same": counts == census(ref, elu)[1], "n": len(zs),
            "detector_dead": counts.count(0),
            "negative": sum(1 for z in zs if z <= 0),
            "relu_flat": sum(1 for z in zs if ref.relu_derivative(z) == 0.0),
            "elu_floor": min(elu_derivative(z) for z in zs),
            "gap": max(abs(elu(z, 0.0) - ref.relu(z)) for z in zs),
            "relu": sweep(ref, ref.relu, ref.relu_derivative, data),
            "elu": sweep(ref, elu, elu_derivative, data)}


def verify(result):
    relu, elu_ = result["relu"], result["elu"]
    best = relu["loss"].index(min(relu["loss"]))
    return [
        practice.Check("ANSWER: ELU's dead-unit rate is 0% at every lr; ReLU's 25%, not at 0.1",
                       all((sum(elu_["dead"]) == 0, max(relu["dead"]) == 2, relu["dead"][0] == 0)),
                       f"{EPOCHS} epochs on circles, units firing for no input at lr "
                       f"{list(RATES)}: ReLU {relu['dead']} of {HIDDEN}, ELU {elu_['dead']} — at "
                       f"the lesson's own lr = {RATES[0]} the dying-ReLU rate is 0%, and the "
                       f"25% starts at lr = {RATES[1]}"),
        practice.Check("FINDING: the lesson's own detector cannot tell ELU from ReLU at all",
                       all((result["same"], result["detector_dead"] == 0, result["negative"] > 0)),
                       f"`dead_neuron_detector` with its `relu` replaced by ELU gives fire "
                       f"counts identical on all 20 neurons over {result['n']} pre-activations "
                       f"({result['negative']} of them <= 0), {result['detector_dead']} dead for "
                       f"both. MECHANISM: elu(z) > 0 exactly when relu(z) > 0, so its "
                       f"`act(z) > 0` test is a test of sign(z), not of the activation"),
        practice.Check("MECHANISM: the difference is in the derivative, not the output",
                       result["relu_flat"] > 0 and result["elu_floor"] > 0,
                       f"relu'(z) is exactly 0.0 on {result['relu_flat']} of {result['n']} z "
                       f"values; elu'(z) = alpha*e^z never falls below {result['elu_floor']:.2e}. "
                       f"A ReLU unit negative on every sample gets zero into w1 and b1 and can "
                       f"never come back; ELU has no such state"),
        practice.Check("FINDING: dead units are not what costs accuracy here",
                       min(relu["loss"]) < min(elu_["loss"]) and relu["dead"][best] > 0,
                       f"the best fit in the sweep is ReLU at lr = {RATES[best]} — loss "
                       f"{relu['loss'][best]:.4f}, accuracy {relu['acc'][best]:.1f}%, with "
                       f"{relu['dead'][best]} of {HIDDEN} units dead. ELU keeps all {HIDDEN} "
                       f"alive, best loss {min(elu_['loss']):.4f}, and at lr = {RATES[-1]} it is "
                       f"{elu_['loss'][-1]:.4f} against ReLU's {relu['loss'][-1]:.4f}"),
        practice.Check("CONTROL: at alpha = 0 this ELU is the lesson's ReLU, exactly",
                       result["gap"] == 0.0,
                       f"max |elu(z, alpha=0) - relu(z)| over the detector's {result['n']} z "
                       f"values is {result['gap']:.1f} — the comparison runs against the "
                       f"lesson's own function, and elu(0) = alpha*(e^0 - 1) = 0 = relu(0)"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
