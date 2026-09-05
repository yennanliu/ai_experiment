"""Exercise 5 — the leaky step on XOR, and what smoothness is actually buying.

    Replace sigmoid with a "leaky step" function: return 0.01 * z if z < 0, else
    1.0. Run the forward pass on XOR with the same hand-tuned weights from Step 4.
    Does it still work? Why is the smooth sigmoid preferred over hard cutoffs?

Reading of the exercise: "does it still work" is answered by running it (check 1) and then at
other weight scales (check 2), since one fixture passing is not an activation working. "Why is
sigmoid preferred" has a measurable answer, so checks 3 and 4 take finite-difference gradients
rather than assert one; check 5 separates the two properties the question conflates.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "02-multi-layer-networks"
XOR = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
THETA = [20.0, 20.0, -20.0, -20.0, -10.0, 30.0, 20.0, 20.0, -30.0]   # the lesson's Step 4
SCALES, EPS, TINY = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], 1e-5, 1e-12
LEAKY = lambda z: 0.01 * z if z < 0 else 1.0                          # noqa: E731
STEP = lambda z: 0.0 if z < 0 else 1.0                                # noqa: E731
LINEAR = lambda z: z                                                  # noqa: E731


def net(ref, theta):        # the lesson's own 2-2-1 stack, from a flat 9-parameter vector
    return ref.Network([
        ref.Layer(2, 2, weights=[theta[0:2], theta[2:4]], biases=list(theta[4:6])),
        ref.Layer(2, 1, weights=[list(theta[6:8])], biases=[theta[8]])])


def out(ref, act, theta, x) -> float:   # one pass, with `act` swapped in for `sigmoid`
    real, ref.sigmoid = ref.sigmoid, act
    value = net(ref, theta).forward(x)[0]
    ref.sigmoid = real
    return value


def rows(ref, act, scale=1.0) -> dict:
    theta = [v * scale for v in THETA]
    outs = [out(ref, act, theta, x) for x, _t in XOR]
    return {"outs": outs, "correct": sum((o >= 0.5) == bool(t) for o, (_, t) in zip(outs, XOR))}


def grads(ref, act, scale) -> dict:     # d(out)/d(theta), central differences, 9 x 4
    base, g = [v * scale for v in THETA], []
    for x, _t in XOR:
        for i in range(len(THETA)):
            hi, lo = list(base), list(base)
            hi[i], lo[i] = base[i] + EPS, base[i] - EPS
            g.append((out(ref, act, hi, x) - out(ref, act, lo, x)) / (2 * EPS))
    return {"max": max(abs(v) for v in g), "zeros": sum(v == 0.0 for v in g), "n": len(g)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    acts = {"sigmoid": ref.sigmoid, "leaky": LEAKY, "step": STEP, "linear": LINEAR}
    return {"rows": {name: rows(ref, act) for name, act in acts.items()},
            "sweep": {name: [rows(ref, acts[name], s)["correct"] for s in SCALES]
                      for name in ("sigmoid", "leaky")},
            "grad": {name: {s: grads(ref, acts[name], s) for s in (1.0, 0.1)}
                     for name in ("sigmoid", "leaky")},
            "jump": {"lk": LEAKY(0.0) - LEAKY(-TINY), "sg": ref.sigmoid(0.0) - ref.sigmoid(-TINY)}}


def digest(result) -> tuple:
    """The cross-scale summaries, kept out of `verify` so that only compares numbers."""
    grad = result["grad"]
    sweep = " ".join(f"s={s}: {sg}/{lk}" for s, sg, lk
                     in zip(SCALES, result["sweep"]["sigmoid"], result["sweep"]["leaky"]))
    fine, hand = {k: grad[k][0.1] for k in grad}, {k: grad[k][1.0] for k in grad}
    return sweep, fine, hand, fine["sigmoid"]["max"] / hand["sigmoid"]["max"]


def verify(result):
    row, jump = result["rows"], result["jump"]
    sweep, fine, hand, dulled = digest(result)
    return [
        practice.Check("ANSWER: yes it still works — but the outputs stop being probabilities",
                       row["leaky"]["correct"] == 4 == row["sigmoid"]["correct"]
                       and min(row["leaky"]["outs"]) < 0,
                       "leaky step on the Step 4 weights gives "
                       + ", ".join(f"{o:+.3f}" for o in row["leaky"]["outs"])
                       + f" — {row['leaky']['correct']}/4, the rows sigmoid gets too, but two are "
                       f"negative, so only the >= 0.5 rule still reads them"),
        practice.Check("FINDING: it works over a wider range of weights than sigmoid does",
                       min(result["sweep"]["leaky"]) == 4 and min(result["sweep"]["sigmoid"]) < 4,
                       f"rows correct as the weights are scaled, sigmoid/leaky — {sweep}. Scaling z "
                       f"by s leaves the leaky step's sign pattern intact but pulls sigmoid off its "
                       f"ends: below s = 0.2 the lesson's weights fail XOR"),
        practice.Check("WHY: the gradient — leaky step has almost none, and never above 0.01",
                       fine["sigmoid"]["max"] > 10 * fine["leaky"]["max"]
                       and fine["leaky"]["zeros"] > 2 * fine["sigmoid"]["zeros"],
                       f"central differences over all {fine['leaky']['n']} (parameter, row) pairs "
                       f"at s = 0.1: max |d out/d theta| is {fine['sigmoid']['max']:.4f} for "
                       f"sigmoid against {fine['leaky']['max']:.4f} for leaky, *exactly* zero in "
                       f"{fine['leaky']['zeros']} of them against {fine['sigmoid']['zeros']}. "
                       f"MECHANISM: its slope is 0.01 below zero and 0 above, and it jumps "
                       f"{jump['lk']:.4f} at z = 0 where sigmoid moves {jump['sg']:.2e}"),
        practice.Check("FINDING: at the Step 4 weights sigmoid has no usable gradient either",
                       hand["sigmoid"]["max"] < 1e-4
                       and hand["leaky"]["max"] > hand["sigmoid"]["max"],
                       f"at s = 1.0 the largest sigmoid gradient is {hand['sigmoid']['max']:.2e}, "
                       f"{dulled:.0f}x smaller than at s = 0.1 — z sits at +-10 and +-30, where "
                       f"sigmoid is flat — and leaky's {hand['leaky']['max']:.4f} is the larger "
                       f"there. Smoothness pays only where a unit is not already saturated"),
        practice.Check("CONTROL: what solves XOR is the nonlinearity, not the smoothness",
                       row["step"]["correct"] == 4 and row["linear"]["correct"] == 2
                       and len(set(row["linear"]["outs"])) == 1,
                       f"a hard step with no leak also gets {row['step']['correct']}/4; drop the "
                       f"nonlinearity instead (z -> z) and the same weights give "
                       f"{row['linear']['outs'][0]:.1f} for all four inputs, "
                       f"{row['linear']['correct']}/4 — two affine layers compose to one"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
