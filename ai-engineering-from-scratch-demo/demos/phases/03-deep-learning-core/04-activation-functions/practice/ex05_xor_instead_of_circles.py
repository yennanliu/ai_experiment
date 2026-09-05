"""Exercise 5 — the training comparison on XOR, and why it disagrees with circles.

    Modify the training comparison to use the XOR dataset from Lesson 01 instead
    of circles. Which activation converges fastest on XOR? Why does this differ
    from the circle results?

Reading of the exercise: "converges fastest" needs a unit, and the unit is where the answer
hides — check 3 shows that an epoch is 4 updates on XOR and 200 on circles, so the two rankings
are not measured in the same thing. Check 1 runs the comparison exactly as the lesson would,
at its own lr and epoch count; check 2 gives it room. Check 5 asks whether the answer is a
property of XOR at all.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "04-activation-functions"
XOR = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
NAMES = ("sigmoid", "tanh", "relu", "gelu", "swish")
HIDDEN, LR, SHORT, LONG, TARGET = 8, 0.1, 200, 10000, 0.05
INITS = (1, 2, 3)             # the lesson's ActivationNetwork re-seeds to 0, so these are ours


def acts(ref) -> dict:
    return {n: (getattr(ref, "tanh_act" if n == "tanh" else n),
                getattr(ref, f"{'tanh' if n == 'tanh' else n}_derivative")) for n in NAMES}


def reinit(net, seed) -> None:
    """A different draw of the lesson's own init — `__init__` hard-codes random.seed(0)."""
    rng = random.Random(seed)
    net.w1 = [[rng.gauss(0, 0.5) for _ in range(2)] for _ in range(HIDDEN)]
    net.w2 = [rng.gauss(0, 0.5) for _ in range(HIDDEN)]


def train(ref, act, deriv, data, epochs, seed=None) -> dict:
    """The lesson's own loop, counting exact-zero hidden-weight gradients as it goes."""
    net = ref.ActivationNetwork(act, deriv, HIDDEN, LR)
    if seed is not None:
        reinit(net, seed)
    reached, zeros, seen, loss, right = None, 0, 0, 1.0, 0
    for epoch in range(epochs):
        loss, right = 0.0, 0
        for x, target in data:
            pred = net.forward(x)
            d_out = (pred - target) * pred * (1 - pred)
            for i in range(HIDDEN):
                d_h = d_out * net.w2[i] * net.act_d(net.z1[i])
                zeros, seen = zeros + (d_h * x[0] == 0.0) + (d_h * x[1] == 0.0), seen + 2
            net.backward(target)
            loss, right = loss + (pred - target) ** 2, right + ((pred >= 0.5) == (target >= 0.5))
        loss = loss / len(data)
        reached = epoch + 1 if reached is None and loss < TARGET else reached
    return {"at": reached, "loss": loss, "acc": 100.0 * right / len(data),
            "zeros": zeros / seen, "updates": None if reached is None else reached * len(data)}


def sweep(ref, table, data, epochs, seed=None) -> dict:
    return {n: train(ref, *table[n], data, epochs, seed) for n in NAMES}


def winner(runs) -> str:
    return min(NAMES, key=lambda n: runs[n]["at"] if runs[n]["at"] else 10 ** 9)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table, circles = acts(ref), ref.make_circle_data()
    return {"short": sweep(ref, table, XOR, SHORT), "long": sweep(ref, table, XOR, LONG),
            "circles": sweep(ref, table, circles, SHORT), "n": len(circles),
            "alt_xor": {s: sweep(ref, table, XOR, 2000, s) for s in INITS},
            "alt_circ": {s: sweep(ref, table, circles, SHORT, s) for s in INITS}}


def order(runs, key="at") -> str:
    ranked = sorted(NAMES, key=lambda n: runs[n][key] if runs[n][key] else 10 ** 9)
    return ", ".join(f"{n} {runs[n][key]}" for n in ranked)


def digest(result) -> dict:
    """Every summary `verify` quotes, so that stays a list of comparisons."""
    short, long_, circles = result["short"], result["long"], result["circles"]
    return {"short_row": ", ".join(f"{n} {short[n]['loss']:.4f}/{short[n]['acc']:.0f}%"
                                   for n in NAMES),
            "solved": [n for n in NAMES if short[n]["at"]],
            "long_row": order(long_), "circ_row": order(circles),
            "long_upd": order(long_, "updates"), "circ_upd": order(circles, "updates"),
            "xor_winners": [winner(result["alt_xor"][s]) for s in INITS],
            "circ_winners": [winner(result["alt_circ"][s]) for s in INITS],
            "failed": [n for s in INITS for n in NAMES if result["alt_xor"][s][n]["at"] is None
                       and n != "sigmoid"],
            "zeros": (long_["relu"]["zeros"], circles["relu"]["zeros"])}


def verify(result):
    d, short, long_ = digest(result), result["short"], result["long"]
    return [
        practice.Check("ANSWER: at the lesson's own settings the comparison is degenerate",
                       d["solved"] == ["relu"] and short["relu"]["at"] == SHORT
                       and short["sigmoid"]["acc"] < 60,
                       f"{SHORT} epochs at the lesson's own lr = {LR}, final loss/accuracy: "
                       f"{d['short_row']}. Only relu reaches loss < {TARGET}, and it does so at "
                       f"epoch {short['relu']['at']} — the last one — while sigmoid ends at "
                       f"{short['sigmoid']['acc']:.0f}%, which is chance on four rows"),
        practice.Check("ANSWER: given room, relu converges fastest on XOR",
                       winner(long_) == "relu",
                       f"epochs to loss < {TARGET} over {LONG:,}: {d['long_row']}. The lesson's "
                       f"{SHORT}-epoch budget stops inside relu's margin and before anything else "
                       f"arrives, so its own settings cannot show this"),
        practice.Check("WHY it differs from circles: a different winner, in a different unit",
                       winner(result["circles"]) != "relu",
                       f"on circles the same code ranks {d['circ_row']} epochs — gelu and swish "
                       f"first, relu third. But an epoch is {len(XOR)} updates on XOR and "
                       f"{result['n']} on circles, so the two rankings are not in the same unit: "
                       f"in *updates* XOR needs {d['long_upd']} against circles' {d['circ_upd']}"),
        practice.Check("MECHANISM: XOR zeroes most of the hidden-layer gradient outright",
                       d["zeros"][0] > 0.8 > d["zeros"][1],
                       f"exact-zero hidden weight gradients, relu: {d['zeros'][0]:.1%} on XOR "
                       f"against {d['zeros'][1]:.1%} on circles. `d_h * x[j]` is exactly 0 "
                       f"whenever x[j] is, and half of XOR's coordinates are 0 — on circles no "
                       f"input coordinate is ever exactly 0, so its {d['zeros'][1]:.1%} is dead "
                       f"units alone. Four clean rows still need fewer updates than 200 noisy "
                       f"ones; what changes is which activation collects them fastest"),
        practice.Check("FINDING: on XOR the answer is not stable, and the lesson cannot see that",
                       len(set(d["xor_winners"])) > 1 and d["failed"]
                       and set(d["circ_winners"]) <= {"gelu", "swish"},
                       f"redraw the same init at three seeds and the XOR winner is "
                       + ", ".join(d["xor_winners"]) + " — including " + ", ".join(d["failed"])
                       + f", which fails to reach loss < {TARGET} in 2,000 epochs at one of them. "
                       f"On circles it is " + ", ".join(d["circ_winners"]) + ", always one of the "
                       "two smooth activations, and nothing fails. `ActivationNetwork.__init__` "
                       "calls random.seed(0), so the comparison the lesson ships is one draw and "
                       "its caller has no way to ask for another"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
