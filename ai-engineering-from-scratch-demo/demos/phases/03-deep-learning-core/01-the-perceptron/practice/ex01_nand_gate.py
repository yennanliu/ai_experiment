"""Exercise 1 — train a perceptron on NAND, then check the boundary it claims.

    Train a perceptron on a NAND gate (the universal gate - any logic circuit can
    be built from NAND). Verify its weights and bias form a valid decision
    boundary.

Reading of the exercise: "verify … a valid decision boundary" is only worth doing
if "valid" is taken strictly — a separating hyperplane, not four lucky
predictions. Check 3 measures the margin the trained weights achieve and check 4
re-runs them with the tie-break flipped; checks 5 and 6 take the parenthesis at
its word, so XOR gets built out of the trained unit.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "01-the-perceptron"
NAND = [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
STRICT = ([-1.0, -1.0], 1.5)   # the hand-set NAND of the lesson's xor_network


def net_input(weights, bias, point) -> float:
    return sum(w * x for w, x in zip(weights, point)) + bias


def train_nand(Perceptron):
    """The lesson's own train loop, re-run so the epoch count is observable."""
    unit = Perceptron(2)
    for epoch in range(100):
        mistakes = 0
        for point, target in NAND:
            error = target - unit.predict(point)
            if error:
                mistakes += 1
                unit.weights = [w + unit.lr * error * x for w, x in zip(unit.weights, point)]
                unit.bias += unit.lr * error
        if not mistakes:
            return unit, epoch + 1
    return unit, None


def from_nand(gate):
    """NOT/AND/OR/XOR wired from one two-input NAND, per the universality claim."""
    inv = lambda a: gate(a, a)                                          # noqa: E731
    wire = {"NOT": lambda a, _b: inv(a), "AND": lambda a, b: inv(gate(a, b)),
            "OR": lambda a, b: gate(inv(a), inv(b)),
            "XOR": lambda a, b: gate(gate(a, gate(a, b)), gate(b, gate(a, b)))}
    truth = {"NOT": lambda a, _b: 1 - a, "AND": min, "OR": max, "XOR": lambda a, b: a ^ b}
    return {name: all(fn(a, b) == truth[name](a, b)
                      for a in (0, 1) for b in (0, 1)) for name, fn in wire.items()}


def boundary(weights, bias) -> dict:
    """What the pair (weights, bias) actually claims about the four NAND rows."""
    scores = [net_input(weights, bias, point) for point, _ in NAND]
    norm = sum(w * w for w in weights) ** 0.5
    strict = [(1 if z > 0 else 0) != target for z, (_p, target) in zip(scores, NAND)]
    return {"scores": scores, "margin": min(abs(z) for z in scores) / norm,
            "strict_wrong": sum(strict)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "perceptron")
    with parity.quiet():
        unit, epochs = train_nand(ref.Perceptron)
    trained, hand = boundary(unit.weights, unit.bias), boundary(*STRICT)
    return {
        "weights": unit.weights, "bias": unit.bias, "epochs": epochs, **trained,
        "correct": sum(unit.predict(p) == t for p, t in NAND),
        "gates": from_nand(lambda a, b: unit.predict([a, b])),
        "hand_margin": hand["margin"], "hand_strict_wrong": hand["strict_wrong"],
    }


def verify(result):
    w, scores = result["weights"], result["scores"]
    line = f"w = [{w[0]:+.2f}, {w[1]:+.2f}], b = {result['bias']:+.2f}"
    return [
        practice.Check("the perceptron rule converges on NAND",
                       result["epochs"] is not None and result["correct"] == 4,
                       f"{result['epochs']} epochs from the all-zero start, all 4 rows "
                       f"correct — {line}"),
        practice.Check("…and those weights do separate the four points",
                       all((z >= 0) == bool(t) for z, (_p, t) in zip(scores, NAND)),
                       "w·x + b at (0,0), (0,1), (1,0), (1,1): "
                       + ", ".join(f"{z:+.4f}" for z in scores)),
        practice.Check("FINDING: the boundary is not a separating hyperplane — a "
                       "training point lies exactly on it",
                       abs(result["margin"]) < 1e-12,
                       f"geometric margin {result['margin']:.2e} — (1,0) scores exactly "
                       f"{scores[2]:+.4f}, so it sits *on* the line w·x + b = 0 rather "
                       f"than on the positive side of it"),
        practice.Check("…so the answer is carried by the tie-break, not by the weights",
                       result["strict_wrong"] == 1,
                       f"the lesson's `predict` returns 1 when the sum is >= 0; change "
                       f"that one comparison to > and this trained NAND gets "
                       f"{result['strict_wrong']} of 4 rows wrong. MECHANISM: the update "
                       f"fires only on an error, and z = 0 already predicts 1, so the rule "
                       f"halts the instant the last point reaches the line and never pushes "
                       f"it across"),
        practice.Check("ANSWER: universal — XOR built from this one trained unit",
                       all(result["gates"].values()),
                       "NOT, AND, OR and XOR wired from copies of the trained NAND, all 4 "
                       "rows each: " + ", ".join(f"{g} ok" for g in result["gates"])
                       + ". XOR is the gate a single perceptron provably cannot learn, and "
                       "five copies of one that can be learned produce it"),
        practice.Check("CONTROL: the lesson's hand-set NAND has a real margin",
                       result["hand_margin"] > 0.3 and result["hand_strict_wrong"] == 0,
                       f"w = [-1, -1], b = +1.5 — the unit `xor_network` hard-codes — "
                       f"separates with margin {result['hand_margin']:.4f} and survives the "
                       f"strict comparison. Training found a worse boundary than the lesson "
                       f"wrote down by hand, because nothing in the rule asks for margin"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
