"""Exercise 3 — a 3-input majority perceptron, and whether majority is separable.

    Build a 3-input perceptron that outputs 1 only when at least 2 of the 3 inputs
    are 1 (a majority vote function). Is this linearly separable? Why?

Reading of the exercise: "why" deserves better than "because training worked" —
training working is evidence, not a reason. Check 3 gives the reason (the
function's own definition is already a threshold test) and check 5 gives it a
denominator: a linear program decides all 256 three-input Boolean functions, so
"separable" can be reported as a rate rather than as a property majority happens
to have. Parity is the control, and it is the one the lesson's XOR section is about.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "01-the-perceptron"
POINTS = list(itertools.product([0, 1], repeat=3))
MAJORITY = [(list(p), int(sum(p) >= 2)) for p in POINTS]
PARITY = [(list(p), sum(p) % 2) for p in POINTS]
HAND = ([1.0, 1.0, 1.0], -1.5)      # "at least 2 of 3" written directly as a hyperplane


def train(Perceptron, data, epochs=1000):
    unit = Perceptron(3)
    for epoch in range(epochs):
        mistakes = 0
        for point, target in data:
            error = target - unit.predict(point)
            if error:
                mistakes += 1
                unit.weights = [w + unit.lr * error * x
                                for w, x in zip(unit.weights, point)]
                unit.bias += unit.lr * error
        if not mistakes:
            return unit, epoch + 1
    return unit, None


def separable(linprog, labels) -> bool:
    """Exact: maximise the margin t subject to s_i·(w·x_i + b) >= t, |w| <= 1."""
    rows = [[-s * x[0], -s * x[1], -s * x[2], -s, 1.0]
            for x, s in zip(POINTS, [1 if y else -1 for y in labels])]
    found = linprog(c=[0, 0, 0, 0, -1], A_ub=rows, b_ub=[0.0] * len(rows),
                    bounds=[(-1, 1)] * 4 + [(0, 1)], method="highs")
    return bool(found.success and found.x[4] > 1e-7)


def scores_of(weights, bias) -> list:
    return [sum(w * x for w, x in zip(weights, p)) + bias for p, _ in MAJORITY]


def solve():
    try:
        from scipy.optimize import linprog
    except ImportError as exc:                       # pragma: no cover - env guard
        raise practice.Skip(f"needs scipy for the exact census: uv sync --extra math ({exc})")
    ref = parity.load_reference(PHASE, LESSON, "perceptron")
    with parity.quiet():
        unit, epochs = train(ref.Perceptron, MAJORITY)
        _stuck, parity_epochs = train(ref.Perceptron, PARITY)
    scores, hand = scores_of(unit.weights, unit.bias), scores_of(*HAND)
    census = sum(separable(linprog, bits) for bits in itertools.product([0, 1], repeat=8))
    return {
        "weights": unit.weights, "bias": unit.bias, "epochs": epochs,
        "correct": sum(unit.predict(p) == t for p, t in MAJORITY),
        "parity_epochs": parity_epochs, "scores": scores,
        "on_boundary": sum(abs(z) < 1e-12 for z in scores),
        "hand_margin": min(abs(z) for z in hand) / 3 ** 0.5,
        "separable": census,
        "majority_separable": separable(linprog, [t for _p, t in MAJORITY]),
        "parity_separable": separable(linprog, [t for _p, t in PARITY]),
    }


def verify(result):
    weights = ", ".join(f"{w:+.1f}" for w in result["weights"])
    return [
        practice.Check("ANSWER: yes — the 3-input majority perceptron trains and is exact",
                       result["epochs"] is not None and result["correct"] == 8,
                       f"converged in {result['epochs']} epochs, all 8 rows correct, "
                       f"w = [{weights}], b = {result['bias']:+.1f} — one weight per input, "
                       f"equal, with a threshold between 1 and 2 votes"),
        practice.Check("CONTROL: 3-input parity, the same shape of problem, does not train",
                       result["parity_epochs"] is None and not result["parity_separable"],
                       "XOR of three inputs runs 1000 epochs without converging, and the "
                       "linear program confirms no hyperplane exists — this is the lesson's "
                       "XOR result at n = 3"),
        practice.Check("WHY: majority is already written as a hyperplane test",
                       result["majority_separable"],
                       "'at least 2 of the 3 inputs are 1' is sum(x) >= 2, which is "
                       "w·x + b >= 0 at w = [1, 1, 1], b = -1.5 — the specification of the "
                       "function *is* the separating plane, so no search is needed. Every "
                       "symmetric threshold function is separable for the same reason; "
                       "parity is not one, because its output is not monotone in the vote count"),
        practice.Check("FINDING: three of the four positive rows sit exactly on the "
                       "trained boundary",
                       result["on_boundary"] == 3 and result["hand_margin"] > 0.28,
                       "w·x + b: " + ", ".join(f"{z:+.1f}" for z in result["scores"])
                       + f" — the three two-vote rows score exactly 0 and are called 1 only "
                       f"by `predict`'s >= tie-break. The hand-written w = [1, 1, 1], "
                       f"b = -1.5 separates the same 8 rows with margin "
                       f"{result['hand_margin']:.4f} and needs no tie-break"),
        practice.Check("…and separability is the exception, not the rule",
                       result["separable"] == 104,
                       f"a linear program decides all 2^8 = 256 Boolean functions of three "
                       f"inputs: {result['separable']} are linearly separable — 40.6%. "
                       f"Majority is one of them and parity is not, so 'is this separable?' "
                       f"is a real question at n = 3 even though every 2-input gate but XOR "
                       f"and XNOR is (14 of 16)"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
