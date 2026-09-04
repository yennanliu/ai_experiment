"""Exercise 2 — track w1*x1 + w2*x2 + b = 0 per epoch while AND is learned.

    Modify the Perceptron class to track the decision boundary (w1*x1 + w2*x2 + b
    = 0) at each epoch. Print how the line shifts during training on the AND gate.

Reading of the exercise: the interesting thing about a line's history is where it
starts, what can move it, and where it stops. Check 2 reports the start (there
isn't one), check 3 the mechanism that limits each move, and checks 4 and 5 the
stop — which turns out to be two epochs early, and a rounding error is why. The
whole trajectory is in the detail strings, so the "print" is the checks' output.
"""

from __future__ import annotations

from fractions import Fraction

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "01-the-perceptron"
AND = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
MAX_MARGIN = 0.5 / 2 ** 0.5     # the AND hyperplane x1 + x2 = 1.5, for comparison


def tracking(Perceptron):
    """The lesson's class, with the boundary recorded per epoch. `predict` is
    inherited untouched, so this tracks the lesson's own trajectory."""
    class Tracking(Perceptron):
        def train(self, data, epochs=60):           # noqa: D102 - overrides the lesson's
            self.history = []
            for _epoch in range(epochs):
                moved = [tuple(point) for point, target in data
                         if self._correct(point, target - self.predict(point))]
                self.history.append((list(self.weights), self.bias, moved))
                if not moved:
                    return self.history
            return self.history

        def _correct(self, point, error):
            if not error:
                return False
            self.weights = [w + self.lr * error * x for w, x in zip(self.weights, point)]
            self.bias = self.bias + self.lr * error
            return True
    return Tracking


def track(Perceptron, zero, rate):
    unit = tracking(Perceptron)(2, rate)
    unit.weights, unit.bias = [zero, zero], zero
    return unit.train(AND)


def as_line(weights, bias) -> str:
    if not weights[1]:
        return "no line — 0·x1 + 0·x2 + 0 = 0 holds everywhere" if not weights[0] \
            else f"vertical, x1 = {-bias / weights[0]:+.2f}"
    return f"x2 = {-weights[0] / weights[1]:+.3f}·x1 {-bias / weights[1]:+.3f}"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "perceptron")
    history = track(ref.Perceptron, 0.0, 0.1)
    weights, bias, _ = history[-1]
    scores = [sum(w * x for w, x in zip(weights, point)) + bias for point, _ in AND]
    exact = track(ref.Perceptron, Fraction(0), Fraction(1, 10))
    return {
        "history": history, "weights": weights, "bias": bias, "scores": scores,
        "epochs": len(history), "exact_epochs": len(exact), "exact_bias": exact[-1][1],
        "slope": -weights[0] / weights[1],
        "margin": min(abs(z) for z in scores) / sum(w * w for w in weights) ** 0.5,
        "translation_only": [ep for ep, (_w, _b, moved) in enumerate(history, 1)
                             if (0, 0) in moved],
    }


def verify(result):
    history, scores = result["history"], result["scores"]
    trail = " | ".join(f"e{i}: {as_line(w, b)}" for i, (w, b, _m) in enumerate(history, 1))
    return [
        practice.Check("the AND boundary, epoch by epoch",
                       result["epochs"] == 4,
                       f"from w = [0, 0], b = 0 — {trail}"),
        practice.Check("at the all-zero start there is no boundary to track",
                       as_line([0.0, 0.0], 0.0).startswith("no line"),
                       "the lesson initialises weights and bias to zero, so the tracked "
                       "equation is 0·x1 + 0·x2 + 0 = 0 — satisfied by every point in the "
                       "plane. A line only exists from epoch 1, after the first update"),
        practice.Check("MECHANISM: the (0,0) row can translate the line but never rotate it",
                       result["translation_only"] == [1, 2],
                       f"the update is w_i += lr·error·x_i, and (0,0) has x_i = 0 — so its "
                       f"corrections in epochs {result['translation_only']} move the bias "
                       f"alone. Every rotation in the trail above comes from one of the "
                       f"other three rows"),
        practice.Check("FINDING: the float run stops two epochs early, on a line that is "
                       "not an AND boundary",
                       result["exact_epochs"] == 6
                       and result["exact_bias"] == Fraction(-3, 10),
                       f"it halts at epoch {result['epochs']} with b = {result['bias']!r}, "
                       f"where (1,0) scores {scores[2]:.2e} — negative only by rounding. "
                       f"Re-run in exact rationals and (1,0) scores exactly 0, `predict` "
                       f"returns 1 for a 0-labelled row, and training continues to epoch "
                       f"{result['exact_epochs']} and b = {result['exact_bias']}. The "
                       f"reported answer is a float artefact of accumulating ±0.1"),
        practice.Check("ANSWER: the line stops at the first one that fits, not the best one",
                       result["margin"] < 1e-12 and abs(result["slope"] + 2.0) < 1e-12,
                       f"final slope {result['slope']:+.1f} at geometric margin "
                       f"{result['margin']:.2e}, against slope -1.0 and margin "
                       f"{MAX_MARGIN:.4f} for the max-margin line x1 + x2 = 1.5. The rule "
                       f"updates only on an error, so it has no reason to keep moving once "
                       f"every point is on the correct closed side — and the line it leaves "
                       f"behind is touching the data"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
