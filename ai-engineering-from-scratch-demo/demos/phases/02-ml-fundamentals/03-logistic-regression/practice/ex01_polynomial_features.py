"""Exercise 1 — concentric circles: linear logistic fails, polynomial features fix it.

    Generate a dataset that is NOT linearly separable (e.g., two concentric
    circles). Train logistic regression and observe its failure. Then add
    polynomial features (x1^2, x2^2, x1*x2) and train again. Show that the
    accuracy improves.

Reading of the exercise: "observe its failure" needs a floor to compare against,
or 50% looks like a number rather than a verdict — for balanced concentric
circles, a linear model cannot beat the majority-class baseline, and check 2
asserts exactly that. Of the three features the exercise names, only x1²+x2²
carries the signal; check 5 shows x1·x2 alone reaches only 59% against the
squares' 100%, so "add polynomial features" works because of *which* one.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "03-logistic-regression"
SEED, N_PER, EPOCHS, LR = 42, 150, 4_000, 0.5
INNER, OUTER = 1.0, 2.5


def circles(rng):
    X, y = [], []
    for label, radius in ((0, INNER), (1, OUTER)):
        for _ in range(N_PER):
            angle = rng.uniform(0, 2 * math.pi)
            r = radius + rng.gauss(0, 0.15)
            X.append([r * math.cos(angle), r * math.sin(angle)])
            y.append(label)
    return X, y


def expand(X, which):
    """which: 'linear', 'radius' (x1²+x2²), 'cross' (x1·x2), or 'full'."""
    out = []
    for x1, x2 in X:
        row = [x1, x2]
        if which in ("radius", "full"):
            row += [x1 * x1, x2 * x2]
        if which in ("cross", "full"):
            row.append(x1 * x2)
        out.append(row)
    return out


def train(ref, X, y):
    model = ref.LogisticRegression(n_features=len(X[0]), learning_rate=LR)
    model.fit(X, y, epochs=EPOCHS, print_every=EPOCHS + 1)
    return model.accuracy(X, y)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "logistic_regression")
    rng = random.Random(SEED)
    X, y = circles(rng)
    baseline = max(y.count(0), y.count(1)) / len(y)
    scores = {k: train(ref, expand(X, k), y)
              for k in ("linear", "radius", "cross", "full")}
    return {"scores": scores, "baseline": baseline, "n": len(y)}


def verify(result):
    s, baseline = result["scores"], result["baseline"]
    return [
        practice.Check(f"{result['n']} points on two concentric circles, balanced",
                       abs(baseline - 0.5) < 1e-9,
                       f"majority-class baseline {baseline:.1%} — the floor any model must "
                       f"beat to have learned anything"),
        practice.Check("ANSWER: the linear model stays near that floor",
                       s["linear"] < 0.60 and s["linear"] < s["radius"] - 0.35,
                       f"accuracy {s['linear']:.1%} against a {baseline:.1%} baseline — "
                       f"{100 * (s['radius'] - s['linear']):.0f} points short of what the "
                       f"right features give. No line separates an annulus from its "
                       f"centre; the few points above chance come from sampling noise in "
                       f"the ring, not from learning"),
        practice.Check("ANSWER: adding x1², x2² and x1·x2 fixes it",
                       s["full"] > 0.95,
                       f"accuracy {s['full']:.1%} — a {100 * (s['full'] - s['linear']):.0f} "
                       f"point gain, from features alone with the same model"),
        practice.Check("…because x1² + x2² is a circle in the original space",
                       s["radius"] > 0.95,
                       f"the two squared terms alone give {s['radius']:.1%}: a linear "
                       f"boundary in (x1², x2²) is a conic in (x1, x2), which is exactly "
                       f"what separates two radii"),
        practice.Check("FINDING: x1·x2 alone barely helps — which term matters, not how many",
                       s["cross"] < 0.70 and s["cross"] < s["radius"] - 0.30,
                       f"cross term only: {s['cross']:.1%} against {s['radius']:.1%} for "
                       f"the squares. x1·x2 encodes orientation and these classes differ "
                       f"only in radius, so 'add polynomial features' works because of "
                       f"*which* term is added"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
