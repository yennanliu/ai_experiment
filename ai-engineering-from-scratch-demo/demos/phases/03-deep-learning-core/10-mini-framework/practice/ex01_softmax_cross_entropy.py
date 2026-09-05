"""Exercise 1 — SoftmaxCrossEntropyLoss, and why its backward pass is combined.

    Add a `SoftmaxCrossEntropyLoss` class for multi-class classification. Softmax the
    predictions, compute cross-entropy loss, and handle the combined backward pass. Test
    it on a 3-class spiral dataset.

Reading of the exercise: "the combined backward pass" is the load-bearing clause — check 1
gradient-checks `p - y` against a central difference, check 4 recomputes it un-combined to find
where that form dies, and 2-3 are the spiral test against a label-permuted control.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "10-mini-framework"
H, CLASSES, PPC = 1e-6, 3, 60
CUT, CHANCE = int(0.8 * CLASSES * PPC), 100.0 / CLASSES     # 144 of 180 rows train; chance accuracy


class SoftmaxCrossEntropyLoss:
    def __call__(self, logits, target):          # softmax + CE as one node
        exps = [math.exp(v - max(logits)) for v in logits]
        self.probs, self.target = [e / sum(exps) for e in exps], target
        return -sum(t * math.log(max(p, 1e-15)) for p, t in zip(self.probs, target))

    def backward(self):
        return [p - t for p, t in zip(self.probs, self.target)]


def two_stage(logits, target):
    (crit := SoftmaxCrossEntropyLoss())(logits, target)   # combined p - y, then the un-combined route
    p, n, dp = crit.probs, len(logits), [-t / max(q, 1e-15) for q, t in zip(crit.probs, target)]
    return crit.backward(), [sum(dp[i] * p[i] * ((i == j) - p[j]) for i in range(n)) for j in range(n)]


def gradcheck(rng, trials=200):
    crit, worst = SoftmaxCrossEntropyLoss(), 0.0
    for _ in range(trials):
        z, y = [rng.uniform(-6, 6) for _ in range(CLASSES)], [0.0] * CLASSES
        y[rng.randrange(CLASSES)] = 1.0
        crit(z, y)
        for k, analytic in enumerate(crit.backward()):
            bump = lambda d: crit([v + (d if m == k else 0.0) for m, v in enumerate(z)], y)  # noqa: E731
            worst = max(worst, abs((bump(H) - bump(-H)) / (2 * H) - analytic))
    return worst


def spiral(seed=3, noise=0.15):
    rng = random.Random(seed)
    arms = [(c, 0.1 + 0.9 * i / PPC, rng.gauss(0, noise)) for c in range(CLASSES) for i in range(PPC)]
    data = [([r * math.sin(a), r * math.cos(a)], [1.0 * (k == c) for k in range(CLASSES)])
            for c, r, g in arms for a in [c * 2 * math.pi / CLASSES + g + 4 * r]]
    return rng.sample(data, len(data))


def fit(ref, train, epochs=30, lr=0.02):
    random.seed(42)                          # the reference Linear draws from the global RNG
    model = ref.Sequential(ref.Linear(2, 16), ref.ReLU(), ref.Linear(16, 16), ref.ReLU(), ref.Linear(16, 3))
    crit, opt, hist = SoftmaxCrossEntropyLoss(), ref.Adam(model.parameters(), lr=lr), []
    loader = ref.DataLoader(train, batch_size=16, shuffle=True)
    for _ in range(epochs):
        run = 0.0
        for inputs, targets in loader:
            for x, t in zip(inputs, targets):
                run += crit(model.forward(x), t)
                opt.zero_grad(), model.backward(crit.backward()), opt.step()
        hist.append(run / len(train))
    return model, hist


def accuracy(model, rows):
    top = lambda x: max(range(CLASSES), key=model.forward(x).__getitem__)      # noqa: E731
    return 100.0 * sum(top(x) == t.index(1.0) for x, t in rows) / len(rows)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    train, test = (data := spiral())[:CUT], data[CUT:]
    model, hist = fit(ref, train)
    random.Random(5).shuffle(labels := [t for _, t in train])
    ctrl, ctrl_hist = fit(ref, shuffled := list(zip([x for x, _ in train], labels)))
    return {"gradcheck": gradcheck(random.Random(11)), "loss0": hist[0], "loss_end": hist[-1], "n": CUT,
            "sat": {s: two_stage([s, 0.0, -s], [0.0, 0.0, 1.0]) for s in (5.0, 30.0, 400.0)},
            "train_acc": accuracy(model, train), "test_acc": accuracy(model, test), "n_test": len(test),
            "ctrl_train": accuracy(ctrl, shuffled), "ctrl_test": accuracy(ctrl, test), "ctrl_loss": ctrl_hist[-1]}


def verify(r):
    sat, (hi, lo) = r["sat"], (r["sat"][30.0][0][0], r["sat"][30.0][1][0])
    return [
        practice.Check("ANSWER: `p - y` IS the gradient of the softmax cross-entropy loss",
                       r["gradcheck"] < 1e-7,
                       f"worst |analytic - central difference| {r['gradcheck']:.3e} over 200 random logit "
                       f"triples x 3 logits at h = {H:g} — the difference quotient's own O(h^2) error"),
        practice.Check("ANSWER: the 3-class spiral is learned with it", r["test_acc"] > 85.0,
                       f"{r['n']}/{r['n_test']} split, Linear(2,16)-ReLU-Linear(16,16)-ReLU-Linear(16,3), "
                       f"30 epochs of Adam(lr=0.02): train {r['train_acc']:.1f}%, test {r['test_acc']:.1f}% "
                       f"against {CHANCE:.1f}% chance, mean loss {r['loss0']:.4f} -> {r['loss_end']:.6f}"),
        practice.Check("CONTROL: the identical run on permuted labels lands at chance",
                       r["ctrl_test"] < CHANCE + 5.0 and r["ctrl_loss"] > 0.9,
                       f"labels shuffled among the same points: train {r['ctrl_train']:.1f}%, test "
                       f"{r['ctrl_test']:.1f}%, loss stalls at {r['ctrl_loss']:.4f} vs ln 3 = {math.log(3):.4f}"),
        practice.Check("FINDING: the un-combined backward underflows to zero on confident logits",
                       abs(lo) < 1e-9 < abs(hi) and not any(sat[400.0][1]),
                       f"the softmax Jacobian composed with dCE/dp reproduces `p - y` exactly at logits (5, 0, -5), "
                       f"returns {lo:.3e} at (30, 0, -30) where the true gradient is {hi:.6f}, and returns exactly "
                       f"0.0 at (400, 0, -400). MECHANISM: dCE/dp = -1/p reaches ~1e+26 while the Jacobian row "
                       f"carries a factor p ~ 1e-26, so the product survives only if neither end is rounded away "
                       f"first — and `p - y` never forms either factor"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
