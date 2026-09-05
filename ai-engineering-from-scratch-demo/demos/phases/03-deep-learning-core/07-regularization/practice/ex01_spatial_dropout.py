"""Exercise 1 — drop whole channels instead of single units, and compare the gap.

    Implement spatial dropout for 2D data: instead of dropping individual
    neurons, drop entire feature channels. Simulate this by treating groups of
    consecutive features as channels and dropping whole groups. Compare the
    train-test gap to standard dropout on the circle dataset with
    hidden_size=32.

Reading of the exercise: channel dropout is one Bernoulli draw per channel rather
than per unit, so it is the lesson's own `Dropout` run on a length-C vector of
ones and broadcast — which makes channel width 1 exactly the lesson's dropout
(check 1) and lets the same `RegularizedNetwork` train both. The comparison the
exercise asks for is then a gap table (check 4), but it only means something next
to its own noise, so checks 5 and 6 measure the seed spread and the one width
that actually moves anything.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "07-regularization"
HIDDEN, P, SEEDS = 32, 0.3, tuple(range(1, 9))


class Spatial:
    """Channel dropout: the lesson's Dropout drawn once per channel, broadcast."""

    def __init__(self, ref, p, n_features, width):
        self.p, self.width = p, width
        self.inner = ref.Dropout(p=p)
        self.n_features, self.training, self.mask = n_features, True, None

    def forward(self, x):
        if not self.training:
            return list(x)
        self.inner.training = True
        mult = self.inner.forward([1.0] * (self.n_features // self.width))
        self.mask = [1 if mult[i // self.width] else 0 for i in range(len(x))]
        return [x[i] * mult[i // self.width] for i in range(len(x))]


def train(ref, data, width, seed):
    """One 300-epoch run of the lesson's network; width 0 means no dropout at all."""
    net = ref.RegularizedNetwork(hidden_size=HIDDEN, lr=0.05, dropout_p=P if width else 0.0)
    if width > 1:
        net.dropout = Spatial(ref, P, HIDDEN, width)
    random.seed(seed)                      # the lesson's Dropout draws from the global RNG
    with parity.quiet():
        _, train_acc, _, test_acc = net.train_model(data[:150], data[150:], epochs=300)[-1]
    return train_acc - test_acc, test_acc


def spread(ref, width, trials=6000):
    """Mask statistics for a flat activation vector: mean, sd of the sum, all-off rate."""
    drop, ones = Spatial(ref, P, HIDDEN, width), [1.0] * HIDDEN
    random.seed(4)
    sums = [sum(drop.forward(ones)) for _ in range(trials)]
    return {"mean": statistics.mean(sums), "sd": statistics.pstdev(sums),
            "off": sum(1 for s in sums if s == 0) / trials}


def _gap(pairs):
    return statistics.mean(g for g, _ in pairs), statistics.pstdev(g for g, _ in pairs)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data(n=300, seed=42)
    noise = {w: spread(ref, w) for w in (1, 2, 4, 8, 16)}
    gaps = {w: _gap([train(ref, data, w, s) for s in SEEDS]) for w in (1, 4, 16)}
    random.seed(9)
    lesson_out = ref.Dropout(p=P).forward([1.0, 2.0, 3.0, 4.0])
    random.seed(9)
    return {"plain": train(ref, data, 0, 1)[0], "gaps": gaps, "noise": noise,
            "identical": lesson_out == Spatial(ref, P, 4, 1).forward([1.0, 2.0, 3.0, 4.0]),
            "test16": statistics.mean(a for _, a in [train(ref, data, 16, s) for s in SEEDS]),
            "means": ", ".join(f"w{w} {noise[w]['mean']:.2f}" for w in noise),
            "sds": ", ".join(f"w{w} {noise[w]['sd']:.2f}" for w in noise)}


def verify(result):
    noise, gaps, plain = result["noise"], result["gaps"], result["plain"]
    drift = max(abs(noise[w]["mean"] / HIDDEN - 1) for w in noise)
    ratios = [noise[w]["sd"] / noise[1]["sd"] / w ** 0.5 for w in (2, 4, 8, 16)]
    table = " | ".join(f"width {w}: {gaps[w][0]:+.2f} ± {gaps[w][1]:.2f} pp" for w in (1, 4, 16))
    effect = abs(gaps[4][0] - gaps[1][0])
    return [
        practice.Check("CONTROL: at channel width 1 this is the lesson's Dropout, value for value",
                       result["identical"],
                       "same global RNG stream, same 1/(1-p) scaling — so every width below is "
                       "the same code regrouped, not a reimplementation"),
        practice.Check("MECHANISM: grouping keeps the expected activation and multiplies only "
                       "its variance, by sqrt(channel width)",
                       drift < 0.02 and max(abs(r - 1) for r in ratios) < 0.06,
                       f"over 6000 masks the summed activation has mean {result['means']} against "
                       f"32.00 flat (worst drift {drift * 100:.1f}%) and sd {result['sds']} — which "
                       f"divided by sqrt(width) is a flat {noise[1]['sd']:.2f}. Inverted scaling is "
                       f"per-channel, but C draws replace 32"),
        practice.Check("ANSWER: spatial dropout does not beat standard dropout — both sit ~0.8 pp "
                       "under the unregularized gap",
                       effect < 0.5 and gaps[1][0] < plain,
                       f"train-test gap over 8 mask seeds — {table}; no dropout at all: "
                       f"{plain:+.2f} pp (deterministic, no mask to vary)"),
        practice.Check("FINDING: the requested comparison is below the resolution of the fixture "
                       "it is requested on",
                       gaps[4][1] > 2 * effect,
                       f"width 4 and width 1 differ by {effect:.2f} pp against a seed-to-seed sd of "
                       f"{gaps[4][1]:.2f} pp, on a 150-point test set where one flipped label is "
                       f"0.67 pp. The effect is a fraction of one test point"),
        practice.Check("FINDING: only width 16 moves, and it underfits rather than regularizes",
                       gaps[16][0] < 0 and noise[16]["off"] > 0.05,
                       f"gap {gaps[16][0]:+.2f} pp — negative, test above train — at test accuracy "
                       f"{result['test16']:.2f}%. With 2 channels both are off in "
                       f"{noise[16]['off'] * 100:.1f}% of passes (p^C = {P ** 2:.2%}), erasing the "
                       f"hidden layer entirely that often and leaving the output on b2 alone"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
