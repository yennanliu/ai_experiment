"""Exercise 1 — drop whole channels instead of single units, and compare the gap.

    Implement spatial dropout for 2D data: instead of dropping individual
    neurons, drop entire feature channels. Simulate this by treating groups of
    consecutive features as channels and dropping whole groups. Compare the
    train-test gap to standard dropout on the circle dataset with
    hidden_size=32.

Reading of the exercise: channel dropout is one Bernoulli draw per channel rather than per
unit — the lesson's own `Dropout` on a length-C vector of ones, broadcast — so width 1 is
the lesson's dropout exactly (check 1) and one `RegularizedNetwork` trains every width.
Check 3 is the gap table asked for, graded against its own seed noise rather than zero.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "07-regularization"
HIDDEN, P, SEEDS = 32, 0.3, tuple(range(1, 9))


class Spatial:
    """Channel dropout: the lesson's Dropout drawn once per channel, then broadcast."""
    def __init__(self, ref, p, n_features, width):
        self.p, self.width, self.n_features = p, width, n_features
        self.inner, self.training, self.mask = ref.Dropout(p=p), True, None

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


def identical(ref, x=(1.0, 2.0, 3.0, 4.0)):
    """Width 1 against the lesson's Dropout on one shared RNG stream, value for value."""
    random.seed(9)
    lesson = ref.Dropout(p=P).forward(x)
    random.seed(9)
    return lesson == Spatial(ref, P, len(x), 1).forward(x)


def _stats(pairs):
    return (statistics.mean(g for g, _a in pairs), statistics.pstdev(g for g, _a in pairs),
            statistics.mean(a for _g, a in pairs))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data(n=300, seed=42)
    same = identical(ref)              # first: everything below reseeds the global RNG
    noise = {w: spread(ref, w) for w in (1, 2, 4, 8, 16)}
    runs = {w: [train(ref, data, w, s) for s in SEEDS] for w in (1, 4, 16)}
    return {"plain": train(ref, data, 0, 1)[0], "identical": same, "noise": noise,
            "gaps": {w: _stats(pairs) for w, pairs in runs.items()},
            "table": ", ".join(f"w{w} {n['mean']:.2f}/{n['sd']:.2f}" for w, n in noise.items())}


def verify(result):
    noise, gaps, plain = result["noise"], result["gaps"], result["plain"]
    drift = max(abs(noise[w]["mean"] / HIDDEN - 1) for w in noise)
    ratios = [abs(noise[w]["sd"] / noise[1]["sd"] / w ** 0.5 - 1) for w in (2, 4, 8, 16)]
    table = " | ".join(f"w{w} {gaps[w][0]:+.2f} ± {gaps[w][1]:.2f} pp" for w in (1, 4, 16))
    effect = abs(gaps[4][0] - gaps[1][0])
    return [
        practice.Check("CONTROL: at width 1 this is the lesson's Dropout, value for value",
                       result["identical"],
                       "same global RNG stream, same 1/(1-p) scaling — every width below is "
                       "the lesson's code regrouped, not a reimplementation"),
        practice.Check("MECHANISM: grouping holds the mean and scales the sd by sqrt(width)",
                       drift < 0.02 and max(ratios) < 0.06,
                       f"over 6000 masks the summed activation has mean/sd {result['table']} "
                       f"against a flat 32.00 (worst drift {drift * 100:.1f}%); every sd over "
                       f"sqrt(width) is {noise[1]['sd']:.2f} — 1/(1-p) stays per-channel, but "
                       f"C draws replace 32"),
        practice.Check("ANSWER: spatial dropout does not beat standard dropout",
                       effect < gaps[4][1] / 2 and gaps[1][0] < plain,
                       f"train-test gap over 8 mask seeds — {table}; no dropout at all "
                       f"{plain:+.2f} pp. Widths 1 and 4 differ by {effect:.2f} pp against a seed "
                       f"sd of {gaps[4][1]:.2f} pp — under half the noise the answer is read "
                       f"through, on 150 test points where one flipped label is 0.67 pp"),
        practice.Check("FINDING: only width 16 moves, and it underfits rather than regularizes",
                       gaps[16][0] < 0 and noise[16]["off"] > 0.05,
                       f"gap {gaps[16][0]:+.2f} pp — negative, test above train — at test "
                       f"accuracy {gaps[16][2]:.2f}%. With 2 channels both are off in "
                       f"{noise[16]['off'] * 100:.1f}% of passes (p^C = {P ** 2:.2%}), erasing "
                       f"the hidden layer that often and leaving the output on b2 alone"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
