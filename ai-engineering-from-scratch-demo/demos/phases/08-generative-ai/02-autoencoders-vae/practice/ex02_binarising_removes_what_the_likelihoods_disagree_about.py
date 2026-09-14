"""Exercise 2 — binarising the data removes the thing the two likelihoods disagree about.

    **Medium.** Replace the Gaussian decoder likelihood with a Bernoulli
    likelihood (cross-entropy loss). Compare sample quality on a binarized
    version of the same synthetic data.

Reading of the exercise: only the decoder likelihood changes. Both arms run the
lesson's own `init_vae`, `forward` and `backward` at the lesson's own settings,
and the Bernoulli arm reaches the lesson's `backward` unmodified by handing it
the target that makes its `d recon / d x_hat = 2(x_hat - t)` equal the Bernoulli
gradient `sigmoid(x_hat) - x`; the CONTROL below checks that identity holds to
1e-12 rather than asserting it. "Sample quality" is scored by decoding draws
from the `N(0, I)` prior and asking how many land on one of the two patterns the
binarised data actually contains.

**ANSWER: the same, to inside the measurement.** Over SEEDS seeds and DRAWS
prior draws each, valid samples come to **0.979** for the Gaussian decoder and
**0.987** for the Bernoulli one. That **0.009** gap is about one standard error
of a 400-draw proportion (**0.007**), so this experiment cannot tell the two
apart on the axis the exercise names.

**FINDING: what the swap changes is the support, not the quality.** The Gaussian
decoder is unbounded, and **38.6%** of prior draws come back with at least one
coordinate outside `[0, 1]` -- the worst excursion averages **+0.143** across
seeds, a probability of 1.14 for a bit. The Bernoulli decoder cannot do this:
its output is a sigmoid.
Both models then threshold to the same patterns, which is exactly why the quality
numbers agree and the honest difference does not show up on that axis at all.

**FINDING: the exercise's own binarisation destroys its comparison.** Two
likelihoods differ in how they model the *residual* around the mean. Binarising
`sample_mixture` leaves **2 distinct patterns in 400 rows, with 0 deviations**:
the centres are at +-1 and the noise is `gauss(0, 0.2)`, so a sign flip is a 5
sigma event and never happens in a sample this size. There is no residual left
to model, so the choice of residual model cannot matter. To make this comparison
informative the noise would have to survive binarisation -- centres nearer 0, or
a genuine Bernoulli sampling step.

Structure: `train` runs either likelihood through the lesson's own backward;
`measure` scores prior draws; `bernoulli_target` is the one-line substitution the
whole exercise comes down to.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "02-autoencoders-vae"
IN_DIM, HIDDEN, Z_DIM, POINTS = 8, 10, 2, 60
EPOCHS, RATE, BETA, SEEDS, DRAWS = 40, 0.01, 0.2, 5, 400
VALID = {(1.0,) * 4 + (0.0,) * 4, (0.0,) * 4 + (1.0,) * 4}


def sigmoid(v):
    """Saturating logistic, so a confident logit cannot overflow math.exp."""
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v))))


def binarise(rows):
    """The exercise's 'binarized version of the same synthetic data': the sign bit."""
    return [[1.0 if v > 0 else 0.0 for v in row] for row in rows]


def bernoulli_target(x_hat, x):
    """The t for which the lesson's 2(x_hat - t) equals the Bernoulli sigmoid(x_hat) - x."""
    return [xh - (sigmoid(xh) - xi) / 2 for xh, xi in zip(x_hat, x)]


def train(ref, random, bernoulli, seed):
    """The lesson's loop, with only the decoder likelihood swapped."""
    rng = random.Random(seed)
    params = ref.init_vae(IN_DIM, HIDDEN, Z_DIM, rng)
    data = binarise(ref.sample_mixture(POINTS, IN_DIM, rng))
    for _ in range(EPOCHS):
        for x in data:
            fwd = ref.forward(x, params, [rng.gauss(0, 1) for _ in range(Z_DIM)])
            target = bernoulli_target(fwd["x_hat"], x) if bernoulli else x
            ref.apply_update(params, ref.backward(target, fwd, params, BETA), RATE)
    return params


def decode(ref, params, z):
    """The lesson's own decoder path, as main() runs it when sampling from the prior."""
    dec = params["dec"]
    hidden = ref.tanh(ref.add(ref.matmul(dec["W1"], z), dec["b1"]))
    return ref.add(ref.matmul(dec["W_out"], hidden), dec["b_out"])


def score(out):
    """(lands on a real pattern, leaves [0, 1], by how much) for one decoded sample."""
    excursion = max(max(v - 1.0, -v) for v in out)
    return tuple(1.0 if v > 0.5 else 0.0 for v in out) in VALID, excursion > 0.0, excursion


def measure(ref, random, params, bernoulli, draws=DRAWS, seed=99):
    """(valid fraction, fraction leaving [0, 1], worst excursion) over prior draws."""
    rng, good, outside, worst = random.Random(seed), 0, 0, 0.0
    for _ in range(draws):
        raw = decode(ref, params, [rng.gauss(0, 1) for _ in range(Z_DIM)])
        valid, left, excursion = score([sigmoid(v) for v in raw] if bernoulli else raw)
        good, outside, worst = good + valid, outside + left, max(worst, excursion)
    return good / draws, outside / draws, worst


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = {}
    for name, bernoulli in (("gaussian", False), ("bernoulli", True)):
        rows = [measure(ref, random, train(ref, random, bernoulli, s), bernoulli)
                for s in range(SEEDS)]
        arms[name] = tuple(statistics.fmean(col) for col in zip(*rows))
    sample = binarise(ref.sample_mixture(400, IN_DIM, random.Random(7)))
    probe = train(ref, random, True, 0)
    fwd = ref.forward(sample[0], probe, [0.0] * Z_DIM)
    target = bernoulli_target(fwd["x_hat"], sample[0])
    return {
        "arms": arms, "patterns": len({tuple(r) for r in sample}),
        "deviations": sum(tuple(r) not in VALID for r in sample), "rows": len(sample),
        "identity": max(abs(2 * (xh - t) - (sigmoid(xh) - xi))
                        for xh, t, xi in zip(fwd["x_hat"], target, sample[0])),
    }


def verify(result):
    gauss, bern = result["arms"]["gaussian"], result["arms"]["bernoulli"]
    return [
        practice.Check(
            "ANSWER: sample quality is the same, to inside the measurement",
            abs(gauss[0] - bern[0]) < 0.05,
            f"over {SEEDS} seeds and {DRAWS} draws each, decoded samples land on one of the two "
            f"patterns the data contains {gauss[0]:.3f} of the time for the Gaussian decoder and "
            f"{bern[0]:.3f} for the Bernoulli one: a gap of {abs(gauss[0] - bern[0]):.3f} against "
            f"one standard error of {math.sqrt(0.98 * 0.02 / DRAWS):.3f}, so the axis the exercise "
            "names cannot separate them",
        ),
        practice.Check(
            "FINDING: the swap changes the support, not the quality",
            gauss[1] > 0.2 and bern[1] == 0.0,
            f"{gauss[1]:.1%} of draws leave the Gaussian decoder with a coordinate outside "
            f"[0, 1], the worst by {gauss[2]:+.3f} -- a probability of {1 + gauss[2]:.3f} for a "
            f"bit. The Bernoulli decoder does this {bern[1]:.1%} of the time, its output being a "
            "sigmoid. Both threshold to the same patterns, which is why the quality numbers "
            "agree: the difference is real and it is not on that axis",
        ),
        practice.Check(
            "FINDING: binarising the data destroys the comparison it is asked to serve",
            result["patterns"] == 2 and result["deviations"] == 0,
            f"two likelihoods differ in how they model the residual around the mean, and "
            f"binarising leaves {result['patterns']} patterns in {result['rows']} rows with "
            f"{result['deviations']} deviations -- centres at +-1 against gauss(0, 0.2) make a "
            "sign flip a 5-sigma event. No residual survives, so no model of one can matter",
        ),
        practice.Check(
            "CONTROL: the Bernoulli arm reuses the lesson's own backward, exactly",
            result["identity"] < 1e-12,
            f"the swap is one substitution: hand ref.backward the target t for which its "
            f"2(x_hat - t) equals sigmoid(x_hat) - x. The two differ by {result['identity']:.1e}, "
            "so this is the lesson's own backward with a different target rather than a second "
            "implementation that could quietly disagree with it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
