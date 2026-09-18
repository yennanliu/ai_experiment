"""Exercise 2 — the bias is the class prior in log-odds.

    SigLIP uses a bias parameter `b` in addition to temperature: `S'[i,j] =
    S[i,j]/tau + b`. What role does `b` play when the batch has a large class
    imbalance (many more negatives than positives per row)? Read SigLIP Section
    3 (arXiv:2303.15343).

Reading of the exercise: "what role does b play" is answered by measuring the
b that minimises the lesson's own `sigmoid_loss` at each batch size, rather
than by restating the paper, because the role is a number and the number is
predictable. A batch of N has N positives and N^2 - N negatives, so the
positive rate is 1/N and the log-odds of that rate is -log(N - 1).

**ANSWER: b is the positive rate in log-odds.** Across N = 4, 8, 16, 32, 64 the
measured argmin tracks **-log(N - 1)** to within **0.051** -- and that is the
bias with no training at all, just the prior the batch shape implies.

**FINDING: without b the loss cannot see the imbalance.** At b = 0 the loss is
0.696-0.711 for every N, within 3% of log 2, because at initialisation every
pair sits at sigmoid(0) = 0.5 and pays the same. The positive-to-negative ratio
moves from 1:3 to 1:63 across the sweep and the loss does not move at all; the
negatives, which are 98.4% of the pairs at N = 64, get the same gradient as
the positives.

**FINDING: SigLIP's published b = -10 is the logit of their batch.** At their
32,768 batch, -log(32,767) = **-10.397**. The initialisation in the paper is
the prior of the batch size they trained at, to within 0.4 nats.

**FINDING: tau and b are not independent.** Sharpening tau from 1.0 to 0.07
moves the optimal bias a further 1.2-1.4 nats negative at the same N, because
dividing by tau widens the similarity distribution. A learned temperature drags
the optimal bias with it, which is why SigLIP learns both.

Structure: `batch` builds a random (untrained) similarity matrix from the
lesson's own embedding maker, `best_bias` is a coarse-then-fine minimisation of
the lesson's `sigmoid_loss` over b, and `CASES` is the (N, tau) sweep.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "02-clip-contrastive-pretraining"
SIZES = (4, 8, 16, 32, 64)
SHARP = (4, 16, 64)
SIGLIP_BATCH, SIGLIP_BIAS = 32768, -10.0
COARSE, FINE = 0.1, 0.01


def batch(ref, size, tau, seed=0):
    """An untrained batch: independent image and text embeddings, so S is near 0."""
    images = [ref.make_fake_embedding(seed + i) for i in range(size)]
    texts = [ref.make_fake_embedding(seed + 1000 + i) for i in range(size)]
    return ref.similarity_matrix(images, texts, tau)


def _scan(loss, low, high, step):
    points = [low + step * k for k in range(int((high - low) / step) + 1)]
    return min(points, key=loss)


def best_bias(ref, matrix):
    """The b minimising the lesson's sigmoid_loss, to 0.01, by coarse then fine scan."""
    def loss(bias):
        return ref.sigmoid_loss(matrix, bias)

    coarse = _scan(loss, -14.0, 4.0, COARSE)
    return round(_scan(loss, coarse - COARSE, coarse + COARSE, FINE), 2)


def optima(ref, matrices):
    return {size: best_bias(ref, matrix) for size, matrix in matrices.items()}


def losses(ref, matrices, biases):
    return {size: round(ref.sigmoid_loss(matrix, biases[size]), 4)
            for size, matrix in matrices.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    flat = {size: batch(ref, size, 1.0) for size in SIZES}
    sharp = {size: batch(ref, size, 0.07) for size in SHARP}
    measured, sharpened = optima(ref, flat), optima(ref, sharp)
    predicted = {size: -math.log(size - 1) for size in SIZES}
    zero = dict.fromkeys(SIZES, 0.0)
    return {
        "measured": measured, "predicted": {k: round(v, 3) for k, v in predicted.items()},
        "worst_gap": round(max(abs(measured[n] - predicted[n]) for n in SIZES), 3),
        "unbiased": losses(ref, flat, zero), "log2": math.log(2),
        "at_optimum": losses(ref, flat, measured),
        "negative_share": round((SIZES[-1] ** 2 - SIZES[-1]) / SIZES[-1] ** 2 * 100, 1),
        "siglip_logit": round(-math.log(SIGLIP_BATCH - 1), 3),
        "siglip_gap": round(abs(-math.log(SIGLIP_BATCH - 1) - SIGLIP_BIAS), 3),
        "sharpened": sharpened,
        "shift": {size: round(measured[size] - sharpened[size], 2) for size in SHARP},
    }


def verify(result):
    measured, unbiased, optimum = result["measured"], result["unbiased"], result["at_optimum"]
    return [
        practice.Check(
            "ANSWER: b is the positive rate in log-odds -- argmin tracks -log(N-1) to 0.051",
            all([result["worst_gap"] <= 0.051, measured[4] == -1.15, measured[64] == -4.15,
                 len(measured) == len(SIZES)]),
            f"measured {measured} against -log(N-1) {result['predicted']} -- worst gap "
            f"{result['worst_gap']}. A batch of N has N positives and N^2-N negatives, so "
            "the prior is 1/N and its logit is -log(N-1); the optimum is that prior, before "
            "any training",
        ),
        practice.Check(
            "FINDING: without b the loss cannot see the imbalance",
            all([all(abs(value / math.log(2) - 1) <= 0.03 for value in unbiased.values()),
                 max(unbiased.values()) - min(unbiased.values()) <= 0.02,
                 result["negative_share"] == 98.4]),
            f"at b=0 the loss is {unbiased} for N from {SIZES[0]} to {SIZES[-1]} -- all "
            f"within 3% of log 2 = {result['log2']:.4f}, because every pair sits at "
            f"sigmoid(0) = 0.5. The ratio moves from 1:3 to 1:{SIZES[-1] - 1} and the loss "
            f"does not; the negatives are {result['negative_share']}% of the pairs at "
            f"N={SIZES[-1]} and get the same gradient as the positives",
        ),
        practice.Check(
            "FINDING: SigLIP's published b = -10 is the logit of their batch",
            all([result["siglip_logit"] == -10.397, result["siglip_gap"] <= 0.4]),
            f"at a {SIGLIP_BATCH:,} batch the positive rate is 1/{SIGLIP_BATCH:,} and its "
            f"logit is {result['siglip_logit']}; the paper initialises b at {SIGLIP_BIAS:g}, "
            f"{result['siglip_gap']} nats away. And moving b to its optimum cuts the loss "
            f"from {unbiased[64]} to {optimum[64]} at N=64 -- "
            f"{(1 - optimum[64] / unbiased[64]) * 100:.0f}% -- without changing an embedding",
        ),
        practice.Check(
            "FINDING: tau and b are not independent",
            all([all(1.1 <= shift <= 1.5 for shift in result["shift"].values()),
                 len(result["sharpened"]) == len(SHARP)]),
            f"sharpening tau from 1.0 to 0.07 moves the optimum from "
            f"{ {n: measured[n] for n in SHARP} } to {result['sharpened']} -- a further "
            f"{result['shift']} nats negative -- because dividing by tau widens the "
            "similarity distribution. A learned temperature drags the optimal bias with it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
