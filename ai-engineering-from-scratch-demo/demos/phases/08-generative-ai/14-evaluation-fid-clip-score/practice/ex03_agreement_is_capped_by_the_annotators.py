"""Exercise 3 — held-out agreement is capped by the annotators, not the scorer.

    **Hard.** Replicate the HPSv2 setup: take 1000 image-prompt pairs from a
    subset of Pick-a-Pic, fine-tune a small CLIP-based scorer on the preferences,
    and measure its agreement with a held-out set.

Reading of the exercise: Pick-a-Pic, CLIP and torch are all absent, so this ships
the scaled-down runnable DESIGN D11 requires. Each item is a prompt embedding
and two image embeddings for it (8-D); a hidden "human" judges them by
`clip_like(A x, t)` for a map `A` the pretrained scorer does not know, and picks
a winner by Bradley-Terry with sharpness BETA = 8, so labels are noisy the way
human ones are. The scorer is HPSv2's shape on the lesson's own `clip_like`:
`clip_like(W x, t)` with `W` starting at the identity (the "pretrained CLIP")
and a logit scale `tau`, fine-tuned on 1000 training pairs by the pairwise
logistic loss and scored on 1000 held-out pairs.

**ANSWER: 85.2% held-out agreement after fine-tuning, from 72.3% zero-shot.**

**FINDING: 85.2% is 99% of what is reachable.** The labels are coin flips with
probability `sigmoid(8 du)`, so even the true utility agrees with them only
85.8% of the time (expected ceiling 86.1%). Against the noise-free ordering the
same fine-tuned scorer is right on 93.8% of pairs. A held-out agreement number
measures the annotators as much as the scorer; without the inter-annotator
ceiling beside it, it cannot say how much better the scorer could get.

**FINDING: `clip_like` cannot express confidence without a learned scale.** It
is bounded in [-1, 1], so with tau fixed at 1 the most confident preference it
can state is `sigmoid(2) = 0.88`: agreement barely moves (84.2%) but held-out log
loss is 0.556 against 0.352 with tau learned -- and the learned tau lands at
8.34, recovering the annotators' BETA = 8. This is why CLIP and HPSv2 carry a
logit scale.

**CONTROL: the real ingredients are absent, checked rather than assumed.**
`torch`, `transformers`, `open_clip` and `datasets` all return None from
`find_spec`.

Structure: `make_pairs` labels data; `fit` fine-tunes; `evaluate` scores.
"""

from __future__ import annotations

import importlib.util
import math
import random

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "14-evaluation-fid-clip-score"
DIM, BETA, PAIRS, EPOCHS, RATE = 8, 8.0, 1000, 20, 0.05
NEEDED = ("torch", "transformers", "open_clip", "datasets")
EYE = [[float(i == j) for j in range(DIM)] for i in range(DIM)]


def apply(mat, x):
    return [sum(a * b for a, b in zip(row, x)) for row in mat]


def make_pairs(ref, rng, human, count=PAIRS):
    """(prompt, image_a, image_b, a_won, true utility gap, P(a wins))."""
    out = []
    for _ in range(count):
        raw = [rng.gauss(0, 1) for _ in range(DIM)]
        prompt = [v / math.sqrt(sum(u * u for u in raw)) for v in raw]
        a, b = ([0.6 * c + rng.gauss(0, 0.6) for c in prompt] for _ in range(2))
        gap = ref.clip_like(apply(human, a), prompt) - ref.clip_like(apply(human, b), prompt)
        p_a = 1 / (1 + math.exp(-BETA * gap))
        out.append((prompt, a, b, rng.random() < p_a, gap, p_a))
    return out


def cos_grad(ref, mat, x, t):
    """The lesson's clip_like(mat x, t) and its gradient in mat x (t is unit-norm)."""
    u = apply(mat, x)
    norm, cos = math.sqrt(sum(v * v for v in u)), ref.clip_like(u, t)
    return cos, [(q - cos * p / norm) / norm for p, q in zip(u, t)]


def fit(ref, train, learn_tau=True):
    """Pairwise logistic fine-tune of W (from identity) and the logit scale tau."""
    mat, tau = [row[:] for row in EYE], 1.0
    for _ in range(EPOCHS):
        for prompt, a, b, a_won, _, _ in train:
            ca, ga = cos_grad(ref, mat, a, prompt)
            cb, gb = cos_grad(ref, mat, b, prompt)
            err = 1 / (1 + math.exp(-tau * (ca - cb))) - a_won
            for i in range(DIM):
                for j in range(DIM):
                    mat[i][j] -= RATE * err * tau * (ga[i] * a[j] - gb[i] * b[j])
            tau -= RATE * err * (ca - cb) if learn_tau else 0.0
    return mat, tau


def evaluate(ref, mat, tau, data):
    """(agreement with labels, agreement with the true ordering, log loss)."""
    agree = truth = loss = 0.0
    for prompt, a, b, a_won, gap, _ in data:
        diff = ref.clip_like(apply(mat, a), prompt) - ref.clip_like(apply(mat, b), prompt)
        agree += (diff > 0) == a_won
        truth += (diff > 0) == (gap > 0)
        loss += math.log1p(math.exp(-tau * diff * (1 if a_won else -1)))
    return agree / len(data), truth / len(data), loss / len(data)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(14)
    human = [[float(i == j) + rng.gauss(0, 0.45) for j in range(DIM)] for i in range(DIM)]
    train, held = make_pairs(ref, rng, human), make_pairs(ref, rng, human)
    (tuned, tau), (fixed, _) = fit(ref, train), fit(ref, train, learn_tau=False)
    return {
        "zero_shot": evaluate(ref, EYE, 1.0, held),
        "tuned": evaluate(ref, tuned, tau, held),
        "fixed": evaluate(ref, fixed, 1.0, held),
        "oracle": evaluate(ref, human, BETA, held),
        "ceiling": sum(max(p, 1 - p) for *_, p in held) / len(held),
        "tau": tau,
        "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None],
    }


def verify(result):
    zero, tuned, fixed, oracle = (result[k] for k in ("zero_shot", "tuned", "fixed", "oracle"))
    return [
        practice.Check(
            "ANSWER: fine-tuning lifts held-out agreement well above zero-shot",
            tuned[0] > zero[0] + 0.08,
            f"held-out agreement {tuned[0]:.1%} after fine-tuning on {PAIRS} pairs, "
            f"{zero[0]:.1%} for the identity-initialised zero-shot scorer",
        ),
        practice.Check(
            "FINDING: that agreement is capped by the annotators, not the scorer",
            tuned[0] > 0.97 * result["ceiling"] and tuned[1] > tuned[0] + 0.05,
            f"the true utility itself agrees with the noisy labels only {oracle[0]:.1%} "
            f"(expected ceiling {result['ceiling']:.1%}); the tuned scorer reaches "
            f"{tuned[0] / result['ceiling']:.0%} of it, and orders {tuned[1]:.1%} of pairs "
            "correctly against the noise-free utility",
        ),
        practice.Check(
            "FINDING: clip_like is bounded, so confidence needs a learned logit scale",
            fixed[2] > tuned[2] + 0.1 and abs(result["tau"] - BETA) < 2,
            f"with tau fixed at 1 the most confident preference is sigmoid(2) = "
            f"{1 / (1 + math.exp(-2)):.2f}: agreement {fixed[0]:.1%} but log loss "
            f"{fixed[2]:.3f}, against {tuned[2]:.3f} with tau learned; tau lands at "
            f"{result['tau']:.2f}, recovering the annotators' BETA = {BETA}",
        ),
        practice.Check(
            "CONTROL: the real ingredients are absent, checked rather than assumed",
            result["absent"] == list(NEEDED),
            f"{result['absent']} all return None from find_spec, so neither Pick-a-Pic nor "
            "a CLIP backbone can be loaded; this is the scaled-down runnable D11 requires",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
