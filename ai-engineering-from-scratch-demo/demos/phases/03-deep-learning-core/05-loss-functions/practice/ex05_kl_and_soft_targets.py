"""Exercise 5 — KL divergence, its identity with cross-entropy, and soft targets.

    Implement KL divergence loss and verify that minimizing KL(true || predicted)
    gives the same gradients as cross-entropy when the true distribution is
    one-hot. Then try soft targets (like knowledge distillation) where the "true"
    distribution comes from a teacher model's softmax output.

Reading of the exercise: "the same gradients" is checked against the lesson's own
`cce_gradient` by central differences through its own `softmax`, over random logits rather
than one hand-picked vector. The soft-target half is where the identity stops holding, so
checks 3-5 measure exactly what breaks: the loss moves and the gradient does not, the
temperature the technique needs is not in the formula, and KL is not symmetric.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "05-loss-functions"
CLASSES, TRIALS, EPS, H = 5, 200, 1e-15, 1e-6
TEMPS = (1.0, 2.0, 4.0)


def kl(ref, logits, target, temp=1.0) -> float:
    """KL(target || softmax(logits / temp)) — the exercise's loss, in nats."""
    probs = ref.softmax([z / temp for z in logits])
    return sum(q * math.log(max(EPS, q) / max(EPS, p)) for q, p in zip(target, probs))


def numeric(ref, logits, target, temp=1.0) -> list:
    """d(KL)/d(logit) by central difference through the lesson's own softmax."""
    out = []
    for i in range(len(logits)):
        hi, lo = list(logits), list(logits)
        hi[i], lo[i] = hi[i] + H, lo[i] - H
        out.append((kl(ref, hi, target, temp) - kl(ref, lo, target, temp)) / (2 * H))
    return out


def onehot(index) -> list:
    return [1.0 if i == index else 0.0 for i in range(CLASSES)]


def teacher(ref, rng, temp=1.0) -> list:
    return ref.softmax([rng.gauss(0, 2.0) / temp for _ in range(CLASSES)])


def worst(left, right) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def hard_case(ref, rng) -> dict:
    """One-hot targets: KL against the lesson's own cross-entropy, value and gradient."""
    logits, index = [rng.gauss(0, 2.0) for _ in range(CLASSES)], rng.randrange(CLASSES)
    target = onehot(index)
    return {"value": abs(kl(ref, logits, target) - ref.categorical_cross_entropy(logits, index)),
            "grad": worst(numeric(ref, logits, target), ref.cce_gradient(logits, index)),
            "form": worst(numeric(ref, logits, target),
                          [p - q for p, q in zip(ref.softmax(logits), target)])}


def soft_case(ref, rng) -> dict:
    """Soft targets: the same gradient under two teachers of different entropy."""
    logits = [rng.gauss(0, 2.0) for _ in range(CLASSES)]
    sharp, blunt = teacher(ref, rng, 0.5), teacher(ref, rng, 4.0)
    entropy = [-sum(q * math.log(max(EPS, q)) for q in t) for t in (sharp, blunt)]
    gap = _cross(ref, logits, sharp) - kl(ref, logits, sharp)
    return {"gap": gap, "off": abs(gap - entropy[0]), "entropy": entropy[0],
            "blunt_entropy": entropy[1], "temps": _temps(ref, logits, sharp),
            "grad_form": worst(numeric(ref, logits, sharp),
                               [p - q for p, q in zip(ref.softmax(logits), sharp)]),
            "self": kl(ref, [math.log(max(EPS, q)) for q in sharp], sharp),
            "asym": (kl(ref, logits, sharp), _reverse(ref, logits, sharp)),
            "scaled": [max(map(abs, numeric(ref, logits, sharp, t))) for t in TEMPS]}


def _temps(ref, logits, target) -> float:
    """Worst deviation from d/dz KL(q || softmax(z/T)) = (softmax(z/T) - q) / T."""
    return max(worst(numeric(ref, logits, target, t),
                     [(p - q) / t for p, q in zip(ref.softmax([z / t for z in logits]), target)])
               for t in TEMPS)


def _cross(ref, logits, target) -> float:
    return -sum(q * math.log(max(EPS, p)) for q, p in zip(target, ref.softmax(logits)))


def _reverse(ref, logits, target) -> float:
    probs = ref.softmax(logits)
    return sum(p * math.log(max(EPS, p) / max(EPS, q)) for p, q in zip(probs, target))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(11)
    hard = [hard_case(ref, rng) for _ in range(TRIALS)]
    soft = [soft_case(ref, rng) for _ in range(TRIALS)]
    return {"hard": {k: max(c[k] for c in hard) for k in hard[0]},
            "one": soft[0], "scaled": soft[0]["scaled"], **peaks(soft)}


def peaks(soft) -> dict:
    """The worst case over the soft-target trials, one number per claim."""
    return {"soft_grad": max(c["grad_form"] for c in soft),
            "off": max(c["off"] for c in soft), "temps": max(c["temps"] for c in soft),
            "self": max(abs(c["self"]) for c in soft),
            "asym": max(abs(c["asym"][0] - c["asym"][1]) for c in soft)}


def verify(result):
    hard, one = result["hard"], result["one"]
    return [
        practice.Check("ANSWER: with a one-hot target KL is cross-entropy, value and gradient",
                       hard["value"] < 1e-12 and hard["grad"] < 1e-6,
                       f"over {TRIALS} random 5-class logit vectors the largest gap between "
                       f"KL(onehot || softmax) and the lesson's `categorical_cross_entropy` is "
                       f"{hard['value']:.1e}, and between a central difference of KL and its own "
                       f"`cce_gradient` is {hard['grad']:.1e} — the difference quotient's own O(h^2)"),
        practice.Check("MECHANISM: KL(q||p) = CE(q,p) - H(q), and one-hot has H(q) = 0",
                       hard["form"] < 1e-6 and result["soft_grad"] < 1e-6,
                       f"the gradient is p - q under both targets — worst deviation "
                       f"{hard['form']:.1e} one-hot and {result['soft_grad']:.1e} soft — because "
                       f"H(q) does not depend on the logits. So KL and cross-entropy are the same "
                       f"objective plus a constant, and only one-hot makes that constant zero"),
        practice.Check("FINDING: with soft targets the loss moves and the gradient does not, so "
                       "KL values are not comparable across teachers",
                       result["off"] < 1e-12 and one["gap"] > 0.05,
                       f"cross-entropy exceeds KL by {one['gap']:.4f} nats here, which is exactly "
                       f"the teacher's entropy H(q) = {one['entropy']:.4f} — worst mismatch over "
                       f"{TRIALS} teachers {result['off']:.1e}. A blunter "
                       f"teacher (H = {one['blunt_entropy']:.4f}) reports a smaller KL for the same "
                       f"student, so a distillation run cannot be compared to another by its loss "
                       f"— it can only be compared to itself"),
        practice.Check("MECHANISM: the temperature the technique needs is not in this formula",
                       result["temps"] < 1e-6,
                       f"d/dz KL(q || softmax(z/T)) is exactly (softmax(z/T) - q)/T — worst "
                       f"deviation {result['temps']:.1e} over {TRIALS} teachers at T = {TEMPS}, "
                       f"with largest |dKL/dlogit| "
                       + ", ".join(f"{v:.4f}" for v in result["scaled"])
                       + ". The 1/T is why distillation multiplies the loss by T^2; the exercise's "
                       "'teacher's softmax output' leaves T at 1, so a student trained this way "
                       "never sees the flattened targets the technique is named for"),
        practice.Check("CONTROL: KL is zero at a perfect match and is not symmetric",
                       result["self"] < 1e-9 and result["asym"] > 0.1,
                       f"feeding the teacher's own log-probabilities back as logits gives KL = "
                       f"{result['self']:.1e}; swapping the arguments moves it by "
                       f"{result['asym']:.4f} nats at worst, {one['asym'][0]:.4f} against "
                       f"{one['asym'][1]:.4f} here. KL(true || predicted) is the one the exercise "
                       f"names, and it is the one that penalises a student for missing mass the "
                       f"teacher put somewhere"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
