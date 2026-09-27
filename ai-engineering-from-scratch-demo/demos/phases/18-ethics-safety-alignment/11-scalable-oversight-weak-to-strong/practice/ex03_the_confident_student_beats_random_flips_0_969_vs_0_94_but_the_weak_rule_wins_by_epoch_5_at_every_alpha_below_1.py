"""Exercise 3 — the confident student beats random flips (0.969 vs 0.94), but the weak rule wins by epoch 5 at every alpha below 1.

    Read Burns et al. 2023 Section 4.3 (NLP tasks). Reproduce the "confidence
    auxiliary loss" intuition: when the strong model is more confident than
    the weak labels, who wins?

Reading of the exercise: the loss mixes two targets, (1 - alpha) * weak label
+ alpha * the student's own hardened prediction, so on an example where they
disagree the student's side wins the target only when alpha > 0.5. "Who wins"
is then measured as accuracy after fine-tuning, and as the share of
prior-vs-weak disagreements where the student keeps its own answer. The
student needs something to be confident *about*. The reference strong model
starts from zero weights, so here it starts from `train_strong` on 20 gold
labels, a stand-in for pre-trained knowledge (0.94 accurate). It is then
fine-tuned with `train_strong`'s own SGD (reproduced bit for bit at alpha 0)
on 1000 weak labels of two kinds: the reference `weak_label(x, 0.7)`, whose
errors follow the rule `x[0] > 0`, and gold labels with 30% flipped at
random. Four seeds;
"confident" means |logit| > 2.

**ANSWER: the confident student wins against random weak errors and loses to
systematic ones.** Accuracy after fine-tuning, against a 0.940 prior:

| weak labels | epochs | alpha 0 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| random flips | 2 | 0.900 | 0.969 | 0.962 | 0.930 |
| random flips | 5 | 0.919 | 0.958 | 0.945 | 0.920 |
| reference x0 rule | 2 | 0.794 | 0.777 | 0.877 | 0.930 |
| reference x0 rule | 5 | 0.806 | 0.786 | 0.780 | 0.920 |

With random flips, the confidence loss is the only setting in which
fine-tuning on weak labels beats the prior (0.969). With the reference's
rule, alpha 0.75 holds off the weak labels for 2 epochs (0.877 vs 0.794). By
5 epochs it is back to 0.780, below plain fine-tuning. Alpha 1.0 ignores the
labels entirely, which is self-training, and it still drifts down from 0.940.

**FINDING: confidence decides which disagreements the student keeps.** At
alpha 0 and 2 epochs, the student keeps its answer on 69.5% of the confident
disagreements against 45.8% of the unconfident ones (93.0% vs 57.6% against
random flips). At alpha 0.75 against flips, it keeps 99.5% of the confident
ones.

**FINDING: a consistent weak error wins in the end, because the student can
represent it.** Each disagreement point that crosses the boundary flips its
hardened target to agree with the weak label. With a linear student and a
linear weak rule, this feedback runs one way. Alpha 0.5 is not enough even at
the start (0.777 at 2 epochs, worse than alpha 0).

**FINDING: the lesson's simulator cannot show any of this.** `train_strong`
starts at `w = [0.0, 0.0, 0.0]`, so there is no prior to be confident with.
The TAKEAWAY's "using its own pre-trained priors" names a component the code
does not have.

Structure: `train()` is `train_strong` plus a starting point and alpha;
`one_seed()` builds the prior and both label sets; `kept()` measures who won
on the disagreement points.
"""

from __future__ import annotations

import contextlib
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "11-scalable-oversight-weak-to-strong"
SEEDS, ALPHAS, EPOCHS, SURE = range(4), (0.0, 0.5, 0.75, 1.0), (2, 5), 2.0


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def train(data, rng, init=None, alpha=0.0, steps=200, lr=0.05):
    """`train_strong` from `init`, target (1 - alpha) * label + alpha * own hard prediction."""
    w, b = (list(init[:3]), init[3]) if init else ([0.0, 0.0, 0.0], 0.0)
    for _ in range(steps):
        rng.shuffle(data)
        for x, y in data:
            z = b + sum(wi * xi for wi, xi in zip(w, x))
            p = 1.0 / (1.0 + pow(2.71828, -z))
            err = p - ((1 - alpha) * y + alpha * (z > 0))
            w = [wi - lr * err * xi for wi, xi in zip(w, x)]
            b -= lr * err
    return w + [b]


def parity_check(ref):
    """alpha = 0 from zero weights is `train_strong` itself, bit for bit."""
    with seeded(ref, 5):
        data, rng = ref.gen(200), random.Random()
        rng.setstate(ref.random.getstate())
        theirs = ref.train_strong(list(data), steps=20)
    return train(list(data), rng, steps=20) == theirs


def logit(model, x):
    return model[3] + sum(wi * xi for wi, xi in zip(model[:3], x))


def kept(prior, model, points):
    """Share of `points` where the fine-tuned model still sides with the prior."""
    return sum((logit(model, x) > 0) == (logit(prior, x) > 0) for x in points) / len(points)


def disagreements(prior, weak):
    """Training points where prior and weak label disagree: (confident, unconfident)."""
    against = [x for x, y in weak if (logit(prior, x) > 0) != y]
    return ([x for x in against if abs(logit(prior, x)) > SURE],
            [x for x in against if abs(logit(prior, x)) <= SURE])


def one_seed(ref, seed, noise):
    """Prior from 20 gold labels, fine-tuned on 1000 weak labels at every (epochs, alpha)."""
    with seeded(ref, seed):
        ev, tr, gold20 = ref.gen(1000), ref.gen(1000), ref.gen(20)
        prior, rng = ref.train_strong(list(gold20)), random.Random(7 + seed)
        flip = (lambda x, y: 1 - y if rng.random() < 0.3 else y) if noise else (
            lambda x, y: ref.weak_label(x, 0.7))
        weak = [(x, flip(x, y)) for x, y in tr]
    sure, unsure = disagreements(prior, weak)
    models = {(e, a): train(list(weak), random.Random(seed), prior, a, steps=e)
              for e in EPOCHS for a in ALPHAS}
    return ({k: (ref.accuracy(m, ev), kept(prior, m, sure), kept(prior, m, unsure))
             for k, m in models.items()}, ref.accuracy(prior, ev))


def table(ref, noise):
    runs = [one_seed(ref, s, noise) for s in SEEDS]
    mean = {k: tuple(round(sum(r[0][k][i] for r in runs) / len(runs), 3) for i in range(3))
            for k in runs[0][0]}
    return mean, [v[0] for v in mean.values()], round(sum(r[1] for r in runs) / len(runs), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"parity": parity_check(ref), "rule": table(ref, False), "flips": table(ref, True),
            "zero_start": "w = [0.0, 0.0, 0.0]" in inspect.getsource(ref.train_strong),
            "takeaway": "pre-trained priors" in inspect.getsource(ref.main)}


def verify(result):
    (rule, rule_acc, prior), (flips, flip_acc, prior2) = result["rule"], result["flips"]
    return [
        practice.Check(
            "ANSWER: the confident student wins against random weak errors, loses to systematic",
            all([result["parity"], prior == prior2 == 0.94,
                 flip_acc == [0.9, 0.969, 0.962, 0.93, 0.919, 0.958, 0.945, 0.92],
                 rule_acc == [0.794, 0.777, 0.877, 0.93, 0.806, 0.786, 0.78, 0.92]]),
            f"prior {prior}; (epochs 2, 5) x alpha {ALPHAS}: flips {flip_acc}, rule {rule_acc}",
        ),
        practice.Check(
            "FINDING: confidence decides which disagreements the student keeps",
            (rule[2, 0.0][1:], flips[2, 0.0][1:], flips[2, 0.75][1]) == ((0.695, 0.458), (0.93, 0.576), 0.995),
            f"(sure, unsure) kept, alpha 0: rule {rule[2, 0.0][1:]}, flips {flips[2, 0.0][1:]}",
        ),
        practice.Check(
            "FINDING: a consistent weak error wins in the end, because the student can express it",
            all([max(rule_acc[4:7]) < prior, rule[2, 0.5][0] < rule[2, 0.0][0]]),
            f"rule labels after 5 epochs, alpha < 1: {rule_acc[4:7]}",
        ),
        practice.Check(
            "FINDING: the lesson's simulator cannot show any of this",
            result["zero_start"] and result["takeaway"], "zero start; main() claims priors"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
