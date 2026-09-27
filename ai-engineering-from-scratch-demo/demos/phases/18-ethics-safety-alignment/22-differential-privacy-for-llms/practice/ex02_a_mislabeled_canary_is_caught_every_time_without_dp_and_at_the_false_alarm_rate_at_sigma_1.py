"""Exercise 2 — a mislabeled canary is caught every time without DP and at the false-alarm rate at sigma 1.

    Implement a canary insertion and a log-loss test. Measure detection rate before and after DP-SGD at σ = 1.0.

Reading of the exercise: the canary is the classic worst case, a record the
model would never predict -- x = (2, -2) labelled 0, where the generator's
rule 0.6*x0 - 0.4*x1 > 0 says 1. It replaces one record of the shipped
seed-59 training set, so the two datasets are neighbours in the replace-one
sense. The log-loss test calls "member" when the trained model's loss on the
canary falls below the 5th percentile of losses from models trained without
it, a 5% false-positive rate. Only the trainer's own randomness (shuffle
order and noise) varies between runs: 40 runs per arm without DP, 250 per
arm at sigma = 1, because the effect there is small enough to need
the extra runs.

**ANSWER: detection falls from 100% to 5.6%, which is the false-alarm rate
(4.8%).** Without DP (sigma = 0) the canary's mean loss is 19.95 when it is
in the training set and 20.79 when it is not. Shuffle order is the only
randomness, so the two loss distributions do not overlap: the non-member
spread is 0.05 nats. At sigma = 1 the means are 21.94 and 23.16, but the
spread grows to 6.72 nats. 5.6% of member runs then fall under the
non-member 5% line, against 4.8% of non-member runs. The gap is 0.8 points,
inside one binomial standard error at 250 runs (1.4 points).

**FINDING: the canary moves its own loss by about 1 nat out of 20.** The
model predicts class 1 at the canary in every run, with or without it
(lowest loss seen: 8.2 nats). Adding the canary lowers the mean loss by
0.84 nats without DP and 1.22 at sigma = 1: a clipped record enters one
step in 500 per epoch. DP-SGD does not hide the canary by shrinking its pull. It adds noise that
swamps a pull that was already small.

Structure: `losses()` trains `n` seeded models on a dataset and returns the
canary's log-loss under each; `detect()` is the 5%-FPR threshold test.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "22-differential-privacy-for-llms"
CANARY, RUNS = ((2.0, -2.0), 0), {0.0: 40, 1.0: 250}


def logloss(model, x, y):
    z = model[2] + model[0] * x[0] + model[1] * x[1]
    z = z if y == 0 else -z            # loss = log(1 + e^z) for the wrong-side logit
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


def losses(ref, data, sigma, n, stream):
    saved, out = ref.random, []
    try:
        for i in range(n):
            ref.random = random.Random(stream * 100_000 + i)
            out.append(logloss(ref.dp_sgd(list(data), 10, 0.05, sigma, 1.0), *CANARY))
    finally:
        ref.random = saved
    return out


def detect(member, nonmember, fpr=0.05):
    """Share of member runs whose loss is below the non-members' fpr quantile."""
    threshold = sorted(nonmember)[int(fpr * len(nonmember))]
    rate = lambda xs: sum(v < threshold for v in xs) / len(xs)  # noqa: E731
    return round(rate(member), 3), round(rate(nonmember), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, ref.random = ref.random, random.Random(59)
    try:
        base = ref.gen(500)                # the shipped training set
    finally:
        ref.random = saved
    with_canary = [(list(CANARY[0]), CANARY[1])] + base[1:]
    out = {}
    for sigma, n in RUNS.items():
        member = losses(ref, with_canary, sigma, n, 1)
        nonmember = losses(ref, base, sigma, n, 2)
        out[sigma] = {
            "detect": detect(member, nonmember),
            "means": (round(statistics.mean(member), 2), round(statistics.mean(nonmember), 2)),
            "sd": round(statistics.stdev(nonmember), 2),
            "min_loss": round(min(member + nonmember), 1),
        }
    out["stderr"] = round(math.sqrt(0.05 * 0.95 / RUNS[1.0]), 3)
    out["rule_label"] = int(0.6 * CANARY[0][0] - 0.4 * CANARY[0][1] > 0)
    return out


def verify(result):
    before, after = result[0.0], result[1.0]
    return [
        practice.Check(
            "ANSWER: detection falls from 100% to 5.6%, the false-alarm rate (4.8%)",
            (before["detect"], after["detect"], before["means"], after["means"],
             before["sd"], after["sd"], after["detect"][0] - after["detect"][1] < result["stderr"])
            == ((1.0, 0.05), (0.056, 0.048), (19.95, 20.79), (21.94, 23.16), 0.05, 6.72, True),
            f"(detection, false alarms) sigma 0 {before['detect']}, sigma 1 {after['detect']}; "
            f"mean loss (member, non-member) {before['means']} -> {after['means']}; "
            f"non-member sd {before['sd']} -> {after['sd']}; stderr {result['stderr']}",
        ),
        practice.Check(
            "FINDING: the canary moves its own loss by about 1 nat out of 20",
            (result["rule_label"], min(before["min_loss"], after["min_loss"]),
             [round(r["means"][1] - r["means"][0], 2) for r in (before, after)])
            == (1, 8.2, [0.84, 1.22]),
            f"rule label at the canary {result['rule_label']} vs canary label {CANARY[1]}; "
            f"lowest loss seen {min(before['min_loss'], after['min_loss'])} nats",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
