"""Exercise 3 — DP-trained models recover 96.5% of training labels, and 25 copies of a canary are caught 92.5% of the time.

    Read Nasr et al. 2025 on training-data extraction. Why does extraction success not collapse under moderate ε? What does this imply about MIA-as-evaluation?

Reading of the exercise: the paper's argument is rebuilt as the smallest
runnable model on the lesson's own trainer, rather than paraphrased.
"Extraction" is recovering a training record's label from its features, the
toy analogue of completing a training prefix. "MIA" is exercise 2's log-loss
test at a 5% false-positive rate. The trainer is the lesson's DP setting,
sigma = 1, on the shipped seed-59 data. (Its epsilon is far from moderate,
which only makes the point stronger: even this much noise does not stop
extraction.)

**ANSWER: extraction does not collapse because DP bounds how much one record
changes the output, not how much of the data the output reveals.** At
sigma = 1 the model recovers 96.5% of training labels, and 96.9% of labels
for records it never saw (mean of 20 runs). Almost all of that "extraction"
is generalisation, which DP permits by design. The second reason is
repetition. DP protects one record, so k copies of a record get only group
privacy, which is roughly k times weaker. At the same sigma = 1, a
mislabeled canary inserted 5 times is detected in 31.2% of 80 runs, and
inserted 25 times in 92.5%. Web-scale training data repeats text, and repeated text is
what extraction attacks recover.

**FINDING: an MIA score measures the canary as much as the model.** With no
DP at all, the log-loss test catches an in-distribution canary (x = (0.3,
0.2), correct label) in 13.8% of 80 runs and a mislabeled outlier in 100%. A
low MIA number bounds only the canary that was tested. It says nothing about
the most extractable record, and it says nothing about extraction, which
succeeds here with no membership signal at all. The lesson makes the same
point: canaries "under-report" because they are not built to be extractable.
The implication is that MIA-as-evaluation needs worst-case canaries plus an
extraction test, not either one alone.

Structure: `train()` returns seeded models; `detect()` is exercise 2's
threshold test, restated because exercise files do not import each other.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "22-differential-privacy-for-llms"
OUTLIER, TYPICAL, RUNS = ((2.0, -2.0), 0), ((0.3, 0.2), 1), 80


def logloss(model, x, y):
    z = model[2] + model[0] * x[0] + model[1] * x[1]
    z = z if y == 0 else -z
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


def train(ref, data, sigma, n, stream):
    saved, out = ref.random, []
    try:
        for i in range(n):
            ref.random = random.Random(stream * 100_000 + i)
            out.append(ref.dp_sgd(list(data), 10, 0.05, sigma, 1.0))
    finally:
        ref.random = saved
    return out


def detect(member, nonmember, canary):
    """Share of member models under the non-members' 5% loss quantile."""
    threshold = sorted(logloss(m, *canary) for m in nonmember)[int(0.05 * len(nonmember))]
    return round(sum(logloss(m, *canary) < threshold for m in member) / len(member), 3)


def insert(base, canary, copies):
    return [(list(canary[0]), canary[1])] * copies + base[copies:]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, ref.random = ref.random, random.Random(59)
    try:
        base, test = ref.gen(500), ref.gen(200)      # the shipped train and test sets
    finally:
        ref.random = saved
    dp = train(ref, base, 1.0, 20, 3)
    clean, noisy = train(ref, base, 0.0, RUNS, 2), train(ref, base, 1.0, RUNS, 2)
    mean = lambda data: round(sum(ref.accuracy(m, data) for m in dp) / len(dp), 3)  # noqa: E731
    return {
        "extract": (mean(base), mean(test)),
        "copies": {k: detect(train(ref, insert(base, OUTLIER, k), 1.0, RUNS, 1), noisy, OUTLIER)
                   for k in (5, 25)},
        "design": {name: detect(train(ref, insert(base, c, 1), 0.0, RUNS, 1), clean, c)
                   for name, c in (("typical", TYPICAL), ("outlier", OUTLIER))},
        "doc_says": "under-report" in parity.doc_text(PHASE, LESSON, "en"),
    }


def verify(result):
    train_acc, test_acc = result["extract"]
    return [
        practice.Check(
            "ANSWER: DP bounds one record's influence, not what the output reveals",
            result["extract"] == (0.965, 0.969) and abs(train_acc - test_acc) < 0.01
            and result["copies"] == {5: 0.312, 25: 0.925},
            f"sigma 1 label recovery train {train_acc:.1%} vs unseen {test_acc:.1%}; "
            f"canary detection by copies {result['copies']}",
        ),
        practice.Check(
            "FINDING: an MIA score measures the canary as much as the model",
            result["design"] == {"typical": 0.138, "outlier": 1.0} and result["doc_says"],
            f"no-DP detection by canary design {result['design']}; lesson says canaries "
            f"under-report: {result['doc_says']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
