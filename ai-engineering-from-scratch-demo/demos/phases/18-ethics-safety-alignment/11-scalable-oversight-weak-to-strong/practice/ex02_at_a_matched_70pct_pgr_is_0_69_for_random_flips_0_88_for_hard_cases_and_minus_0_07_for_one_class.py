"""Exercise 2 — at a matched 70%, PGR is 0.69 for random flips, 0.88 for hard cases and -0.07 for one class.

    Modify the weak labeler to have structured error (e.g., always wrong on a
    specific input class). Does PGR increase, decrease, or stay the same?
    Explain.

Reading of the exercise: "increase or decrease" needs a baseline at the same
accuracy, or the comparison only measures how many labels are wrong. So every
labeler here is wrong on 30% of inputs and differs only in *where*. The
baseline flips gold labels at random. The four structured labelers are always
wrong on a fixed input region: the hard cases nearest the gold boundary, a
slab x2 > 0.524 unrelated to the boundary, one input class (positives with
x0 < 0.76, which are always labelled 0), and the lesson's own `weak_label`
with its knob set to 0.931 so it is also 70% right. Strong-on-weak is the
reference `train_strong`, scored with the reference `accuracy`; three seeds.

**ANSWER: it depends on where the errors sit, and three of four structures
lower PGR.** At weak accuracy 0.698 to 0.711:

| weak labeler | strong-on-weak | PGR | weak errors strong fixes |
|---|---:|---:|---:|
| random flips | 0.907 | 0.689 | 91.3% |
| hard cases (near the boundary) | 0.961 | 0.876 | 86.8% |
| x2 slab | 0.756 | 0.154 | 58.9% |
| one class, always wrong | 0.678 | -0.066 | 4.0% |
| reference x0 rule | 0.732 | 0.074 | 18.9% |

**FINDING: the strong model copies an error that is consistent with an input
region.** Random flips carry no pattern, so the fit averages them away and
fixes 91% of them. An always-wrong class is a pattern that a linear model can
partly represent, so it learns it. It fixes 4% of those errors and ends up
worse than its supervisor (PGR -0.066). The lesson's own labeler behaves the
same way: its errors are the rule `x[0] > 0`, and the strong model fixes 19%.

**FINDING: errors on hard cases raise PGR above the random baseline.** Wrong
labels that straddle the boundary evenly on both sides do not move the best
linear separator. With the other 70% of labels clean, strong-on-weak reaches
0.961 and PGR 0.876, against 0.689 for random flips at the same accuracy.
Structure alone does not lower PGR. Structure the student can express does.

Structure: `labelers()` derives each threshold for 30% error (normal
quantiles, and a 20,000-draw quantile of reference data for the class cut);
`score()` trains and grades one labeler; `seeded()` swaps the reference's
module `random`.
"""

from __future__ import annotations

import contextlib
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "11-scalable-oversight-weak-to-strong"
SEEDS, ERROR = range(3), 0.30          # every labeler is wrong on 30% of inputs


def margin(x):
    return x[0] + x[1] - 0.5 * x[2]


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def class_cut(ref):
    """x0 below which positives make up 30% of all inputs, from 20,000 reference draws."""
    with seeded(ref, 999):
        data = ref.gen(20000)
    return sorted(x[0] for x, y in data if y == 1)[int(ERROR * len(data))]


def labelers(ref):
    """Five weak labelers, each wrong on 30% of inputs in expectation."""
    agree = 1 - math.acos(2 / 3) / math.pi
    keep = (1 - ERROR - (1 - agree)) / (2 * agree - 1)   # weak_label's accuracy knob
    slab, band, cut = (statistics.NormalDist().inv_cdf(1 - ERROR),
                       1.5 * statistics.NormalDist().inv_cdf(0.5 + ERROR / 2), class_cut(ref))
    return {
        "random flips": lambda x, y, r: 1 - y if r.random() < ERROR else y,
        "hard cases": lambda x, y, r: 1 - y if abs(margin(x)) < band else y,
        "x2 slab": lambda x, y, r: 1 - y if x[2] > slab else y,
        "one class": lambda x, y, r: 0 if y == 1 and x[0] < cut else y,
        "reference x0 rule": lambda x, y, r: ref.weak_label(x, keep),
    }, (round(keep, 3), round(slab, 3), round(cut, 2))


def score(ref, label, ev, tr, ceiling, rng):
    """(weak accuracy, strong-on-weak accuracy, PGR, share of weak errors strong fixes)."""
    ev_labels = [label(x, y, rng) for x, y in ev]
    weak = sum(w == y for w, (_, y) in zip(ev_labels, ev)) / len(ev)
    model = ref.train_strong([(x, label(x, y, rng)) for x, y in tr])
    wrong = [(x, y) for w, (x, y) in zip(ev_labels, ev) if w != y]
    w2s = ref.accuracy(model, ev)
    return weak, w2s, (w2s - weak) / (ceiling - weak), ref.accuracy(model, wrong)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fns, knobs = labelers(ref)
    runs = {name: [] for name in fns}
    for seed in SEEDS:
        with seeded(ref, seed):
            ev, tr = ref.gen(1000), ref.gen(1000)
            ceiling = ref.accuracy(ref.train_strong(list(tr)), ev)
            for name, fn in fns.items():
                runs[name].append(score(ref, fn, ev, tr, ceiling, random.Random(100 + seed)))
    table = {n: tuple(round(sum(c) / len(c), 3) for c in zip(*rows)) for n, rows in runs.items()}
    return {"table": table, "knobs": knobs}


def verify(result):
    t = result["table"]
    col = {i: {n: v[i] for n, v in t.items()} for i in range(4)}
    return [
        practice.Check(
            "ANSWER: it depends on where the errors sit, and three of four structures lower PGR",
            max(abs(w - 0.70) for w in col[0].values()) < 0.012
            and result["knobs"] == (0.931, 0.524, 0.76)
            and col[2] == {"random flips": 0.689, "hard cases": 0.876, "x2 slab": 0.154,
                           "one class": -0.066, "reference x0 rule": 0.074}
            and col[1] == {"random flips": 0.907, "hard cases": 0.961, "x2 slab": 0.756,
                           "one class": 0.678, "reference x0 rule": 0.732},
            f"(x0-rule knob, slab, class cut) {result['knobs']}; weak accuracy {col[0]}; "
            f"strong-on-weak {col[1]}; PGR {col[2]}",
        ),
        practice.Check(
            "FINDING: the strong model copies an error that is consistent with an input region",
            col[3] == {"random flips": 0.913, "hard cases": 0.868, "x2 slab": 0.589,
                       "one class": 0.04, "reference x0 rule": 0.189}
            and col[1]["one class"] < col[0]["one class"],
            f"share of the labeler's errors strong-on-weak gets right {col[3]}",
        ),
        practice.Check(
            "FINDING: errors on hard cases raise PGR above the random baseline",
            col[2]["hard cases"] > col[2]["random flips"] > col[2]["x2 slab"],
            f"PGR hard cases {col[2]['hard cases']} vs random flips {col[2]['random flips']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
