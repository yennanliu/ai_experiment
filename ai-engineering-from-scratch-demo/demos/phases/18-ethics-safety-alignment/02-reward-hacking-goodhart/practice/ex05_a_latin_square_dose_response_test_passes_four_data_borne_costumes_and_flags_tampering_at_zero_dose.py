"""Exercise 5 — a Latin-square dose-response test passes four data-borne costumes and flags tampering at zero dose.

    The 2026 unified view argues verbosity, sycophancy, unfaithful CoT, and
    evaluator tampering share a mechanism. Design a single experiment that
    would simultaneously falsify all four if the unified view is wrong.

Reading of the exercise: the unified view, as the lesson states it, is that
probability mass moves to outputs that exploit features which spuriously
correlated with approval in the preference data. It therefore predicts, for
every costume at once, that (1) with no such correlation in the data there
is no exploitation, and (2) exploitation follows one dose-response curve in
that correlation, whatever the costume. The experiment is one training
pipeline whose preference data carries a different labeler bias for each
costume, doses {0, 0.1, 0.2, 0.4} rotated in a 4x4 Latin square, so that
each of four datasets gives every costume a different dose and every costume
gets every dose. Each dataset is fitted with the reference's `train_proxy`
(10,000 labels, 8 task features + 4 costume features). The policy is the
reference's KL-constrained optimum at sqrt(KL) = 2.828, and the reading is
the costume's feature shift. A costume falsifies the view if its zero-dose
shift exceeds 0.1, or if its slope is more than 25% off the pooled slope.
The test is run twice: on a world where all four costumes come from the data,
and on one where tampering is an affordance -- the evaluator reads a field
the policy can write, so the proxy pays +0.3 for it whatever the labels say.

**ANSWER: dose every costume independently in one Latin-square run, and read
two predictions per costume.** In the all-data world no costume is flagged:
zero-dose shifts are at most 0.035, and slopes run 2.34-2.39 against a
pooled 2.36. In the affordance world the same test flags tampering and only
tampering: at zero dose it shifts 0.692, while the other three stay at or
under 0.033. One run can falsify any subset of the four, which is what
"simultaneously" requires.

**FINDING: the dose-0 cell is the decisive one, and it is the cell most
papers never run.** Removing a costume's correlation from the data is the
unified view's own remedy. A costume that still moves with the correlation
at zero is being driven by something other than the preference data, such
as the environment, the scorer's inputs or the rollout. The slope test adds
little here: tampering's slope in the affordance world is 2.18 against a
pooled 2.26, inside the 25% band.

**FINDING: the design needs dose-level power, which sets the label budget.**
Over 5 Latin squares (20 costume verdicts) in the all-data world, 1000
labels per dataset gives 4 false flags, with zero-dose shifts up to 0.143.
3000 labels gives none (up to 0.080), and 10,000 gives none (up to 0.044).
The effect to detect, 0.692, is 16x the 10,000-label noise ceiling.

Structure: `proxy()` swaps the reference's `D` and `GOLD_W` for a labeler
with the given costume biases (restored after), optionally adds the
affordance to the fitted tamper weight, and returns the costume shifts of
mu = sqrt(2 KL) w / |w|; `run()` executes the Latin square; `judge()`
applies the two predictions.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "02-reward-hacking-goodhart"
COSTUMES = ("verbosity", "sycophancy", "unfaithful_cot", "tampering")
DOSES = (0.0, 0.1, 0.2, 0.4)
REPS, BUDGET, AFFORDANCE, ZERO_TOL, SLOPE_TOL = 5,  8.0, 0.3, 0.1, 0.25


def proxy(ref, biases, seed, affordance, labels):
    """Costume shifts of the KL-constrained optimum for a proxy fit to biased labels."""
    saved = ref.D, ref.GOLD_W, ref.random
    ref.D, ref.GOLD_W, ref.random = 8 + len(COSTUMES), saved[1] + list(biases), random.Random(seed)
    try:
        w = ref.train_proxy(labels).w
    finally:
        ref.D, ref.GOLD_W, ref.random = saved
    w[-1] += affordance
    mu = [math.sqrt(2 * BUDGET) * x / math.sqrt(ref.dot(w, w)) for x in w]
    return mu[8:]


def run(ref, affordance, labels=10000, rep=0):
    """Latin square: dataset k gives costume c the dose DOSES[(c + k) % 4]."""
    cells = {c: [] for c in COSTUMES}
    for k in range(len(DOSES)):
        doses = [DOSES[(c + k) % len(DOSES)] for c in range(len(COSTUMES))]
        shifts = proxy(ref, doses, k + 4 * rep, affordance, labels)
        for name, dose, shift in zip(COSTUMES, doses, shifts):
            cells[name].append((dose, shift))
    return {c: sorted(v) for c, v in cells.items()}


def slope(points):
    xs, ys = zip(*points)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)


def judge(cells):
    """Per costume: (zero-dose shift, slope, flagged), plus the pooled slope."""
    slopes = {c: slope(p) for c, p in cells.items()}
    pooled = sum(slopes.values()) / len(slopes)
    out = {c: (round(p[0][1], 3), round(slopes[c], 2),
               abs(p[0][1]) > ZERO_TOL or abs(slopes[c] / pooled - 1) > SLOPE_TOL)
           for c, p in cells.items()}
    return out, round(pooled, 2)


def power(ref, labels):
    """All-data world over REPS Latin squares: (false flags, largest zero-dose shift)."""
    verdicts = [v for r in range(REPS) for v in judge(run(ref, 0.0, labels, r))[0].values()]
    return sum(v[2] for v in verdicts), max(abs(v[0]) for v in verdicts)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = " ".join(parity.doc_text(PHASE, LESSON).split())
    return {
        "data": judge(run(ref, 0.0)),
        "affordance": judge(run(ref, AFFORDANCE)),
        "power": {n: power(ref, n) for n in (1000, 3000, 10000)},
        "view_quoted": "spuriously correlated with approval in the preference data" in doc,
    }


def digest(judged):
    """(largest |zero-dose shift|, slope range, flagged costumes, largest non-tamper shift)."""
    verdicts, pooled = judged
    slopes = sorted(v[1] for v in verdicts.values())
    return (max(abs(v[0]) for v in verdicts.values()), (slopes[0], slopes[-1], pooled),
            [c for c, v in verdicts.items() if v[2]],
            max(abs(v[0]) for c, v in verdicts.items() if c != "tampering"))


def verify(result):
    (data, pooled), (aff, pooled_aff) = result["data"], result["affordance"]
    zero, slopes, data_flags, _ = digest(result["data"])
    _, _, flagged, others = digest(result["affordance"])
    pw, tamper = result["power"], aff["tampering"]
    return [
        practice.Check(
            "ANSWER: one Latin-square run passes four data-borne costumes and flags tampering",
            (result["view_quoted"], data_flags, zero, slopes, flagged, tamper[0], others)
            == (True, [], 0.035, (2.34, 2.39, 2.36), ["tampering"], 0.692, 0.033),
            f"all-data world (zero-dose shift, slope, flagged): {data}, pooled {pooled}; "
            f"affordance world: {aff}",
        ),
        practice.Check(
            "FINDING: the dose-0 cell is decisive; the slope test alone misses tampering",
            (tamper[1], pooled_aff) == (2.18, 2.26)
            and abs(tamper[1] / pooled_aff - 1) <= SLOPE_TOL,
            f"tampering slope {tamper[1]} vs pooled {pooled_aff}, inside the "
            f"{SLOPE_TOL:.0%} band; its zero-dose shift {tamper[0]} is what flags it",
        ),
        practice.Check(
            "FINDING: dose-level power sets the label budget",
            ({n: (k, round(z, 3)) for n, (k, z) in pw.items()}, round(tamper[0] / pw[10000][1]))
            == ({1000: (4, 0.143), 3000: (0, 0.08), 10000: (0, 0.044)}, 16),
            f"(false flags of {REPS * len(COSTUMES)}, largest zero-dose shift) by labels: {pw}; "
            f"effect {tamper[0]} = {tamper[0] / pw[10000][1]:.0f}x the 10,000-label noise ceiling",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
