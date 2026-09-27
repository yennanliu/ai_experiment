"""Exercise 2 — regenerated from its datasheet the model scores 0.996 and a 0.03 parity gap is inside the 0.088 noise.

    Extend the model card with a quantitative disaggregated analysis across
    two demographic groups (Lesson 20).

Reading of the exercise: a disaggregated analysis has to be measured, so the
dataset is regenerated exactly as the datasheet describes it (1,500 examples,
2-d Gaussian features, one binary sensitive attribute, label
`x[0] + x[1] > 0`, split 1000/500 as Lesson 21 does) with seed 26 -- the
datasheet names none. The classifier and the group metrics are Lesson 21's
own `train`, `predict`, `demographic_parity` and `equalized_odds`, imported,
with their module RNG swapped for a seeded one: Lesson 20 measures
embeddings (WEAT) and has no classifier to disaggregate, while Lesson 21 is
where the card's own Metrics line points. The same pipeline is also run on
Lesson 21's data under its seed 53.

**ANSWER: the new Quantitative Analysis rows, per group (group 0 / group 1,
n = 250 / 250): accuracy 0.992 / 1.000, selection rate 0.536 / 0.500, TPR
0.985 / 1.000, FPR 0.000 / 0.000.** Demographic-parity gap -0.036 with a 95%
interval of +/-0.088; TPR gap +0.015.

**FINDING: the card's numbers match neither dataset.** On the datasheet's
data the model scores 0.996, not 0.97, and the parity gap has the opposite
sign to the card's. On Lesson 21's data -- where the Metrics line sends the
reader -- it scores 0.710 with a parity gap of +0.429 and a TPR gap of +0.326
(the same +0.429 Lesson 21's own `main()` prints), against the card's +0.03
and -0.01.

**FINDING: at the datasheet's size a 0.03 gap cannot be told from zero.**
The attribute plays no part in the label rule, so any selection gap is
sampling noise. With 500 test rows the 95% half-width on the parity gap is
0.088, and over 1000 seeds the test split's own label-rate gap -- what a
perfect classifier's parity gap equals -- reaches |0.03| or more in 50.4% of
draws. The card has to report the interval, or the gap means nothing.

Structure: `datasheet_data()` builds the data from the datasheet text;
`fit()` runs Lesson 21's pipeline; `table()` is the per-group block.
"""

from __future__ import annotations

import contextlib
import functools
import io
import math
import pathlib
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "26-model-system-dataset-cards"
L21 = "21-fairness-criteria-group-individual-counterfactual"
SEED, TRAIN, TEST, SWEEP = 26, 1000, 500, 1000
EX01 = practice.load_module(next(pathlib.Path(__file__).resolve().parent.glob("ex01_*.py")))


def datasheet_data(n, rng):
    """(features, label, attribute) rows as the datasheet's Composition describes them."""
    rows = [(rng.choice([0, 1]), rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)]
    return [([x0, x1, float(a)], int(x0 + x1 > 0), a) for a, x0, x1 in rows]


def seeded(l21, rng, fn):
    """Run `fn` with Lesson 21's module RNG swapped for `rng` (restored after)."""
    saved, l21.random = l21.random, rng
    try:
        return fn()
    finally:
        l21.random = saved


def fit(l21, rng, make):
    """Lesson 21's train/predict on a 1000/500 split drawn by `make`, in main()'s order."""
    _, test, model = seeded(l21, rng, lambda: (tr := make(TRAIN), make(TEST), l21.train(tr)))
    return model, test, l21.predict(model, test)


def table(l21, preds):
    """Per-group rows, the parity gap, its 95% half-width and the TPR gap."""
    dp, eo = l21.demographic_parity(preds), l21.equalized_odds(preds)
    rows = {g: dict(n=sum(a == g for *_, a in preds), accuracy=accuracy([t for t in preds if t[2] == g]),
                    selection=round(dp[g], 3), tpr=round(eo[g][0], 3), fpr=round(eo[g][1], 3))
            for g in (0, 1)}
    half = 1.96 * math.sqrt(sum(dp[g] * (1 - dp[g]) / rows[g]["n"] for g in (0, 1)))
    return rows, round(dp[1] - dp[0], 3), round(half, 3), round(eo[1][0] - eo[0][0], 3)


def accuracy(preds):
    return round(sum(p == y for p, y, _ in preds) / len(preds), 3)


@functools.lru_cache(maxsize=1)
def measured():
    """The datasheet model and its test predictions -- shared with Exercises 3 and 4."""
    l21 = parity.load_reference(PHASE, L21, "main")
    rng = random.Random(SEED)
    model, test, preds = fit(l21, rng, lambda n: datasheet_data(n, rng))
    return l21, model, test, preds


def chance_gap(threshold=0.03):
    """Share of seeds whose test split alone has a label-rate gap >= threshold."""
    rates = ([sum(y for _, y, a in rows if a == g) / sum(a == g for *_, a in rows) for g in (0, 1)]
             for rows in (datasheet_data(TEST, random.Random(s)) for s in range(SWEEP)))
    return sum(abs(r[1] - r[0]) >= threshold for r in rates) / SWEEP


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    l21, _, _, preds = measured()
    _, _, preds21 = fit(l21, random.Random(53), l21.gen)
    with contextlib.redirect_stdout(out := io.StringIO()):
        seeded(l21, random.Random(53), l21.main)     # Lesson 21's own printed baseline
    rows, gap, half, tpr_gap = table(l21, preds)
    return {
        "rows": rows, "gap": gap, "half": half, "tpr_gap": tpr_gap, "accuracy": accuracy(preds),
        "l21": (accuracy(preds21), *table(l21, preds21)[1::2]), "chance": chance_gap(),
        "printed": float(re.search(r"parity .*gap=([-+.\d]+)", out.getvalue()).group(1)),
        "card": tuple(map(float, EX01.provenance(ref, EX01.cards(ref))["qa_numbers"])),
    }


def verify(result):
    rows, l21 = result["rows"], result["l21"]
    return [
        practice.Check(
            "ANSWER: per-group accuracy, selection rate, TPR and FPR with the gap's interval",
            rows == {0: dict(n=250, accuracy=0.992, selection=0.536, tpr=0.985, fpr=0.0),
                     1: dict(n=250, accuracy=1.0, selection=0.5, tpr=1.0, fpr=0.0)}
            and (result["gap"], result["half"], result["tpr_gap"]) == (-0.036, 0.088, 0.015),
            f"{rows}; parity gap {result['gap']:+} +/- {result['half']}, "
            f"TPR gap {result['tpr_gap']:+}",
        ),
        practice.Check(
            "FINDING: the card's numbers match neither dataset",
            result["card"] == (0.97, 0.03, -0.01) and result["accuracy"] == 0.996
            and result["gap"] < 0 < result["card"][1]
            and l21 == (0.71, 0.429, 0.326) and result["printed"] == l21[1],
            f"card (accuracy, parity gap, TPR gap) {result['card']}; datasheet data accuracy "
            f"{result['accuracy']}, parity gap {result['gap']:+}; Lesson 21 data (accuracy, parity gap, TPR gap) {l21}, "
            f"its main() prints gap {result['printed']:+}",
        ),
        practice.Check(
            "FINDING: at the datasheet's size a 0.03 gap cannot be told from zero",
            result["half"] == 0.088 and result["chance"] == 0.504,
            f"95% half-width {result['half']}; |label-rate gap| >= 0.03 in "
            f"{result['chance']:.1%} of {SWEEP} seeded 500-row test splits",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
