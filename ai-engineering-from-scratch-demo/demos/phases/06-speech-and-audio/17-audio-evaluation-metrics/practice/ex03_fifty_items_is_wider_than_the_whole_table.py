"""Exercise 3 — fifty items is three times wider than the whole table.

    **Hard.** Score your Lesson 10 LALM choice on MMAU-Pro speech + multi-audio
    subsets (50 items each). Report per-category accuracy and compare with the
    published number.

Reading of the exercise: no LALM is installed and MMAU-Pro is not here, so no
model is scored -- but "compare with the published number" is arithmetic, and the
arithmetic settles the exercise. The published numbers are read straight out of
Lesson 10's own benchmark table, parsed from its source rather than retyped.

**A 50-item accuracy cannot separate any two rows of that table.** At the
published 52.2% the binomial standard deviation over 50 items is **7.06 pp** and
the 95% interval is **±13.85 pp**, spanning **38.4% to 66.0%**. The whole table
spans **7.8 pp**, from Gemini 2.5 Pro at 60.0 to Qwen2.5-Omni-7B at 52.2 -- the
interval on one sample is **3.6x** the entire published spread.

Sampling 50 items for each of the five models at their published rates, 5,000
times: the published order comes back **5.2%** of the time (chance for five
labels is 0.83%) and the best model is picked **47.2%** of the time against 20%
for a coin.

**50 items also puts accuracy on a 2 pp grid**, and **all 11** of the exactly
reported figures in Lesson 10's table -- 21.2, 26.5, 47.6, 50.5, 51.9, 52.2,
52.5, 57.4, 61.5, 64.9, 73.4 -- fall between its rungs. The only entries that do
land on it are the four written with a tilde, which were already rounded. There
is no 50-item result that can equal any precisely published number.

**And three of the four published multi-audio numbers are below chance.**
MMAU-Pro's multi-audio subset is four-way multiple choice, so 25.0% is the floor;
Gemini 2.5 Pro at 22.0, Gemini 2.5 Flash at 21.2 and Qwen2.5-Omni-7B at 20.0 all
sit under it. Comparing a noisy 50-item score against a number that is itself
below chance is not a comparison of models.

**`mmau_accuracy` also fails silently on a length mismatch.** It zips the two
lists and divides by `len(predictions)`, so ten predictions against five golds
score **0.5** while five predictions against ten golds score **1.0** -- the same
disagreement read two ways, neither raising.

Structure: `published` parses Lesson 10's table out of its own source; `half_width`
is the binomial 95% half-interval; `sample` runs the Monte Carlo; `off_grid`
reports which published figures a 50-item score can never equal.
"""

from __future__ import annotations

import ast
import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "17-audio-evaluation-metrics"
TABLE_LESSON = "10-audio-language-models"
ITEMS, DRAWS, Z, CHOICES = 50, 5000, 1.96, 4


def published(ref):
    """Lesson 10's benchmark table, read out of its `main` rather than retyped."""
    tree = ast.parse(inspect.getsource(ref.main))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "models")
    return ast.literal_eval(node.value)


def as_percent(cell):
    return None if cell.strip("~%") in ("", "—") else float(cell.strip("~%"))


def half_width(rate, items=ITEMS, z=Z):
    return z * math.sqrt(rate * (1 - rate) / items)


def sample(rates, rnd, draws=DRAWS, items=ITEMS):
    """How often 50 items per model reproduces the published order."""
    names = [name for name, _ in rates]
    exact = best = 0
    for _ in range(draws):
        scored = {name: sum(rnd.random() < rate for _ in range(items)) for name, rate in rates}
        order = sorted(scored, key=lambda name: -scored[name])
        exact += order == names
        best += order[0] == names[0]
    return exact / draws, best / draws


def off_grid(values, items=ITEMS):
    step = 100 / items
    return [v for v in values if abs(v / step - round(v / step)) > 1e-9]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = published(parity.load_reference(PHASE, TABLE_LESSON, "main"))
    overall = [(row[0], as_percent(row[1]) / 100) for row in table]
    overall.sort(key=lambda pair: -pair[1])
    exact_cells = sorted({as_percent(c) for r in table for c in r[1:]
                          if as_percent(c) and "~" not in c})
    multi = [(row[0], as_percent(row[5])) for row in table if as_percent(row[5])]
    anchor = min(rate for _, rate in overall)
    exact, best = sample(overall, random.Random(3))
    return {
        "overall": overall, "anchor": anchor, "half": half_width(anchor),
        "spread": (overall[0][1] - overall[-1][1]) * 100,
        "exact": exact, "best": best, "models": len(overall),
        "off_grid": len(off_grid(exact_cells)), "cells": len(exact_cells),
        "below_chance": [name for name, value in multi if value < 100 / CHOICES],
        "multi": len(multi), "step": 100 / ITEMS,
        "short_golds": ref.mmau_accuracy(["A"] * 10, ["A"] * 5),
        "short_preds": ref.mmau_accuracy(["A"] * 5, ["A"] * 10),
    }


def verify(result):
    half, spread = result["half"] * 100, result["spread"]
    return [
        practice.Check(
            "ANSWER: a 50-item interval is 3.6x the whole published spread",
            2 * half > 3 * spread,
            f"at the published {result['anchor'] * 100:.1f}% the binomial standard deviation "
            f"over {ITEMS} items is {half / Z:.2f} pp and the 95% interval is +/-{half:.2f} pp, "
            f"spanning {(result['anchor'] * 100 - half):.1f}% to "
            f"{(result['anchor'] * 100 + half):.1f}%. Lesson 10's table spans {spread:.1f} pp "
            f"end to end, so the interval on one sample is {2 * half / spread:.1f}x all of it",
        ),
        practice.Check(
            "ANSWER: at 50 items the published order comes back 5% of the time",
            result["exact"] < 0.15 and result["best"] < 0.6,
            f"over {DRAWS} draws of {ITEMS} items for each of {result['models']} models the "
            f"published order returns {result['exact'] * 100:.1f}% of the time, against "
            f"{100 / math.factorial(result['models']):.2f}% for a shuffle, and the best model "
            f"is picked {result['best'] * 100:.1f}% of the time against "
            f"{100 / result['models']:.0f}% for a coin",
        ),
        practice.Check(
            "FINDING: no published figure lies on the grid a 50-item score can reach",
            result["off_grid"] == result["cells"],
            f"{ITEMS} items puts accuracy on multiples of {result['step']:.1f} pp, and all "
            f"{result['off_grid']} of the {result['cells']} exactly reported figures in Lesson "
            "10's table fall between the rungs. The only entries that land on it are the four "
            "written with a tilde, which were already rounded",
        ),
        practice.Check(
            "FINDING: three of the four published multi-audio numbers are below chance",
            len(result["below_chance"]) == 3,
            f"the multi-audio subset is {CHOICES}-way multiple choice, so "
            f"{100 / CHOICES:.1f}% is the floor, and {result['below_chance']} sit under it -- "
            f"{len(result['below_chance'])} of {result['multi']} rows that report the column. "
            "Comparing a noisy 50-item score against a below-chance number compares nothing",
        ),
        practice.Check(
            "CONTROL: `mmau_accuracy` reads a length mismatch two ways, and neither raises",
            result["short_golds"] == 0.5 and result["short_preds"] == 1.0,
            f"it zips the two lists and divides by len(predictions), so ten predictions against "
            f"five golds score {result['short_golds']} while five against ten score "
            f"{result['short_preds']}. A truncated run and a truncated key are the same bug and "
            "produce opposite numbers",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
