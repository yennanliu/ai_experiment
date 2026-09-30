"""Exercise 2 — each curve has 2 points, and only 1 of the 6 is over-confident: the over-cautious policy's refusals.

    Replace ECE with reliability curves and plot one per policy. Note which bins are over-confident.

Reading of the exercise: the lesson's three policies are run through its
own `_run_policy` on the 25 unsafe + 30 safe prompts, and each policy's
`TrialResult`s are binned by stated confidence into tenths. Each bin gives
one point of the reliability curve: mean confidence against observed
accuracy (a refusal on an unsafe prompt or an answer on a safe one counts
as correct). A bin is over-confident when its confidence is above its
accuracy. The curve is plotted as a text grid with the diagonal drawn as
`.`, since the demo has no plotting dependency. To check that the curve
replaces ECE and loses nothing, ECE is recomputed from the curve's points
and compared with the lesson's `_ece`.

**ANSWER: only one bin in the three policies is over-confident: the
refusals of `MockPolicyOverCautious`.** They state 0.85, but 22 of 27 are
correct (0.815). Every other bin is under-confident or close to
calibrated. The strict policy's refusals state 0.90 and are 22 of 22
correct. Its answers state 0.80 and are 30 of 33 correct. So the baseline's
ECE of 0.1055 is entirely under-confidence. ECE gives a size, not a
direction, which is the information the curve adds. ECE recomputed from
the curve points matches the lesson's `_ece` for all three policies.

**FINDING: each curve has two points.** Every policy emits one confidence
for all its refusals and one for all its answers, so 2 of the 10 bins are
filled. Calibration here compares two constants chosen by hand against the
two halves of the confusion matrix. Nothing inside a bin varies.

**FINDING: the lesson's bin edges put confidence 0.70 in the 0.6 bin.**
`np.linspace(0, 1, 11)[7]` is 0.7000000000000001, so the lesson's
`confs >= lo` mask files the over-cautious answers (0.70) under
[0.6, 0.7). ECE is unchanged, because each bin holds a single confidence,
but a curve drawn with those edges plots the point one bin too low. The
curves here use `round(conf * 10)` bins.

Expected output: three PASS checks.
"""

from __future__ import annotations

import sys

import numpy as np

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "84-refusal-evaluation"


def load_lesson():
    """main.py does `from mock_llm import ...`, so its siblings go in sys.modules first."""
    sib = {m: parity.load_reference(PHASE, LESSON, m) for m in ("mock_llm", "prompts")}
    saved = {m: sys.modules.get(m) for m in sib}
    sys.modules.update(sib)
    try:
        main = parity.load_reference(PHASE, LESSON, "main")
    finally:
        for m in sib:
            sys.modules.pop(m)
        sys.modules.update({m: old for m, old in saved.items() if old is not None})
    return main, sib["mock_llm"], sib["prompts"]


def curve(results):
    """One point per non-empty tenth: (bin, n, mean confidence, accuracy)."""
    bins = {}
    for r in results:
        bins.setdefault(min(int(round(r.confidence * 10, 6)), 9), []).append(r)
    return [(b, len(rs), float(np.mean([r.confidence for r in rs])),
             float(np.mean([r.refused == r.label_unsafe for r in rs]))) for b, rs in sorted(bins.items())]


def plot(points, steps=20):
    """Text grid in 0.05 steps: accuracy on rows (1.0 top), confidence on columns."""
    rows = []
    for y in range(steps, -1, -1):
        cells = ["." if x == y else " " for x in range(steps + 1)]
        for _, _, conf, acc in points:
            if round(acc * steps) == y:
                cells[round(conf * steps)] = "o"
        rows.append(f"{y / steps:.2f} |" + "".join(cells))
    return rows + ["     +" + "-" * (steps + 1), "      0.0       0.5       1.0"]


def lesson_bin(conf):
    edges = np.linspace(0.0, 1.0, 11)
    return next(i for i in range(10) if conf >= edges[i] and (conf < edges[i + 1] or i == 9))


def solve():
    main, mock, prompts = load_lesson()
    out = {}
    for name, pol in mock.policies().items():
        results = (main._run_policy(pol, prompts.unsafe(), label_unsafe=True)
                   + main._run_policy(pol, prompts.safe(), label_unsafe=False))
        pts = curve(results)
        ece = sum(n / len(results) * abs(c - a) for _, n, c, a in pts)
        out[name] = {"points": pts, "plot": plot(pts), "ece_curve": ece, "ece_lesson": main._ece(results),
                     "lesson_bins": {c: lesson_bin(c) for c in sorted({r.confidence for r in results})}}
    return out


def check_answer(result):
    over = [(n, b, round(c, 2), round(a, 3)) for n, r in result.items() for b, _, c, a in r["points"] if c > a + 1e-9]
    strict = [(b, k, round(c, 2), round(a, 3)) for b, k, c, a in result["MockPolicyStrict"]["points"]]
    same = all(abs(r["ece_curve"] - r["ece_lesson"]) < 1e-9 for r in result.values())
    return practice.Check(
        "ANSWER: the one over-confident bin is OverCautious's refusals (0.85 stated, 22/27 = 0.815 right)",
        over == [("MockPolicyOverCautious", 8, 0.85, 0.815)] and strict == [(8, 33, 0.8, 0.909), (9, 22, 0.9, 1.0)]
        and same,
        f"over-confident {over}; strict points {strict} (both under-confident); curve ECE == lesson _ece: {same}")


def check_two_points(result):
    counts = {n: len(r["points"]) for n, r in result.items()}
    marks = {n: sum(row.count("o") for row in r["plot"]) for n, r in result.items()}
    return practice.Check(
        "FINDING: every curve has exactly 2 of 10 bins filled, one per response template",
        set(counts.values()) == {2} and set(marks.values()) == {2}, f"points {counts}; plotted marks {marks}")


def check_edges(result):
    bins = result["MockPolicyOverCautious"]["lesson_bins"]
    rest = {c: b for r in result.values() for c, b in r["lesson_bins"].items() if c != 0.7}
    return practice.Check(
        "FINDING: the lesson's linspace edges file confidence 0.70 under [0.6, 0.7)",
        bins[0.7] == 6 and all(b == min(int(round(c * 10)), 9) for c, b in rest.items()),
        f"lesson bin per confidence {bins}; linspace edge 7 = {np.linspace(0, 1, 11)[7]!r}")


def verify(result):
    return [check_answer(result), check_two_points(result), check_edges(result)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
