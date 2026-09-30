"""Exercise 3 — no threshold beats zero, because the wrong labels are the confident ones.

    Add a `confidence_threshold` knob. Sweep it from 0 to 1 and plot precision-recall per category.

Reading of the exercise: the knob wraps the lesson's `Detector` without
touching it: a verdict whose confidence is below the threshold is reported
as `benign`, through the same `analyze()` interface, so the lesson's own
`evaluate` scores every setting against the 50 taxonomy fixtures from
lesson 82 and the 25 benign prompts. The sweep takes 21 steps, 0.00 to
1.00 by 0.05. The plot is per category: the distinct (recall, precision)
points the sweep visits, drawn as a text scatter (`python ex03_...py`
prints it) with each category's initial letter. A point where nothing
fired has no precision and is left out.

**ANSWER: the best threshold is the lesson's default.** Macro F1 is 0.713
for every threshold up to 0.40, then only falls, to 0 at 0.95. Each
category's curve has 2 to 5 distinct points. No setting raises any
category's recall, and above 0.40 recall drops first for role-play (0.60
to 0.40 by 0.50) and multi-turn-ramp. The knob does raise precision in two
places: encoding-trick reaches 1.00 at 0.60 and prefix-injection at 0.65.

**FINDING: most of the sweep is flat.** The 53 rules carry 13 hand-set
scores from 0.40 to 0.90, so the 9 thresholds from 0.00 to 0.40 are the
untouched detector, and 0.95 and 1.00 silence every rule.

**FINDING: raising the threshold lowers precision for two categories.** At
0.85 role-play precision falls from 0.86 to 0.50 and instruction-override
from 0.75 to 0.50. The 5 wrong-category verdicts average confidence 0.732,
above the 0.710 of the 31 true positives (cs-08 at 0.88 and mt-06 at 0.85
are mislabelled with near-top scores), so a cut on confidence does not
remove the wrong answers first. There is no benign false positive at any threshold, so the
knob has nothing to trade on the lesson's benign corpus.

**FINDING: the lesson's metrics report precision 0.0 when nothing fires.**
`PerCategoryMetrics.precision` returns 0.0 for tp + fp = 0, so a plot
drawn from the lesson's report puts every category at (0, 0) at t = 1.0,
as if it were maximally wrong rather than silent.

Expected output: three PASS checks.
"""

from __future__ import annotations

import sys
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "83-prompt-injection-detector"
STEPS = 20          # thresholds 0.00, 0.05, ..., 1.00


def load():
    """main.py does `from rules import ...` and `from benign import ...`; register both for that import."""
    deps = {name: parity.load_reference(PHASE, LESSON, name) for name in ("rules", "benign")}
    saved = {name: sys.modules.get(name) for name in deps}
    sys.modules.update(deps)
    try:
        return deps["rules"], deps["benign"], parity.load_reference(PHASE, LESSON, "main")
    finally:
        for name, module in saved.items():
            sys.modules.pop(name) if module is None else sys.modules.__setitem__(name, module)


def thresholded(main, detector, t):
    """The knob: a verdict whose confidence is below `confidence_threshold` t is reported as benign."""
    analyze = lambda p: v if (v := detector.analyze(p)).confidence >= t else main.Verdict("benign", 0.0, [])  # noqa: E731
    return types.SimpleNamespace(analyze=analyze)


def sweep(main, fixtures, benign):
    rows = []
    for t in (step / STEPS for step in range(STEPS + 1)):
        per = (report := main.evaluate(thresholded(main, main.Detector(), t), fixtures, benign))["per_category"]
        rows.append({"t": t, "per": per, "macro_f1": sum(m["f1"] for m in per.values()) / len(per),
                     "benign_fp": sum(report["benign_false_positives_by_category"].values())})
    return rows


def confidences(main, fixtures, benign):
    """Confidence of every fire, split by whether the fired category is the prompt's own."""
    detector, labelled = main.Detector(), [(f["category"], f["prompt"]) for f in fixtures] + [("benign", p) for p in benign]
    fires = [(v.category == truth, v.confidence) for truth, v in ((t, detector.analyze(str(p))) for t, p in labelled) if v.category != "benign"]
    return [c for ok, c in fires if ok], [c for ok, c in fires if not ok]


def curves(rows):
    """Per category, the distinct (recall, precision) points the sweep visits, where anything fired."""
    out = {}
    for m, cat in ((m, cat) for row in rows for cat, m in row["per"].items()):
        point = (m["recall"], m["precision"])
        if m["tp"] + m["fp"] and point not in out.setdefault(cat, []):
            out[cat].append(point)
    return out


def ascii_plot(curves_by_cat, size=10):
    """Precision up, recall across, each category's initial at its points; '*' where two meet."""
    grid = [[" "] * (size + 1) for _ in range(size + 1)]
    for cat, (recall, precision) in ((c, p) for c, points in curves_by_cat.items() for p in points):
        y, x, mark = size - round(precision * size), round(recall * size), cat[0].upper()
        grid[y][x] = mark if grid[y][x] in (" ", mark) else "*"
    return "\n".join(["  |" + "".join(r) for r in grid] + ["  +" + "-" * size + "- recall 0..1, precision 0..1 up"])


def solve():
    rules, benign_mod, main = load()
    fixtures, benign = main.load_taxonomy(), benign_mod.prompts()
    rows, (right, wrong) = sweep(main, fixtures, benign), confidences(main, fixtures, benign)
    best = max(r["macro_f1"] for r in rows)
    return {"rows": rows, "best": best, "flat_to": max(r["t"] for r in rows if abs(r["macro_f1"] - best) < 1e-9),
            "falling": all(a["macro_f1"] >= b["macro_f1"] for a, b in zip(rows, rows[1:])),
            "mean_right": sum(right) / len(right), "mean_wrong": sum(wrong) / len(wrong), "n": (len(right), len(wrong)),
            "curves": curves(rows), "plot": ascii_plot(curves(rows)), "scores": sorted({float(r["score"]) for r in rules.all_rules()})}


def check_answer(result):
    rows, sizes = result["rows"], {c: len(p) for c, p in result["curves"].items()}
    return practice.Check(
        "ANSWER: the best threshold is the lesson's default -- macro F1 is 0.713 for every t <= 0.40, then only falls",
        abs(result["best"] - 0.713) < 1e-3 and result["flat_to"] == 0.4 and result["falling"]
        and rows[-2]["macro_f1"] == 0.0 and sorted(sizes.values()) == [2, 3, 4, 4, 5, 5],
        f"best macro F1 {result['best']:.3f} at t in [0, {result['flat_to']:.2f}], non-increasing, 0 from t={rows[-2]['t']:.2f}; "
        f"distinct PR points per category {sizes}",
    )


def check_flat(result):
    scores, top = result["scores"], result["rows"][-1]["per"]
    return practice.Check(
        "FINDING: 13 hand-set scores from 0.40 to 0.90 leave 9 of 21 thresholds equal to no threshold, "
        "and the lesson reports precision 0.0 when nothing fires",
        (len(scores), scores[0], scores[-1]) == (13, 0.4, 0.9) and sum(r["t"] <= 0.4 for r in result["rows"]) == 9
        and all((m["tp"], m["fp"], m["precision"]) == (0, 0, 0.0) for m in top.values()),
        f"scores {scores}; at t=1.0 every category is tp=fp=0 with precision 0.0 (this plot drops those points)",
    )


def check_not_monotone(result):
    t0, t85 = result["rows"][0]["per"], result["rows"][17]["per"]
    rp, io = (t0["role-play"]["precision"], t85["role-play"]["precision"]), (t0["instruction-override"]["precision"], t85["instruction-override"]["precision"])
    return practice.Check(
        "FINDING: raising the threshold lowers precision for role-play and instruction-override -- the wrong labels are the confident ones",
        rp[1] < rp[0] and io[1] < io[0] and result["mean_wrong"] > result["mean_right"]
        and not any(r["benign_fp"] for r in result["rows"]),
        f"precision at t=0 -> 0.85: role-play {rp[0]:.2f} -> {rp[1]:.2f}, instruction-override {io[0]:.2f} -> {io[1]:.2f}; "
        f"{result['n'][1]} wrong-category verdicts average {result['mean_wrong']:.3f} vs {result['mean_right']:.3f} "
        f"for {result['n'][0]} true positives; 0 benign false positives at every threshold",
    )


def verify(result):
    return [check(result) for check in (check_answer, check_flat, check_not_monotone)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    print(solve()["plot"])
    raise SystemExit(practice.selfcheck(globals()))
