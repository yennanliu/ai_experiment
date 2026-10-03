"""Exercise 1 — the sweet spot at 3.0 is seed 42's; on 20 other seeds it lies below 3.

    **Threshold tuning.** Run the Z-score detector with thresholds from 1.0 to
    5.0 in steps of 0.5. Plot precision and recall at each threshold. Where is
    the sweet spot for your data?

Reading of the exercise: "your data" is the lesson's own `make_anomaly_data`
(500 normals, 25 anomalies, seed 42), scored by its own `zscore_detect` and
`precision_recall`. "Plot" becomes the table in the check detail. The sweet
spot is read two ways: the grid threshold with the best F1, and the full
interval of thresholds that separates the two classes.

**ANSWER: 3.0, with F1 = 1.000, the only grid value that reaches it.**
Precision/recall by threshold: 1.0 -> 0.16/1.00, 2.0 -> 0.89/1.00, 2.5 ->
0.96/1.00, 3.0 -> 1.00/1.00, 3.5 -> 1.00/0.76, 4.0 -> 1.00/0.52. The data
separate perfectly: the largest normal score is 2.58 and the smallest
anomaly score is 3.04, so every threshold in that window is a sweet spot.

**FINDING: 3.0 is in that window only for seed 42.** On seeds 0-19 the window
ends below 3.0 every time (its upper end is at most 2.97), so threshold 3.0
misses anomalies on 20 of 20. The classes separate on 15 of 20 seeds.

**FINDING: 3.0 works because the anomalies inflate the std.** The 25 anomalies
are 4.8% of the points but push the per-feature std from 0.68 and 0.70 to 1.04
and 1.03, about 1.5x. That shrinks every z-score, so "3 sigma" here means about
4.5 sigma of the normal data. With the clean normal mean and std, the gap is
3.85-4.38 and threshold 3.0 has precision 0.89.

**CONTROL:** the anomalies are separable by construction. The generator keeps
only candidates more than 3 units (4.2 sigma) from the centre, and average
precision is 1.0 under both standardisations.

Structure: `sweep` is the precision/recall table; `window` is the separating
interval of a score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "16-anomaly-detection"
GRID = np.arange(1.0, 5.01, 0.5)


def sweep(ref, X, y):
    rows = []
    for t in GRID:
        flags, _ = ref.zscore_detect(X, threshold=t)
        rows.append((float(t), *ref.precision_recall(y, flags.astype(int))))
    return rows


def window(scores, y):
    """(largest normal score, smallest anomaly score): any threshold between separates."""
    return float(scores[y == 0].max()), float(scores[y == 1].min())


def clean_scores(X, y):
    normal = X[y == 0]
    return np.abs((X - normal.mean(0)) / normal.std(0)).max(1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "anomaly_detection")
    X, y = ref.make_anomaly_data(seed=42)
    _, scores = ref.zscore_detect(X)
    others = [window(ref.zscore_detect(Xs)[1], ys)
              for Xs, ys in (ref.make_anomaly_data(seed=s) for s in range(20))]
    clean = clean_scores(X, y)
    uppers = [hi for _, hi in others]
    return {
        "upper_range": (min(uppers), max(uppers)),
        "separate": sum(lo < hi for lo, hi in others),
        "table": sweep(ref, X, y), "window": window(scores, y), "others": others,
        "inflation": (X.std(0) / X[y == 0].std(0)).tolist(),
        "clean_window": window(clean, y),
        "clean_precision_at_3": ref.precision_recall(y, (clean > 3.0).astype(int))[0],
        "ap": (average_precision_score(y, scores), average_precision_score(y, clean)),
    }


def verify(result):
    table, (lo, hi) = result["table"], result["window"]
    best = max(table, key=lambda row: row[3])
    low_up, high_up = result["upper_range"]
    return [
        practice.Check(
            "ANSWER: F1 peaks at 3.0, inside a separating window",
            best[0] == 3.0 and best[3] == 1.0 and lo < 3.0 < hi,
            " | ".join(f"{t:.1f}: P {p:.2f} R {r:.2f}" for t, p, r, _ in table)
            + f"; separating window ({lo:.2f}, {hi:.2f})",
        ),
        practice.Check(
            "FINDING: on seeds 0-19 the window always ends below 3.0",
            high_up < 3.0,
            f"upper ends {low_up:.2f}-{high_up:.2f}; classes separate on "
            f"{result['separate']} of 20",
        ),
        practice.Check(
            "FINDING: the anomalies inflate the std, so 3.0 means about 4.5 clean sigma",
            1.4 < min(result["inflation"]) < max(result["inflation"]) < 1.6
            and result["clean_window"][0] > 3.0,
            f"std inflated {result['inflation'][0]:.2f}x and {result['inflation'][1]:.2f}x; "
            f"with clean statistics the window is ({result['clean_window'][0]:.2f}, "
            f"{result['clean_window'][1]:.2f}) and precision at 3.0 is "
            f"{result['clean_precision_at_3']:.3f}",
        ),
        practice.Check(
            "CONTROL: the generator makes the anomalies separable",
            min(result["ap"]) > 0.999,
            f"average precision {result['ap'][0]:.3f} contaminated, {result['ap'][1]:.3f} clean",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
