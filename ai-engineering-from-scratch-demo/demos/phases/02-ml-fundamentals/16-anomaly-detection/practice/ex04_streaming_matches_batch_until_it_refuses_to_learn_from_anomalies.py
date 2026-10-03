"""Exercise 4 — streaming agrees with batch on 523 of 525; skipping anomalies costs precision.

    **Streaming anomaly detection.** Modify the Z-score detector to work in a
    streaming setting: update the running mean and variance as new points arrive
    (Welford's online algorithm). Compare to batch Z-score on the same data.

Reading of the exercise: points of the lesson's `make_anomaly_data` (seed 42)
arrive in their stored order. Each one is scored against the running mean and
population std of the points before it, as the lesson's `zscore_detect` uses
`np.std` (ddof 0), and is then added to the statistics by Welford's update.
Threshold 3.0 and two points of warm-up. Batch is the lesson's `zscore_detect`
on the whole array.

**ANSWER: nearly identical.** Streaming flags 25 points with precision and
recall 0.96 against batch 1.00/1.00. The two disagree on 2 of 525 points: a
normal at position 2, scored against two points of history, and an anomaly at
position 48 (batch z 3.09). By then the two anomalies already seen have pushed
the running std of the first feature to 1.13, above its final 1.04, so this
anomaly scores 2.85. After the
last point the running mean and std equal the batch values within 2.4e-15.

**FINDING: "don't learn from anomalies" makes it worse.** A natural change is
to skip the update for flagged points. Precision then falls to 0.66 (38 flags,
13 extra false alarms) and recall stays 1.00, because the running std stays at
the clean value. The batch detector's precision depends on the anomalies
inflating the std, which is exercise 1's finding seen from the other side.

**CONTROL: this is why Welford.** On 10,000 N(0,1) values shifted by 1e8, the
textbook running `E[x^2] - E[x]^2` gives a variance of -2.0. Welford gives
0.975268, matching the true 0.975268 to 6e-10.

Structure: `stream` is the online detector; `naive_variance` is the one-pass
formula Welford replaces.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "16-anomaly-detection"
THRESHOLD, WARMUP, OFFSET = 3.0, 2, 1e8


def stream(X, skip_flagged=False):
    """Score each point against the running stats, then fold it in with Welford."""
    n, mean, m2 = 0, np.zeros(X.shape[1]), np.zeros(X.shape[1])
    flags = np.zeros(len(X), dtype=bool)
    for i, x in enumerate(X):
        if n >= WARMUP:
            std = np.sqrt(m2 / n)
            std[std == 0] = 1.0
            flags[i] = np.abs((x - mean) / std).max() > THRESHOLD
        if skip_flagged and flags[i]:
            continue
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    return flags, mean, np.sqrt(m2 / n)


def naive_variance(values):
    total = squares = 0.0
    for v in values:
        total, squares = total + v, squares + v * v
    return squares / len(values) - (total / len(values)) ** 2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "anomaly_detection")
    X, y = ref.make_anomaly_data(seed=42)
    batch, _ = ref.zscore_detect(X, threshold=THRESHOLD)
    flags, mean, std = stream(X)
    skipped, _, _ = stream(X, skip_flagged=True)
    noise = np.random.RandomState(0).normal(0, 1, 10_000)
    _, _, welford_std = stream((noise + OFFSET)[:, None])
    differ = np.flatnonzero(flags != batch)
    return {
        "batch": ref.precision_recall(y, batch.astype(int)),
        "stream": ref.precision_recall(y, flags.astype(int)), "n_flags": int(flags.sum()),
        "differ": differ.tolist(), "differ_labels": y[differ].tolist(),
        "stat_gap": max(np.abs(mean - X.mean(0)).max(), np.abs(std - X.std(0)).max()),
        "skip": ref.precision_recall(y, skipped.astype(int)), "skip_flags": int(skipped.sum()),
        "naive": naive_variance(noise + OFFSET), "welford": float(welford_std[0] ** 2),
        "true": float(noise.var()),
    }


def verify(result):
    sp, sr, _ = result["stream"]
    kp, kr, _ = result["skip"]
    return [
        practice.Check(
            "ANSWER: streaming matches batch on all but two points",
            result["batch"][:2] == (1.0, 1.0) and len(result["differ"]) == 2
            and result["stat_gap"] < 1e-12,
            f"streaming flags {result['n_flags']} with P {sp:.2f} R {sr:.2f}; batch P/R "
            f"{result['batch'][0]:.2f}/{result['batch'][1]:.2f}; they differ at positions "
            f"{result['differ']} (labels {result['differ_labels']}); final stats agree to "
            f"{result['stat_gap']:.1e}",
        ),
        practice.Check(
            "FINDING: skipping flagged points from the update costs a third of the precision",
            kp < 0.75 and kr == 1.0,
            f"{result['skip_flags']} flags, precision {kp:.2f}, recall {kr:.2f}: the std stays "
            "clean, so 3.0 becomes a tighter cut than the batch detector's",
        ),
        practice.Check(
            "CONTROL: the one-pass textbook variance collapses at an offset of 1e8",
            result["naive"] < 0 and abs(result["welford"] - result["true"]) < 1e-8,
            f"naive {result['naive']:.1f}, Welford {result['welford']:.6f}, true "
            f"{result['true']:.6f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
