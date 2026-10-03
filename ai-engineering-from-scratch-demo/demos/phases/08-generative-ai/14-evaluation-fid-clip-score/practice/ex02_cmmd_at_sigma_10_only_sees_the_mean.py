"""Exercise 2 — CMMD at the paper's bandwidth only sees the mean; FID sees the shape.

    **Medium.** Implement CMMD from synthetic CLIP-style features (see Jayasumana
    et al., 2024 for the formula). Compare sensitivity to quality differences vs
    FID.

Reading of the exercise: CMMD is the unbiased squared MMD between two pools of
unit-norm CLIP embeddings under a Gaussian RBF kernel of bandwidth sigma = 10,
scaled by 1000. Synthetic "CLIP-style features" are unit vectors in 8-D around a
shared direction; the kernel's distance comes from the lesson's own `clip_like`
(for unit vectors `|a - b|^2 = 2 - 2 cos`), and FID is the lesson's own `fid`.
"Sensitivity" is measured as a z-score: how many null standard deviations a
degraded pool moves each metric, over 12 seeded draws of N=100 per pool.

**ANSWER: CMMD is ahead on a shifted mean and ~80x behind on a reshaped one.**
Tilting the generated cloud by 0.25 rad moves CMMD by z = 7.4 and FID by z = 4.9.
Keeping the cloud's centre and total spread but squeezing its variance from 7
directions into 2 moves FID by z = 64.0 and CMMD by z = 0.79 -- CMMD cannot see it.

**FINDING: at sigma = 10 CMMD is a linear-kernel MMD, i.e. a mean difference.**
Unit vectors are at most distance 2 apart, so `exp(-r^2 / 200)` never leaves
`[0.98, 1]` and is `1 - r^2/200` to first order -- which makes CMMD exactly
`1000/sigma^2 * |mu_r - mu_g|^2` (unbiased) plus a term at least 100x smaller.
The two agree within 0.0039 on every seed. The doc's "no Gaussian assumption" is true;
"better at detecting subtle quality differences" is not, at this bandwidth, for
any difference that leaves the mean alone.

**FINDING: the bandwidth, not the method, is the blind spot.** The same CMMD at
sigma = 0.5 flags the reshaped cloud at z = 29.8.

**CONTROL: CMMD is unbiased where FID is not.** On identical pools CMMD averages
-0.0070 and is negative on 7 of 12 seeds; FID is at least 0.0154 on every seed.

Structure: `cloud` draws features; `cmmd` is the metric; `zscores` compares.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "14-evaluation-fid-clip-score"
DIM, N, SEEDS, SPREAD, TILT = 8, 100, 12, 0.4, 0.25
WIDE, NARROW = range(1, DIM), (1, 2)


def cloud(rng, dims, spread, tilt=0.0):
    """N unit vectors around (cos tilt, sin tilt, 0, ...) with noise in `dims`."""
    out = []
    for _ in range(N):
        vec = [math.cos(tilt), math.sin(tilt)] + [0.0] * (DIM - 2)
        for d in dims:
            vec[d] += rng.gauss(0, spread)
        norm = math.sqrt(sum(v * v for v in vec))
        out.append([v / norm for v in vec])
    return out


def unbiased(kernel, x, y):
    """Unbiased MMD^2 under `kernel`."""
    wx = sum(kernel(a, b) for i, a in enumerate(x) for j, b in enumerate(x) if i != j)
    wy = sum(kernel(a, b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j)
    cross = sum(kernel(a, b) for a in x for b in y)
    n, m = len(x), len(y)
    return wx / (n * (n - 1)) + wy / (m * (m - 1)) - 2 * cross / (n * m)


def cmmd(ref, x, y, sigma=10.0):
    """CMMD: 1000 x unbiased MMD^2, RBF kernel on the lesson's clip_like distance."""
    gamma = 1 / (2 * sigma * sigma)
    return 1000 * unbiased(lambda a, b: math.exp(-gamma * (2 - 2 * ref.clip_like(a, b))), x, y)


def zscores(null, case):
    return (statistics.mean(case) - statistics.mean(null)) / statistics.stdev(null)


def draws(ref):
    """Per-seed scores for null, tilted and reshaped generated pools."""
    keys = ("fid_null", "fid_tilt", "fid_shape", "cmmd_null", "cmmd_tilt", "cmmd_shape")
    rows = {k: [] for k in keys + ("narrow_null", "narrow_shape", "gap")}
    dot = lambda a, b: sum(p * q for p, q in zip(a, b))  # noqa: E731
    for seed in range(SEEDS):
        rng = random.Random(900 + seed)
        real, twin = cloud(rng, WIDE, SPREAD), cloud(rng, WIDE, SPREAD)
        tilt = cloud(rng, WIDE, SPREAD, TILT)
        shape = cloud(rng, NARROW, SPREAD * math.sqrt(len(WIDE) / len(NARROW)))
        for name, gen in (("null", twin), ("tilt", tilt), ("shape", shape)):
            rows[f"fid_{name}"].append(ref.fid(real, gen))
            rows[f"cmmd_{name}"].append(cmmd(ref, real, gen))
        rows["narrow_null"].append(cmmd(ref, real, twin, 0.5))
        rows["narrow_shape"].append(cmmd(ref, real, shape, 0.5))
        rows["gap"].append(abs(rows["cmmd_shape"][-1] - 10 * unbiased(dot, real, shape)))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = draws(ref)
    return {
        "z": {
            f"{m}_{c}": zscores(rows[f"{m}_null"], rows[f"{m}_{c}"])
            for m in ("fid", "cmmd")
            for c in ("tilt", "shape")
        },
        "z_narrow": zscores(rows["narrow_null"], rows["narrow_shape"]),
        "gap": max(rows["gap"]),
        "cmmd_null": statistics.mean(rows["cmmd_null"]),
        "cmmd_negative": sum(v < 0 for v in rows["cmmd_null"]),
        "fid_null_min": min(rows["fid_null"]),
    }


def verify(result):
    z = result["z"]
    return [
        practice.Check(
            "ANSWER: CMMD ahead on a shifted mean, far behind FID on a reshaped cloud",
            z["fid_tilt"] > 3 and z["cmmd_tilt"] > 3 and z["fid_shape"] > 50 * z["cmmd_shape"],
            f"tilting the cloud {TILT} rad moves FID by z={z['fid_tilt']:.1f} and CMMD by "
            f"z={z['cmmd_tilt']:.1f}; moving its variance from {len(WIDE)} directions into "
            f"{len(NARROW)} moves FID by z={z['fid_shape']:.1f} and CMMD by "
            f"z={z['cmmd_shape']:.2f}",
        ),
        practice.Check(
            "FINDING: at sigma=10 CMMD is a linear-kernel MMD, a mean difference",
            result["gap"] < 0.01,
            f"on unit vectors exp(-r^2/200) stays in [0.98, 1], so CMMD equals 10 x the "
            f"unbiased |mu_r - mu_g|^2 within {result['gap']:.4f} on every seed: it is "
            "blind to any degradation that leaves the mean alone",
        ),
        practice.Check(
            "FINDING: the bandwidth is the blind spot, not the method",
            result["z_narrow"] > 10,
            f"the same CMMD at sigma=0.5 flags the reshaped cloud at z={result['z_narrow']:.1f}",
        ),
        practice.Check(
            "CONTROL: CMMD is unbiased on identical pools, FID is not",
            abs(result["cmmd_null"]) < 0.05 and result["fid_null_min"] > 0,
            f"null CMMD averages {result['cmmd_null']:.4f} and is negative on "
            f"{result['cmmd_negative']} of {SEEDS} seeds; null FID is at least "
            f"{result['fid_null_min']:.4f} on every seed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
