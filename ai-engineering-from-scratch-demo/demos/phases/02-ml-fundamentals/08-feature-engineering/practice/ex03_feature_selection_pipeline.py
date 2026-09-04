"""Exercise 3 — variance threshold, correlation filter and MI ranking, in series.

    Build an automated feature selection pipeline that combines variance
    threshold, correlation filtering, and mutual information ranking. Apply it to
    the housing dataset and compare model performance (use a simple linear
    regression) with all features vs selected features.

Reading of the exercise: the housing frame has no junk in it, so the pipeline has
nothing to select. Three columns are added whose right fate is known — one for
each stage — so each stage is scored on removing its own. The README has the rest.
"""

from __future__ import annotations

import random

from harness import parity, practice, stats

PHASE, LESSON = "02-ml-fundamentals", "08-feature-engineering"
SEED, N_ROWS, TRAIN, TOP_K, TARGET_BINS = 3, 200, 150, 5, 5
VAR_THRESHOLD, CORR_THRESHOLD = 0.01, 0.9
NAMES = ["sqft", "bedrooms", "age", "hood_downtown", "hood_rural", "hood_suburbs",
         "has_pool", "sqft_dup", "near_constant", "noise"]


def _junk(index: int, sqft: float, rng) -> list:
    """One column per stage: collinear with sqft, near-constant, independent of y."""
    return [sqft * 1.0000001 + 0.001, 1.0 + (0.001 if index % 97 == 0 else 0.0),
            rng.gauss(0, 1)]


def build_features(ref, rows) -> list:
    """Impute, one-hot, then append the junk. Column order is `NAMES`."""
    sqft, _ = ref.impute_median([r["sqft"] for r in rows])
    age, _ = ref.impute_median([r["age"] for r in rows])
    hood, categories = ref.one_hot_encode([r["neighborhood"] for r in rows])
    assert list(categories) == ["downtown", "rural", "suburbs"], f"order: {categories}"
    rng = random.Random(SEED)
    return [[sqft[i], float(r["bedrooms"]), age[i], *[float(v) for v in hood[i]],
             1.0 if r["has_pool"] else 0.0, *_junk(i, sqft[i], rng)]
            for i, r in enumerate(rows)]


def _take(matrix, indices):
    return [[row[j] for j in indices] for row in matrix]


def select(ref, matrix, target) -> dict:
    """The three stages in series, each reported as the indices it keeps."""
    variance = ref.variance_threshold(matrix, threshold=VAR_THRESHOLD)
    kept = [variance[j] for j in
            ref.remove_correlated(_take(matrix, variance), threshold=CORR_THRESHOLD)]
    binned = ref.bin_values(target, n_bins=TARGET_BINS)
    columns = {j: [row[j] for row in matrix] for j in kept}
    mi = {"scores": {j: ref.mutual_information(c, binned) for j, c in columns.items()},
          "raw_scores": {j: ref.mutual_information(c, target) for j, c in columns.items()}}
    return {"variance": variance, "correlation": kept, **mi,
            "mi": sorted(kept, key=lambda j: -mi["scores"][j])[:TOP_K]}


def solve():
    np = parity.try_numpy()
    if np is None:
        raise practice.Skip("needs numpy for least squares: uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "features")
    rows = ref.make_housing_data(n=N_ROWS, seed=42)
    matrix = build_features(ref, rows)
    price = [r["price"] for r in rows]
    stages = select(ref, matrix, price)
    fits = {s: stats.least_squares(np, _take(matrix, stages[s]), price, TRAIN)
            for s in ("variance", "correlation", "mi")}
    fits["all"] = stats.least_squares(np, matrix, price, TRAIN)
    return {"stages": stages, "fits": fits, "baseline": stats.rmse(
        price[TRAIN:], price[TRAIN:], 0.0, sum(price[:TRAIN]) / TRAIN)}


def _report(stages) -> dict:
    """Pre-formatted rankings, so `verify` stays a flat list of claims."""
    rank = sorted(stages["scores"], key=lambda j: -stages["scores"][j])
    raw = sorted(stages["raw_scores"], key=lambda j: -stages["raw_scores"][j])
    return {"rank": rank, "raw": raw,
            "kept": [NAMES[j] for j in stages["correlation"]],
            "binned": ", ".join(f"{NAMES[j]} {stages['scores'][j]:.4f}" for j in rank),
            "unbinned": ", ".join(f"{NAMES[j]} {stages['raw_scores'][j]:.3f}" for j in raw)}


def verify(result):
    fits, stages, got = result["fits"], result["stages"], _report(result["stages"])
    kept, ranked, raw = got["kept"], got["rank"], got["raw"]
    return [
        practice.Check(f"all {len(NAMES)} columns built, three junk by construction",
                       len(NAMES) == len(stages["variance"]) + 1,
                       "; ".join(f"{s} k={fits[s]['k']}" for s in fits)),
        practice.Check("stages 1 and 2 each remove exactly their own target column",
                       NAMES.index("near_constant") not in stages["variance"]
                       and "sqft_dup" not in kept and len(kept) == len(NAMES) - 2,
                       f"variance_threshold({VAR_THRESHOLD}) drops near_constant, "
                       f"remove_correlated({CORR_THRESHOLD}) drops sqft_dup: {kept}"),
        practice.Check(f"FINDING: stage 3 ranks pure noise 2nd of {len(ranked)}, 3rd unbinned",
                       NAMES[ranked[1]] == "noise" and NAMES[raw[2]] == "noise",
                       f"MI against the {TARGET_BINS}-bin target — {got['binned']} — and "
                       f"against raw price — {got['unbinned']}"),
        practice.Check("ANSWER: selection does not improve held-out error here",
                       fits["mi"]["test"] > fits["correlation"]["test"],
                       f"held-out RMSE {fits['all']['test']:,.0f} on all {fits['all']['k']}, "
                       f"{fits['correlation']['test']:,.0f} after two stages, "
                       f"{fits['mi']['test']:,.0f} on MI top-{TOP_K}; mean-only "
                       f"{result['baseline']:,.0f}"),
        practice.Check("MECHANISM: what selection actually buys is conditioning",
                       fits["all"]["cond"] / fits["mi"]["cond"] > 1e10,
                       f"design condition {fits['all']['cond']:.2e} against "
                       f"{fits['mi']['cond']:.2e}, {fits['all']['cond'] / fits['mi']['cond']:.1e}"
                       f"x better — the full design is singular, `lstsq` only hides it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
