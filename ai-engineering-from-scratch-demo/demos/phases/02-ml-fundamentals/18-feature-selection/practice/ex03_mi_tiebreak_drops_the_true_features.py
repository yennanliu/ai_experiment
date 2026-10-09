"""Exercise 3 — the MI tiebreak drops two of the three true features, and pairs miss mixtures.

    **Multicollinearity detection**: compute the correlation matrix for all
    features. Implement a function that, given a correlation threshold (e.g.,
    0.9), removes one feature from each highly-correlated pair (keeping the one
    with higher mutual information with the target). Test on the synthetic
    dataset and verify it removes the redundant correlated features.

Reading of the exercise: the data is the lesson's
`make_feature_selection_data(500, seed=42)`, all 500 rows. `prune` walks the
pairs with |r| > 0.9 from strongest to weakest and, where both are still
present, drops the one with lower mutual information as scored by the lesson's
own `mutual_information` (10 bins). "Redundant" is checked two ways: pairwise,
as asked, and by regressing each survivor on all the others (R^2), which is
what multicollinearity means.

**ANSWER: it removes 5 features, but not the ones labelled redundant.** Gone are
info_1, info_3, corr_0, corr_1 and info_2; 15 survive, one per signal
(info_0 for x1, info_4 for x2, corr_2 for x3) plus corr_3, corr_4 and the ten
noise columns. Of the five `corr_` features only two are removed.

**FINDING: the MI tiebreak throws away two of the three generating features.**
info_1 *is* x2 and info_2 *is* x3, yet each loses to a noisier copy: info_4
(x2 + 0.1 noise) by MI 0.104 against 0.099, and corr_2 (0.7 x3 + 0.3 noise) by
0.022 against 0.021. Those margins are smaller than the binned estimator's own
bias -- the ten pure-noise columns average MI 0.011 -- so the tiebreak is a
coin flip that happened to land on the copies.

**FINDING: pairwise correlation misses mixtures.** corr_3 = 0.5 x1 + 0.5 x2
survives with no survivor above |r| = 0.672, yet the other survivors explain
R^2 = 0.975 of it (a variance inflation factor of 39); corr_4 (0.6 x2 + 0.4 x3)
likewise, at 0.822 and R^2 = 0.929.

**CONTROL: the noise columns are untouched and genuinely independent:** each
has R^2 at most 0.040 on the other survivors.

Structure: `prune` is the requested function; `r_squared` is the check it lacks.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "18-feature-selection"
THRESHOLD = 0.9


def prune(corr, mi, threshold=THRESHOLD):
    """Drop the lower-MI member of each pair above `threshold`, strongest pair first."""
    n = len(mi)
    pairs = sorted(((abs(corr[i, j]), i, j) for i in range(n) for j in range(i + 1, n)
                    if abs(corr[i, j]) > threshold), reverse=True)
    keep, dropped = set(range(n)), []
    for _, i, j in pairs:
        if i in keep and j in keep:
            loser = j if mi[i] >= mi[j] else i
            keep.discard(loser)
            dropped.append(loser)
    return sorted(keep), dropped


def r_squared(X, target, others):
    A = np.c_[X[:, others], np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(A, X[:, target], rcond=None)
    return float(1 - np.var(X[:, target] - A @ coef) / np.var(X[:, target]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "feature_selection")
    X, y, names = ref.make_feature_selection_data(500, seed=42)
    corr, mi = np.corrcoef(X.T), ref.mutual_information(X, y, n_bins=10)
    keep, dropped = prune(corr, mi)
    survivors = {names[t]: (r_squared(X, t, [k for k in keep if k != t]),
                            float(max(abs(corr[t, k]) for k in keep if k != t))) for t in keep}
    return {"kept": [names[i] for i in keep], "dropped": [names[i] for i in dropped],
            "mi": dict(zip(names, mi.astype(float))), "survivors": survivors,
            "noise_mi": float(np.mean([m for n, m in zip(names, mi) if n.startswith("noise")]))}


def verify(result):
    mi, surv = result["mi"], result["survivors"]
    noise_r2 = max(v[0] for n, v in surv.items() if n.startswith("noise"))
    return [
        practice.Check(
            "ANSWER: five removed, one survivor per signal, but corr_2-4 stay",
            sorted(result["dropped"]) == ["corr_0", "corr_1", "info_1", "info_2", "info_3"]
            and {"info_0", "info_4", "corr_2", "corr_3", "corr_4"} <= set(result["kept"]),
            f"dropped {result['dropped']}; {len(result['kept'])} survive"),
        practice.Check(
            "FINDING: the MI tiebreak drops x2 and x3 themselves for noisier copies",
            mi["info_4"] > mi["info_1"] and mi["corr_2"] > mi["info_2"]
            and mi["info_4"] - mi["info_1"] < result["noise_mi"],
            f"info_4 {mi['info_4']:.3f} beats info_1 {mi['info_1']:.3f}; corr_2 "
            f"{mi['corr_2']:.3f} beats info_2 {mi['info_2']:.3f}; pure noise averages MI "
            f"{result['noise_mi']:.3f}, more than either margin"),
        practice.Check(
            "FINDING: pairwise correlation misses the mixtures corr_3 and corr_4",
            surv["corr_3"][0] > 0.95 and surv["corr_3"][1] < THRESHOLD,
            f"corr_3: max |r| {surv['corr_3'][1]:.3f}, R^2 {surv['corr_3'][0]:.3f} (VIF "
            f"{1 / (1 - surv['corr_3'][0]):.0f}); corr_4: max |r| {surv['corr_4'][1]:.3f}, "
            f"R^2 {surv['corr_4'][0]:.3f}"),
        practice.Check(
            "CONTROL: the noise columns are untouched and independent",
            all(f"noise_{i}" in result["kept"] for i in range(10)) and noise_r2 < 0.1,
            f"all ten survive with R^2 at most {noise_r2:.3f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
