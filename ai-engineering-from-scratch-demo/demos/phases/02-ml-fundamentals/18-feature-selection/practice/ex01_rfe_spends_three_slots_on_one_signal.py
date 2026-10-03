"""Exercise 1 — forward selection keeps one feature per signal; RFE spends 3 slots on one.

    **Forward selection**: implement the opposite of RFE. Start with zero
    features. At each step, add the feature that improves model performance the
    most. Stop when adding features no longer helps. Compare the selected
    features against RFE results. Which is faster? Which gives better results?

Reading of the exercise: data, scaling and model are the lesson's own:
`make_feature_selection_data(500, seed=42)`, standardised on the first 400 rows,
and `simple_logistic_importance` (lr 0.1, 200 epochs, the lesson's RFE setting)
as the model. "Performance" is accuracy on rows 300-400 of a model fitted on
rows 0-300, so the 100 test rows stay unseen; "no longer helps" is no strict
improvement. RFE is the lesson's `rfe` to 5 features on all 400 training rows,
as in its `main`. "Faster" is counted in model fits and in feature-columns
trained, not wall-clock. "Better" is accuracy of each subset refitted on the 400
training rows (300 epochs) and scored both on the lesson's 100 test rows and on
5000 fresh rows from the same generator (seed 7).

**ANSWER: forward selection keeps 3 features -- info_0, info_4, info_2, one
copy of each of the three signals x1, x2, x3 -- and stops; RFE's 5 are info_0,
info_1, info_2, info_3, corr_0.** Three of RFE's five are copies of x1 (pairwise
r = 0.995 and 0.993 on the training rows).

**FINDING: "which is faster" depends on what you count.** RFE fits 15 models,
forward selection 74, so by fits RFE is 4.9x faster. But forward selection's
fits are tiny (1 to 4 features) and RFE's are wide (20 down to 6): by
feature-columns trained it is 180 against 195, a tie.

**FINDING: the lesson's 100-row test set ranks them backwards.** On it RFE
scores 0.96 and forward selection 0.94 -- two rows. On 5000 fresh rows forward
selection scores 0.929, RFE 0.919 and all 20 features 0.912.

**CONTROL: forward selection found the generating features.** The oracle subset
(info_0, info_1, info_2 = x1, x2, x3 themselves) scores 0.930 on the fresh rows,
within 0.002 of forward selection's three.

Structure: `forward` is the greedy search; `accuracy` scores a subset.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "18-feature-selection"


def standardised(ref):
    X, y, names = ref.make_feature_selection_data(500, seed=42)
    mu, sd = X[:400].mean(0), X[:400].std(0)
    fresh, y_fresh, _ = ref.make_feature_selection_data(5000, seed=7)
    return (X - mu) / sd, y, names, (fresh - mu) / sd, y_fresh


def correct(w, b, X, y):
    return float(np.mean(((X @ w + b) >= 0) == y))


def forward(ref, Z, y):
    """Greedy addition on a 300/100 split inside the training rows."""
    chosen, best, fits, columns = [], 0.0, 0, 0
    while len(chosen) < Z.shape[1]:
        scores = []
        for f in (f for f in range(Z.shape[1]) if f not in chosen):
            cols = chosen + [f]
            w, b = ref.simple_logistic_importance(Z[:300, cols], y[:300], 0.1, 200)
            scores.append((correct(w, b, Z[300:400, cols], y[300:400]), f))
            fits, columns = fits + 1, columns + len(cols)
        score, f = max(scores)
        if score <= best:
            break
        chosen, best = chosen + [f], score
    return chosen, best, fits, columns


def accuracy(ref, cols, Z, y, fresh, y_fresh):
    w, b = ref.simple_logistic_importance(Z[:400, cols], y[:400], 0.1, 300)
    return correct(w, b, Z[400:, cols], y[400:]), correct(w, b, fresh[:, cols], y_fresh)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "feature_selection")
    Z, y, names, fresh, y_fresh = standardised(ref)
    chosen, val, fits, columns = forward(ref, Z, y)
    mask, _ = ref.rfe(Z[:400], y[:400], n_features_to_select=5, lr=0.1, epochs=200)
    rfe = [int(i) for i in np.where(mask)[0]]
    r = np.corrcoef(Z[:400].T)
    return {
        "forward": [names[i] for i in chosen], "rfe": [names[i] for i in rfe], "val": val,
        "fits": (fits, len(names) - 5), "columns": (columns, sum(range(6, len(names) + 1))),
        "acc": {k: accuracy(ref, c, Z, y, fresh, y_fresh) for k, c in (
            ("forward", chosen), ("rfe", rfe), ("all", list(range(20))), ("oracle", [0, 1, 2]))},
        "x1_copies": (r[0, 3], r[0, 5]),
    }


def verify(result):
    acc = result["acc"]
    (fwd_fits, rfe_fits), (fwd_cols, rfe_cols) = result["fits"], result["columns"]
    return [
        practice.Check(
            "ANSWER: forward keeps one copy per signal; RFE spends three slots on x1",
            result["forward"] == ["info_0", "info_4", "info_2"]
            and {"info_0", "info_3", "corr_0"} <= set(result["rfe"]),
            f"forward {result['forward']} (validation accuracy {result['val']:.2f}); RFE "
            f"{result['rfe']}, whose info_0, info_3, corr_0 are one signal (r = "
            f"{result['x1_copies'][0]:.3f}, {result['x1_copies'][1]:.3f})"),
        practice.Check(
            "FINDING: RFE is faster by fits, not by work",
            fwd_fits > 4 * rfe_fits and abs(fwd_cols - rfe_cols) < 0.1 * rfe_cols,
            f"{fwd_fits} fits against {rfe_fits}, but {fwd_cols} feature-columns trained "
            f"against {rfe_cols}"),
        practice.Check(
            "FINDING: the lesson's 100-row test set ranks them backwards",
            acc["rfe"][0] > acc["forward"][0] and acc["forward"][1] > acc["rfe"][1] > acc["all"][1],
            f"100 test rows: RFE {acc['rfe'][0]:.2f}, forward {acc['forward'][0]:.2f}; 5000 "
            f"fresh rows: forward {acc['forward'][1]:.3f}, RFE {acc['rfe'][1]:.3f}, all 20 "
            f"{acc['all'][1]:.3f}"),
        practice.Check(
            "CONTROL: forward selection matches the generating features",
            abs(acc["forward"][1] - acc["oracle"][1]) < 0.005,
            f"x1, x2, x3 themselves score {acc['oracle'][1]:.3f} on the fresh rows"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
