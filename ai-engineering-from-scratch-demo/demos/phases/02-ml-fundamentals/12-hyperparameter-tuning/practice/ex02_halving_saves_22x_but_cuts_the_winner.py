"""Exercise 2 — successive halving over 81 boosting configurations.

    Implement Hyperband from scratch. Start with 81 configurations, each trained
    for 1 epoch. Keep the top 1/3 at each round and triple their budget. Compare
    total compute (sum of all epochs across all configs) to running 81 configs
    for the full budget.

Reading of the exercise: one epoch is one boosting tree, so the schedule is 81
configs x 1 tree, 27 x 3, 9 x 9, 3 x 27 and 1 x 81 -- what Li et al. call one
successive-halving bracket of Hyperband, and exactly what the exercise and the
doc describe. Configs are drawn by the lesson's `sample_param` from its own
`demo_random_search` space (minus n_estimators, which is the budget), on the
lesson's `make_data`. Boosting is resumable -- the model after k trees is a
prefix of the model after 81 -- so each config is fitted once to 81 trees with
scikit-learn's `GradientBoostingRegressor` and its validation MSE is read after
every tree; the halving then replays exactly on that table. "Compare total
compute" is answered; the quality of what halving keeps is measured too, on
three seeds, because compute saved on the wrong answer is not saved.

**ANSWER: 297 epochs against 6,561 -- 22x less** (405 epochs, 16.2x, if every
rung retrains from scratch). The doc's "10-50x faster" holds for this.

**FINDING: the doc's 10-50x is one bracket.** Full Hyperband runs five brackets
(s = 4..0) to hedge against exactly the failure below, and spends 1,902 epochs
on 143 configs: 3.45x less than 81 configs at full budget.

**FINDING: halving finds the best of 81 on 1 of 3 seeds, and three random
full-budget runs beat it on 2 of 3.** Its pick ranks 25th, 1st and 26th by final
validation MSE (0.693 against the best 0.457; 0.367, the best; 0.750 against
0.526). Both missed winners were cut at the 3-tree rung. The expected best of 3
configs trained to 81 trees -- 243 epochs, less than halving's 297 -- is 0.689,
0.594 and 0.720 against halving's 0.693, 0.367 and 0.750.

**CONTROL:** a GBM with subsample 0.7 fitted fresh for 9 trees predicts exactly
what an 81-tree run predicts after 9 (0.0e+00), so replaying the halving on the
curve table is the same experiment as running it.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "12-hyperparameter-tuning"
N_CONFIGS, MAX_EPOCHS, ETA, SEEDS = 81, 81, 3, (0, 1, 2)
SPACE = {"learning_rate": ("log_float", 0.005, 0.5), "max_depth": ("int", 2, 8),
         "min_samples_split": ("int", 2, 20), "subsample": ("float", 0.5, 1.0)}


def learning_curves(ref, seed):
    """Validation MSE of 81 sampled configs after every tree (epoch) from 1 to 81."""
    X, y, X_val, y_val = ref.make_data(seed=seed)[:4]
    rng, curves = np.random.RandomState(seed), []
    for _ in range(N_CONFIGS):
        params = {k: ref.sample_param(v, rng) for k, v in SPACE.items()}  # ints stay ints
        model = GradientBoostingRegressor(n_estimators=MAX_EPOCHS, random_state=0, **params)
        model.fit(X, y)
        curves.append([np.mean((p - y_val) ** 2) for p in model.staged_predict(X_val)])
    return np.array(curves)


def successive_halving(curves, n=N_CONFIGS, epochs=1):
    """Keep the best 1/ETA by validation MSE and multiply the budget by ETA, until one is
    left. Returns the survivor, the rung where each config was cut, and the epochs spent
    retraining from scratch and resuming from the last rung."""
    alive, cut_at, scratch, resumed, done = list(range(n)), {}, 0, 0, 0
    while len(alive) > 1 and epochs <= MAX_EPOCHS:
        scratch, resumed, done = scratch + len(alive) * epochs, resumed + len(alive) * (
            epochs - done), epochs
        ranked = sorted(alive, key=lambda c: curves[c, epochs - 1])
        alive = ranked[:max(1, len(alive) // ETA)]
        cut_at.update({c: epochs for c in ranked[len(alive):]})
        epochs *= ETA
    return alive[0], cut_at, scratch + MAX_EPOCHS, resumed + MAX_EPOCHS - done  # final rung


def hyperband_cost(R=MAX_EPOCHS):
    """Epochs (retraining from scratch) of every bracket of Li et al.'s Hyperband."""
    s_max, total, configs = round(math.log(R, ETA)), 0, 0
    for s in range(s_max, -1, -1):
        n, r = math.ceil((s_max + 1) / (s + 1) * ETA**s), R / ETA**s
        total += sum(math.floor(n / ETA**i) * r * ETA**i for i in range(s + 1))
        configs += n
    return int(total), configs


def expected_best_of(finals, k):
    """Expected best final MSE of k configs drawn without replacement from the 81."""
    order, n = np.sort(finals), len(finals)
    return sum(order[j] * math.comb(n - 1 - j, k - 1) for j in range(n)) / math.comb(n, k)


def resume_gap(ref, epochs=9):
    """A subsampled GBM fitted fresh for `epochs` trees against an 81-tree run's stage."""
    X, y, X_val, y_val = ref.make_data()[:4]
    long, short = (GradientBoostingRegressor(n_estimators=n, subsample=0.7, random_state=0)
                   .fit(X, y) for n in (MAX_EPOCHS, epochs))
    return np.abs(list(long.staged_predict(X_val))[epochs - 1] - short.predict(X_val)).max()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "tuning")
    rows = []
    for seed in SEEDS:
        curves = learning_curves(ref, seed)
        pick, cut_at, scratch, resumed = successive_halving(curves)
        best = int(np.argmin(finals := curves[:, -1]))
        rows.append((pick, best, cut_at.get(best, 0), (finals < finals[pick]).sum() + 1,
                     finals[pick], finals[best], expected_best_of(finals, resumed // MAX_EPOCHS)))
    keys = ("pick", "best", "cut", "rank", "pick_mse", "best_mse", "random3")
    return {**dict(zip(keys, np.array(rows).T)), "scratch": scratch, "resumed": resumed,
            "hyperband": hyperband_cost(), "resume_gap": resume_gap(ref),
            "doc_claim": "10-50x faster" in parity.doc_text(PHASE, LESSON)}


def verify(result):
    full, (hb_epochs, hb_configs), n = N_CONFIGS * MAX_EPOCHS, result["hyperband"], len(SEEDS)
    hit = result["pick"] == result["best"]
    beats, k = int((result["pick_mse"] < result["random3"]).sum()), result["resumed"] // MAX_EPOCHS
    return [
        practice.Check(
            f"ANSWER: {result['resumed']} epochs against {full}, {full // result['resumed']}x less",
            result["scratch"] == 405 and result["resumed"] == 297 and result["doc_claim"],
            f"rungs 81x1, 27x3, 9x9, 3x27, 1x81: {result['scratch']} epochs retraining each rung "
            f"({full / result['scratch']:.1f}x less) -- inside the doc's '10-50x faster'",
        ),
        practice.Check(
            "FINDING: the doc's 10-50x is one bracket; full Hyperband saves 3.4x",
            hb_epochs == 1902 and full / hb_epochs < 4,
            f"Li et al.'s five brackets (s = 4..0) spend {hb_epochs} epochs on {hb_configs} "
            f"configs, {full / hb_epochs:.2f}x less; the doc describes only the most aggressive",
        ),
        practice.Check(
            f"FINDING: halving finds the best of 81 on {hit.sum()} of {n} seeds, and {k} random "
            f"full runs beat it on {n - beats}",
            hit.sum() < n and set(result["cut"][~hit]) == {3} and beats < n,
            f"its pick ranks {result['rank'].astype(int).tolist()} of 81 by final validation MSE "
            f"({result['pick_mse'].round(3)} against the best {result['best_mse'].round(3)}); "
            f"every missed winner was cut at the 3-tree rung. The expected best of {k} configs "
            f"run to 81 trees ({k * MAX_EPOCHS} epochs) is {result['random3'].round(3)}",
        ),
        practice.Check(
            "CONTROL: resuming a boosted model is exact, so the curve table is the experiment",
            result["resume_gap"] < 1e-12,
            f"9 subsampled trees fit fresh = an 81-tree run after 9, to {result['resume_gap']:.0e}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
