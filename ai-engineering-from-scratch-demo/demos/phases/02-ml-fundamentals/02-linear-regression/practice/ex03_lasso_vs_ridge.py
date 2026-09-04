"""Exercise 3 — Lasso against Ridge: which weights hit zero, and why.

    Implement Lasso regression (L1 regularization: penalty = alpha * sum(|w_i|)).
    Train on the multi-feature housing data. Compare which weights go to zero vs
    Ridge. Why does L1 produce sparse solutions while L2 does not?

Reading of the exercise: "goes to zero" needs a solver that can *reach* zero.
Subgradient descent on |w| oscillates across 0 and never lands on it, so this uses
**proximal** gradient descent, whose soft-threshold step produces exact 0.0.
Check 2 asserts exact zeros, which a subgradient implementation cannot pass.

The "why" is arithmetic: L1 subtracts a constant α·lr and clips at zero, L2
multiplies by (1 − α·lr) and so cannot arrive. Check 4 found something the
exercise does not anticipate — L2's weights on the *useless* features come out
2.4x **larger** than the unregularised fit's. See the README.
"""

from __future__ import annotations

import random

from harness import practice

SEED, N, N_FEATURES, N_INFORMATIVE = 42, 200, 8, 3
EPOCHS, LR, ALPHA = 3_000, 0.02, 0.35
TRUE_W = [4.0, -3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def make_data(rng):
    X = [[rng.gauss(0, 1) for _ in range(N_FEATURES)] for _ in range(N)]
    return X, [sum(w * x for w, x in zip(TRUE_W, row)) + rng.gauss(0, 0.5)
               for row in X]


def _gradient(X, y, w):
    n = len(y)
    res = [sum(wi * xi for wi, xi in zip(w, row)) - t for row, t in zip(X, y)]
    return [sum(r * row[j] for r, row in zip(res, X)) / n for j in range(len(w))]


def _proximal(w, penalty, alpha):
    """L1 soft-thresholds (can reach 0); L2 shrinks multiplicatively (cannot)."""
    if penalty == "l1":
        step = alpha * LR
        return [0.0 if abs(wi) <= step else wi - step * (1 if wi > 0 else -1)
                for wi in w]
    if penalty == "l2":
        return [wi * (1 - alpha * LR) for wi in w]
    return w


def fit(X, y, penalty, alpha=ALPHA):
    """Proximal gradient descent: a plain step, then the penalty's own operator."""
    w = [0.0] * N_FEATURES
    for _ in range(EPOCHS):
        gradient = _gradient(X, y, w)
        w = _proximal([wi - LR * g for wi, g in zip(w, gradient)], penalty, alpha)
    return w


def _per_fit(fits, fn):
    return {k: fn(v) for k, v in fits.items()}


def _summarise(fits, noise):
    return {
        "zeros": _per_fit(fits, lambda v: sum(1 for wi in v if wi == 0.0)),
        "noise_zeros": _per_fit(fits, lambda v: sum(1 for j in noise if v[j] == 0.0)),
        "signal_kept": _per_fit(fits, lambda v: min(abs(v[j])
                                                    for j in range(N_INFORMATIVE)) > 0.5),
        "noise_mass": _per_fit(fits, lambda v: sum(abs(v[j]) for j in noise)),
    }


def solve():
    rng = random.Random(SEED)
    X, y = make_data(rng)
    fits = {name: fit(X, y, name) for name in ("none", "l1", "l2")}
    noise_idx = [j for j, w in enumerate(TRUE_W) if w == 0.0]
    return {"fits": fits, "n_noise": len(noise_idx),
            **_summarise(fits, noise_idx)}


def verify(result):
    fits, zeros = result["fits"], result["zeros"]
    return [
        practice.Check(f"all three fits recover the {N_INFORMATIVE} informative weights",
                       all(result["signal_kept"].values()),
                       "; ".join(f"{k}: {[round(v, 2) for v in fits[k][:N_INFORMATIVE]]}"
                                 for k in fits) + f" vs true {TRUE_W[:N_INFORMATIVE]}"),
        practice.Check("ANSWER: only L1 produces zeros, and they are EXACT",
                       zeros["l1"] > 0 and zeros["l2"] == 0 and zeros["none"] == 0,
                       f"exact 0.0 weights: L1 {zeros['l1']}, L2 {zeros['l2']}, "
                       f"unregularised {zeros['none']}"),
        practice.Check(f"…and L1 zeroes the {result['n_noise']} genuinely useless features",
                       result["noise_zeros"]["l1"] == result["n_noise"],
                       f"{result['noise_zeros']['l1']} of {result['n_noise']} noise "
                       f"features exactly 0.0, no informative one dropped"),
        practice.Check("FINDING: L2's noise weights GROW — it over-shrinks the signal",
                       result["noise_mass"]["l2"] > result["noise_mass"]["none"]
                       and zeros["l2"] == 0,
                       f"|weight| on noise features: unregularised "
                       f"{result['noise_mass']['none']:.4f} -> L2 "
                       f"{result['noise_mass']['l2']:.4f}, a "
                       f"{result['noise_mass']['l2'] / result['noise_mass']['none']:.1f}x "
                       f"*rise*, with the signal 29% low ({fits['l2'][0]:.2f} vs 4.0). "
                       f"L1 keeps {fits['l1'][0]:.2f} and zeroes the noise"),
        practice.Check("WHY: L1 subtracts a constant and clips; L2 multiplies",
                       result["noise_mass"]["l1"] == 0.0,
                       f"L1's step is |w| − {ALPHA * LR:g} floored at 0; L2's is "
                       f"w × {1 - ALPHA * LR:g}. Subtraction with a floor reaches zero, "
                       f"multiplication cannot — arithmetic, not geometry"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
