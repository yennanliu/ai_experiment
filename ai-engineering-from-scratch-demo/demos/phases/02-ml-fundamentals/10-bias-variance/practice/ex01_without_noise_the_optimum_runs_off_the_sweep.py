"""Exercise 1 — without noise the optimum runs off the end of the sweep.

    Run the decomposition with `noise_std=0` (no noise). What happens to the
    irreducible error term? Does the optimal complexity change?

Reading of the exercise: run the lesson's own `bias_variance_decomposition`
over degrees 1-15 (its `demo_complexity_tradeoff` sweep) twice, at the default
`noise_std=0.5` and at `noise_std=0`, and pick the optimum with the lesson's
`find_optimal`. "Irreducible error" is the `noise` term the function reports.

**ANSWER: the noise term drops from 0.25 to exactly 0, and the optimum moves
from degree 6 to degree 15** -- the largest degree the sweep offers, so the
"optimum" is the edge of the grid rather than a minimum.

**FINDING: with no noise there is no interior optimum.** The noiseless total
error is 1.30e-03 at degree 6 and 6.22e-13 at degree 15, 2.1e9 times lower; with
noise, degree 15 scores 26504. `sin(1.5x) + 0.5x` is analytic, so each extra term
removes bias and, with nothing to chase, costs almost no variance.

**FINDING: variance does not vanish with the noise.** At `noise_std=0` the
variance is still 0.0298 at degree 1 (69% of its noisy 0.0434) and 0.0096 at
degree 3. A misspecified model's best fit depends on where the 30 x values land,
so "variance = fitting the noise" is only half of it.

**CONTROL: the Total column is not evidence for the decomposition.** The lesson
computes it as the noiseless error plus `noise_std**2`, so it equals
Bias^2 + Variance + Noise to 4.7e-10 by algebra, while the doc says it "should
approximately equal" it. Measured instead on 200 degree-6 fits, the error is
0.3413 against freshly drawn noisy labels and 0.0917 against the truth: the
noise adds 0.2497, sigma^2 = 0.25 as claimed.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "10-bias-variance"
DEGREES, CHECK_DEGREE, DRAWS = list(range(1, 16)), 6, 200


def noisy_label_error(ref, degree=CHECK_DEGREE, noise=0.5):
    """Expected error against freshly drawn *noisy* labels, measured, not added."""
    rng = np.random.RandomState(42)
    x_test = np.linspace(-2.5, 2.5, 100)
    to_labels, to_truth = [], []
    for _ in range(DRAWS):
        x, y = ref.generate_data(n_samples=30, noise_std=noise, seed=rng.randint(0, 100000))
        pred = ref.predict_polynomial(x_test, ref.fit_polynomial(x, y, degree))
        truth = ref.true_function(x_test)
        to_labels.append(np.mean((pred - truth - rng.normal(0, noise, x_test.size)) ** 2))
        to_truth.append(np.mean((pred - truth) ** 2))
    return float(np.mean(to_labels)), float(np.mean(to_truth))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bias_variance")
    runs = {s: ref.bias_variance_decomposition(DEGREES, noise_std=s) for s in (0.5, 0.0)}
    noisy, clean = runs[0.5], runs[0.0]
    gap = max(
        abs(r["total_error"] - r["bias_sq"] - r["variance"] - r["noise"])
        for run in runs.values()
        for r in run.values()
    )
    return {
        "noise": (noisy[1]["noise"], clean[1]["noise"]),
        "best": (ref.find_optimal(noisy), ref.find_optimal(clean)),
        "clean_total": {d: clean[d]["total_error"] for d in DEGREES},
        "noisy_total": {d: noisy[d]["total_error"] for d in DEGREES},
        "variance": {d: (noisy[d]["variance"], clean[d]["variance"]) for d in (1, 3)},
        "identity_gap": gap,
        "measured": noisy_label_error(ref),
        "doc_says": "should approximately equal bias^2 + variance + noise"
        in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    (n_noisy, n_clean), (b_noisy, b_clean) = result["noise"], result["best"]
    clean, noisy = result["clean_total"], result["noisy_total"]
    v1, v3 = result["variance"][1], result["variance"][3]
    to_labels, to_truth = result["measured"]
    return [
        practice.Check(
            "ANSWER: the noise term goes to 0 and the optimum moves from degree 6 to 15",
            n_noisy == 0.25 and n_clean == 0.0 and (b_noisy, b_clean) == (6, 15),
            f"noise term {n_noisy} -> {n_clean}; find_optimal picks degree {b_noisy} at "
            f"noise_std=0.5 (total {noisy[b_noisy]:.4f}) and degree {b_clean} at noise_std=0, "
            f"the largest degree the sweep offers",
        ),
        practice.Check(
            "FINDING: with no noise there is no interior optimum, the error just keeps falling",
            clean[15] < 1e-9 and clean[6] / clean[15] > 1e6 and noisy[15] > 100,
            f"noiseless total error is {clean[6]:.2e} at degree 6 and {clean[15]:.2e} at "
            f"degree 15, {clean[6] / clean[15]:.1e}x lower; with noise degree 15 scores "
            f"{noisy[15]:.0f}. Overfitting needs something to overfit: sin(1.5x) + 0.5x is "
            "analytic, so each extra term only removes bias",
        ),
        practice.Check(
            "FINDING: variance does not vanish with the noise",
            v1[1] > 0.5 * v1[0] and v3[1] > 0.005,
            f"at noise_std=0 the variance is still {v1[1]:.4f} at degree 1 ({v1[1] / v1[0]:.0%} "
            f"of its noisy {v1[0]:.4f}) and {v3[1]:.4f} at degree 3: a misspecified model's "
            "fit depends on where the 30 x values land, noise or not",
        ),
        practice.Check(
            "CONTROL: Total = B+V+N is built in, so it is checked against noisy labels instead",
            result["doc_says"] and result["identity_gap"] < 1e-8
            and abs(to_labels - to_truth - 0.25) < 0.01,
            f"the lesson adds noise_std**2 to the noiseless error, so Total and B+V+N agree to "
            f"{result['identity_gap']:.1e} by algebra (the doc says 'should approximately "
            f"equal'). Measured on the same 200 degree-6 fits, the error is {to_labels:.4f} "
            f"against fresh noisy labels and {to_truth:.4f} against the truth: the noise adds "
            f"{to_labels - to_truth:.4f}, sigma^2 = 0.25 as the decomposition says",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
