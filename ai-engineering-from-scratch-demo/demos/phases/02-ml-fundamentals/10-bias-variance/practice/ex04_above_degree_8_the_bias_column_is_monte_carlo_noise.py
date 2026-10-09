"""Exercise 4 — sin(x) moves the optimum to 3; above degree 8 the bias column is sampling noise.

    Modify the true function from a polynomial to `sin(x)`. How does the
    bias-variance decomposition change? Is there still a clear optimal degree?

Reading of the exercise: the lesson's true function is not a polynomial --
`docs/en.md` and the code both use `sin(1.5x) + 0.5x` -- so "from a polynomial"
is read as "from the lesson's function". The swap is made by replacing the
module's `true_function`, which the lesson's `generate_data` and
`bias_variance_decomposition` both look up at call time, and the degree 1-15
sweep is run for the lesson's function, `sin(x)`, and a true cubic
`0.5x^3 - x` as the polynomial the exercise imagines.

**ANSWER: the optimum moves from degree 6 to degree 3, and it is still a clear
one, if modest.** Bias^2 at degree 3 falls 0.0529 -> 0.0020 and at degree 1
0.431 -> 0.122: `sin(x)` has 2/3 the frequency, so a cubic tracks it. Degree 3
totals 0.282, 3.4% ahead of degree 4 (0.292) and ahead of 6 (0.308). Both
functions are odd, so bias falls in pairs: 4 is 3 plus a useless term.

**FINDING: from degree 8 up, the decomposition does not depend on the truth.**
Bias^2 and total agree between `sin(x)` and the lesson's function within 0.5%
at every degree 8-15, though their degree-1 bias^2 differs 3.5x. The sweep
reuses the same seeds, so both see the same noise, and up there the error is
all noise.

**FINDING: the lesson's high-degree "bias" is the variance leaking through 200
draws.** Its bias^2 column reads 0.0008, 0.0017, 0.545, 0.0061 at degrees
7-10. The bias is computed from a 200-draw mean prediction, which carries
variance/199 of its own; at degree 9 that is 0.552, 99% of the reported 0.545.
The rising, jagged bias at high degree is sampling error, not bias.

**CONTROL: the premise needs a real polynomial.** `docs/en.md` uses
`f(x) = sin(1.5x) + 0.5x`. With `0.5x^3 - x` as the truth, bias^2 is 2.78 at
degree 1 and 6.7e-5 at degree 3 (Monte Carlo zero), and the optimum is degree 3.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "10-bias-variance"
DEGREES, DRAWS = list(range(1, 16)), 200


def cubic(x):
    return 0.5 * x**3 - x


def decompose(ref, truth):
    """The lesson's sweep with `truth` swapped in as the module's true_function."""
    original = ref.true_function
    ref.true_function = truth or original
    try:
        return ref.bias_variance_decomposition(DEGREES, n_bootstrap=DRAWS)
    finally:
        ref.true_function = original


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bias_variance")
    runs = {"lesson": decompose(ref, None), "sin": decompose(ref, np.sin),
            "cubic": decompose(ref, cubic)}
    return {
        "best": {k: ref.find_optimal(r) for k, r in runs.items()},
        "bias": {k: {d: r[d]["bias_sq"] for d in DEGREES} for k, r in runs.items()},
        "var": {d: runs["lesson"][d]["variance"] for d in DEGREES},
        "total": {k: {d: r[d]["total_error"] for d in DEGREES} for k, r in runs.items()},
        "doc_truth": "f(x) = sin(1.5x) + 0.5x" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    best, bias, var, total = result["best"], result["bias"], result["var"], result["total"]
    sin_t = total["sin"]
    high = range(8, 16)
    same = max(abs(bias["sin"][d] / bias["lesson"][d] - 1) for d in high)
    leak = bias["lesson"][9] / (var[9] / (DRAWS - 1))
    return [
        practice.Check(
            "ANSWER: sin(x) moves the optimum from degree 6 to 3, a modest but clear win",
            all((best["lesson"] == 6, best["sin"] == 3, sin_t[4] > 1.02 * sin_t[3])),
            f"bias^2 at degree 3 falls {bias['lesson'][3]:.4f} -> {bias['sin'][3]:.4f} and at "
            f"degree 1 {bias['lesson'][1]:.3f} -> {bias['sin'][1]:.3f}; sin(x) has 2/3 the "
            f"frequency, so a cubic tracks it. Degree 3 totals {sin_t[3]:.3f}, ahead of degree 4 "
            f"({sin_t[4]:.3f}, +{sin_t[4] / sin_t[3] - 1:.1%}) and 6 ({sin_t[6]:.3f})",
        ),
        practice.Check(
            "FINDING: from degree 8 up the decomposition does not depend on the truth at all",
            all((same < 0.01, *[abs(sin_t[d] / total["lesson"][d] - 1) < 0.01 for d in high])),
            f"bias^2 and total agree between sin(x) and the lesson's function within "
            f"{same:.1%} at every degree 8-15, though their degree-1 bias^2 differs 3.5x: "
            "the same seeds give the same noise, and there the error is all noise",
        ),
        practice.Check(
            "FINDING: the high-degree 'bias' is the variance leaking through 200 draws",
            all((0.9 < leak < 1.1, bias["lesson"][9] > 100 * bias["lesson"][7])),
            f"the bias^2 column reads {bias['lesson'][7]:.4f}, {bias['lesson'][8]:.4f}, "
            f"{bias['lesson'][9]:.3f}, {bias['lesson'][10]:.4f} at degrees 7-10; the estimated "
            f"mean prediction carries variance/{DRAWS - 1}, and at degree 9 that is "
            f"{var[9] / (DRAWS - 1):.3f}, {leak:.0%} of the reported bias^2. Bias does not "
            "rise with degree here; the sample mean is just too noisy to show it is ~0",
        ),
        practice.Check(
            "CONTROL: the lesson's truth is not a polynomial; a real cubic gives zero bias at 3",
            all((result["doc_truth"], best["cubic"] == 3, bias["cubic"][3] < 1e-3,
                 bias["cubic"][1] > 1)),
            f"docs/en.md uses f(x) = sin(1.5x) + 0.5x. With 0.5x^3 - x as the truth, bias^2 "
            f"is {bias['cubic'][1]:.2f} at degree 1 and {bias['cubic'][3]:.1e} at degree 3 -- "
            f"Monte Carlo zero -- and the optimum is degree {best['cubic']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
