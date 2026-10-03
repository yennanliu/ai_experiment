"""Exercise 2 — tenfold data cuts variance 14x; the lesson's degree-6 optimum is Monte Carlo luck.

    Increase the training set size from 30 to 300. How does this affect the
    variance component? Does the optimal polynomial degree shift?

Reading of the exercise: run the lesson's own `bias_variance_decomposition` over
its degree 1-15 sweep at `n_train=30` and `n_train=300` and pick the optimum
with `find_optimal`. That function draws fresh datasets for every degree, so
its degree-to-degree differences carry Monte Carlo noise; to tell whether an
optimum is real, the same sweep is repeated on common datasets (every degree fit
to the same 300 draws, through the lesson's `generate_data`/`fit_polynomial`).

**ANSWER: variance falls 14.4x at the lesson's optimum (0.0583 -> 0.0040 at
degree 6), and `find_optimal` picks degree 6 at both sizes.** The fall is
steeper than 1/n, and far steeper at high degree: degree 15 drops 2.5e6x
(2.63e4 -> 0.0105), because at n=30 it is nearly interpolating 30 points.

**FINDING: degree 6 never wins once every degree sees the same data.** The
target `sin(1.5x) + 0.5x` is odd, so the x^6 term buys no bias (bias^2 0.00122
for both 5 and 6 at n=300) and only adds variance: degree 5 beats 6 at n=30
(0.0556 vs 0.0660), n=300 (0.00532 vs 0.00594) and n=1000. The lesson's sweep
draws fresh datasets per degree, and that Monte Carlo noise is larger than the
5-vs-6 gap, so its "optimal degree 6" is a lucky draw.

**FINDING: the optimum does shift, but past n=300.** On common data degree 5
wins at n=30; at n=300 degrees 5 and 7 differ by 0.4% (0.00532 vs 0.00530); at
n=1000 degree 7 wins clearly, 0.00154 vs 0.00231. More data lets the next
odd term pay for its variance.

**CONTROL: more data leaves bias alone** -- bias^2 0.4310 -> 0.4379 at degree 1
and 0.00117 -> 0.00117 at degree 5, as the lesson says.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "10-bias-variance"
SWEEP, COMMON, DRAWS = list(range(1, 16)), list(range(1, 10)), 300


def common_sweep(ref, n_train, draws=DRAWS):
    """(optimum, bias^2 + variance, bias^2) per degree, all degrees on the same datasets."""
    x_test = np.linspace(-2.5, 2.5, 100)
    truth = ref.true_function(x_test)
    preds = {d: [] for d in COMMON}
    for seed in range(draws):
        x, y = ref.generate_data(n_samples=n_train, noise_std=0.5, seed=seed)
        for d in COMMON:
            preds[d].append(ref.predict_polynomial(x_test, ref.fit_polynomial(x, y, d)))
    bias, total = {}, {}
    for d, p in preds.items():
        p = np.array(p)
        bias[d] = np.mean((p.mean(axis=0) - truth) ** 2)
        total[d] = bias[d] + np.mean(p.var(axis=0))
    return min(total, key=total.get), total, bias


def summarise(ref, run, degrees=(1, 5, 6, 15)):
    """The lesson's optimum and (bias^2, variance) at a few degrees."""
    return ref.find_optimal(run), {d: (run[d]["bias_sq"], run[d]["variance"]) for d in degrees}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bias_variance")
    lesson = {n: summarise(ref, ref.bias_variance_decomposition(SWEEP, n_train=n))
              for n in (30, 300)}
    common = {n: common_sweep(ref, n) for n in (30, 300, 1000)}
    return {
        "lesson_best": {n: best for n, (best, _) in lesson.items()},
        "lesson": {n: table for n, (_, table) in lesson.items()},
        "common_best": {n: best for n, (best, _, _) in common.items()},
        "total": {n: total for n, (_, total, _) in common.items()},
        "bias": (common[300][2][5], common[300][2][6]),
    }


def verify(result):
    lb, lesson, cb = result["lesson_best"], result["lesson"], result["common_best"]
    total, (b5, b6) = result["total"], result["bias"]
    v30, v300 = lesson[30][6][1], lesson[300][6][1]
    h30, h300 = lesson[30][15][1], lesson[300][15][1]
    tie = abs(total[300][7] / total[300][5] - 1)
    return [
        practice.Check(
            "ANSWER: variance falls ~14x at the optimum, and the lesson's optimum stays at 6",
            all((lb == {30: 6, 300: 6}, 10 < v30 / v300 < 20, h30 / h300 > 1e5)),
            f"degree-6 variance {v30:.4f} -> {v300:.4f} ({v30 / v300:.1f}x for 10x the data); "
            f"degree 15 falls {h30 / h300:.1e}x ({h30:.3g} -> {h300:.4f}), because at n=30 it "
            f"is near-interpolating. find_optimal picks degree {lb[30]} at both sizes",
        ),
        practice.Check(
            "FINDING: degree 6 never wins on common data; the target is odd, so 6 is 5 plus noise",
            all((*[t[5] < t[6] for t in total.values()], cb[30] == 5, abs(b5 - b6) < 2e-4)),
            f"fit to the same {DRAWS} datasets, degree 5 beats 6 at n=30 ({total[30][5]:.4f} vs "
            f"{total[30][6]:.4f}) and n=300 ({total[300][5]:.5f} vs {total[300][6]:.5f}); their "
            f"bias^2 at n=300 is {b5:.5f} vs {b6:.5f}, since sin(1.5x) + 0.5x has no even part. "
            "The lesson's sweep draws new data per degree and its noise exceeds that gap",
        ),
        practice.Check(
            "FINDING: the true optimum is 5 at n=30, a 5/7 tie at n=300, and 7 by n=1000",
            all((cb[30] == 5, tie < 0.02, cb[1000] == 7, total[1000][5] > 1.3 * total[1000][7])),
            f"on common data degree 5 wins at n=30 ({total[30][5]:.4f} vs degree 7's "
            f"{total[30][7]:.4f}); at n=300 degrees 5 and 7 differ by {tie:.1%} "
            f"({total[300][5]:.5f} vs {total[300][7]:.5f}); at n=1000 degree 7 wins, "
            f"{total[1000][7]:.5f} vs {total[1000][5]:.5f}: the shift is real but lands later",
        ),
        practice.Check(
            "CONTROL: more data leaves bias alone",
            all(abs(lesson[30][d][0] / lesson[300][d][0] - 1) < 0.05 for d in (1, 5)),
            f"bias^2 is {lesson[30][1][0]:.4f} -> {lesson[300][1][0]:.4f} at degree 1 and "
            f"{lesson[30][5][0]:.5f} -> {lesson[300][5][0]:.5f} at degree 5",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
