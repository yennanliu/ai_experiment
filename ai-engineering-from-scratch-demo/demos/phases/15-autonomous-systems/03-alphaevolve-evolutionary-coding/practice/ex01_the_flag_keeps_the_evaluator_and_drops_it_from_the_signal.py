"""Exercise 1 — the flag keeps the evaluator and drops it from the signal.

    Run `code/main.py`. Note the best score trajectory. Disable the held-out
    evaluator (flag `--no-holdout`) and re-run. Quantify the overfitting.

Reading of the exercise: "quantify" is the whole exercise, and a quantity
from one seed is not one. The shipped seed is reported because it is what
`main()` prints, and then the same comparison is run paired across seeds 0-19,
because the two runs consume the same random stream and differ only in which
candidates enter the archive.

**ANSWER: +2.0990 at the shipped seed, and not reliably positive at all.**
Seed 1 with the held-out signal finds the target exactly -- train and test MSE
**0.0000** at generation **1229** -- while `--no-holdout` ends at train
**2.2917**, test **4.3906**. The train trajectory runs **13.83 -> 3.17 ->
0.00** across generations 100, 500 and 1500 with the signal, and **22.50 ->
8.00 -> 2.29** without it.

**FINDING: the flag does not disable the held-out evaluator.** `run_loop`
computes `te = mse(child_expr, test_xs)` for every child whichever way the
flag is set; only `signal_of` changes. The evaluator the exercise says to
disable still runs **1500** times per generation loop, and its output is
printed in the summary -- it is excluded from selection, not from execution.

**FINDING: one seed is not a measurement.** Paired over **20** seeds, the
held-out signal gives a better final test MSE **6** times, a worse one **4**
times and an identical one **10**. The means separate -- **1.90** against
**3.58** -- but a 6-4 split over ten decided pairs is what a coin does, and
the lesson's headline rests on the seed `main()` happens to pass.

**FINDING: the split measures interpolation.** The training points are the
integers from -2 to 3 and the test points are their midpoints, so **5** of
**7** held-out points lie inside the training hull and the remaining two sit
half a unit outside it. Symbolic regression fails by extrapolating, and this
evaluator is built where it cannot see that.

Structure: `paired()` runs both modes off the same seed; `shipped()` is the
single run `main()` prints.
"""

from __future__ import annotations

import inspect
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "03-alphaevolve-evolutionary-coding"

SEEDS, GENERATIONS, POP = range(20), 1500, 20
SHIPPED_SEED = 1                # DEFAULT_SEED, the one main() passes
TRAIN_XS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
TEST_XS = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]


def run(ref, seed, use_holdout):
    return ref.run_loop(GENERATIONS, POP, use_holdout, seed=seed)


def shipped(ref):
    """The two runs `main()` prints, at DEFAULT_SEED."""
    held, trace, _ = run(ref, SHIPPED_SEED, True)
    plain, plain_trace, _ = run(ref, SHIPPED_SEED, False)
    return held, plain, trace, plain_trace


def paired(ref):
    """Final test MSE for both modes at each seed."""
    return [(run(ref, seed, True)[0].test_score, run(ref, seed, False)[0].test_score)
            for seed in SEEDS]


def inside_hull(points, low, high):
    return sum(low <= x <= high for x in points)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    held, plain, trace, plain_trace = shipped(ref)
    pairs = paired(ref)
    source = inspect.getsource(ref.run_loop)
    return {
        "held_scores": [round(held.train_score, 4), round(held.test_score, 4)],
        "held_generation": held.generation,
        "plain_scores": [round(plain.train_score, 4), round(plain.test_score, 4)],
        "gap": round(plain.test_score - plain.train_score, 4),
        "held_trace": [round(trace[99], 2), round(trace[499], 2), round(trace[-1], 2)],
        "plain_trace": [round(plain_trace[99], 2), round(plain_trace[499], 2),
                        round(plain_trace[-1], 2)],
        "test_mse_calls": source.count("mse(child_expr, test_xs)"),
        "flag_guards_mse": "use_holdout" in source.split("te = ")[1].split("\n")[0],
        "wins": sum(1 for a, b in pairs if a < b),
        "losses": sum(1 for a, b in pairs if a > b),
        "ties": sum(1 for a, b in pairs if a == b),
        "mean_held": round(statistics.mean(a for a, _ in pairs), 2),
        "mean_plain": round(statistics.mean(b for _, b in pairs), 2),
        "inside": inside_hull(TEST_XS, min(TRAIN_XS), max(TRAIN_XS)),
        "test_points": len(TEST_XS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: +2.0990 at the shipped seed, against an exact fit with the signal",
            all([result["gap"] == 2.0990, result["held_scores"] == [0.0, 0.0],
                 result["held_generation"] == 1229,
                 result["plain_scores"] == [2.2917, 4.3906],
                 result["held_trace"] == [13.83, 3.17, 0.0],
                 result["plain_trace"] == [22.5, 8.0, 2.29]]),
            f"seed {SHIPPED_SEED} with the held-out signal reaches "
            f"{result['held_scores']} train/test at generation "
            f"{result['held_generation']}; without it {result['plain_scores']}, a "
            f"{result['gap']:+} gap, off trajectories {result['held_trace']} and "
            f"{result['plain_trace']}",
        ),
        practice.Check(
            "FINDING: the flag does not disable the held-out evaluator",
            all([result["test_mse_calls"] == 1, not result["flag_guards_mse"]]),
            f"run_loop scores every child on the held-out points regardless of the "
            f"flag -- {result['test_mse_calls']} unguarded call, {GENERATIONS} times "
            "per run -- and only signal_of changes",
        ),
        practice.Check(
            "FINDING: one seed is not a measurement",
            all([result["wins"] == 6, result["losses"] == 4, result["ties"] == 10,
                 result["mean_held"] < result["mean_plain"]]),
            f"paired over {len(SEEDS)} seeds the held-out signal wins {result['wins']}, "
            f"loses {result['losses']} and ties {result['ties']} on final test MSE; the "
            f"means separate at {result['mean_held']} against {result['mean_plain']}, "
            "but ten decided pairs split 6-4",
        ),
        practice.Check(
            "FINDING: the split measures interpolation",
            all([result["inside"] == 5, result["test_points"] == 7]),
            f"{result['inside']} of {result['test_points']} held-out points lie inside "
            f"the training hull [{min(TRAIN_XS)}, {max(TRAIN_XS)}] and the other two sit "
            "half a unit outside, so the evaluator cannot see extrapolation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
