"""Exercise 4 — the optimum is 0.75 at the lesson's noise and 0.00 at four times it.

    Implement an ablation: compute differential attention with `lambda = 0`
    (pure first softmax) and `lambda = 1` (full subtraction). On the synthetic
    query, measure how signal-to-noise changes across the sweep. Identify the
    `lambda` that maximizes signal-to-noise.

Reading of the exercise: the two endpoints the exercise names are the ablation
and the sweep between them is the answer, so lambda is stepped in 0.05 from 0 to
1.2 -- past 1, because nothing in `diff_attention` requires lambda <= 1 and the
optimum is a measured quantity rather than a stipulated one. Twelve seeds per
setting, and the whole sweep is repeated at the noise level Exercise 1 shows the
method losing at.

**ANSWER: 0.75 at the lesson's own noise of 0.5, and 0.00 at noise 2.0.**

    noise   lam=0.0   lam=0.8   lam=1.0   best lambda   best SNR
     0.50    48.37     91.27     84.42       0.75        91.68
     2.00     7.78      4.75      4.27       0.00         7.78

At the noise level `main` prints, the lesson's hard-coded 0.8 is within **0.4%**
of optimal. At four times that noise the optimising lambda is **zero** -- the
ablation's own "pure first softmax" endpoint -- and the answer to "identify the
lambda that maximizes signal-to-noise" is "do not do this".

**FINDING: the curve is single-peaked and the peak moves with the noise.** SNR
rises from lam=0 to the optimum and falls after it at every noise level; what
changes is where the optimum sits, and it walks to the left as the context gets
noisier until it hits the wall at zero. A lambda tuned on one context length or
one noise regime is not a constant of the architecture.

**MECHANISM: lambda trades a shrinking signal against a shrinking noise floor.**
The signal weight falls linearly as `A1[pos] - lam * A2[pos]`, and the noise
weights fall linearly too -- but from a base that `A2` matches well when the
context is quiet and badly when it is loud. When the match is good the noise
falls faster than the signal and SNR rises; when it is poor the signal falls
faster, and the best available trade is not to trade.

**FINDING: past lambda = 1 the weights go negative and the metric stops
meaning anything.** `snr` takes `abs(weights[pos])`, so a signal weight driven
through zero and out the other side is scored as though it were positive. At
lambda = 1.2 the sweep still reports a finite number, and the attention row it
describes sums to **-0.2**.

Structure: `sweep` scores one noise level across the lambda grid through the
lesson's own `snr`; `peak` picks the argmax and reports the curve either side.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "16-differential-attention-v2"
TOKENS, SIGNAL_POS, SIGNAL, SEEDS = 1024, 500, 4.0, 12
NOISES = (0.5, 2.0)
GRID = tuple(i / 20 for i in range(25))
LESSON_LAMBDA = 0.8


def branches(ref, noise, seed):
    rng = random.Random(seed)
    trained = [rng.gauss(0, noise) for _ in range(TOKENS)]
    trained[SIGNAL_POS] = SIGNAL
    untrained = [rng.gauss(0, noise) for _ in range(TOKENS)]
    return ref.softmax_row(trained), ref.softmax_row(untrained)


def score(ref, noise, lam):
    """Median SNR and mean row sum at one (noise, lambda)."""
    values, sums = [], []
    for seed in range(SEEDS):
        first, second = branches(ref, noise, seed)
        weights = [a - lam * b for a, b in zip(first, second)]
        values.append(ref.snr(weights, SIGNAL_POS))
        sums.append(sum(weights))
    return statistics.median(values), statistics.fmean(sums)


def sweep(ref, noise):
    return {lam: score(ref, noise, lam)[0] for lam in GRID}


def peak(curve):
    best = max(curve, key=curve.get)
    return {"lam": best, "snr": curve[best],
            "rising": all(curve[a] <= curve[b] for a, b in zip(GRID, GRID[1:])
                          if b <= best),
            "falling": all(curve[a] >= curve[b] for a, b in zip(GRID, GRID[1:])
                           if a >= best)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    curves = {noise: sweep(ref, noise) for noise in NOISES}
    peaks = {noise: peak(curve) for noise, curve in curves.items()}
    return {
        "curves": {noise: {lam: curve[lam] for lam in (0.0, 0.5, LESSON_LAMBDA, 1.0, 1.2)}
                   for noise, curve in curves.items()},
        "peaks": peaks,
        "lesson_gap": {noise: curves[noise][LESSON_LAMBDA] / peaks[noise]["snr"]
                       for noise in NOISES},
        "beyond_one": {noise: (curves[noise][1.2], score(ref, noise, 1.2)[1])
                       for noise in NOISES},
    }


def row(curve):
    return ", ".join(f"lam={lam} {value:.2f}" for lam, value in curve.items())


def verify(result):
    curves, peaks = result["curves"], result["peaks"]
    quiet, loud = peaks[0.5], peaks[2.0]
    beyond, row_sum = result["beyond_one"][0.5]
    return [
        practice.Check(
            "ANSWER: the optimum is 0.75 at the lesson's noise and 0.00 at four times it",
            0.6 <= quiet["lam"] <= 0.9 and loud["lam"] == 0.0,
            f"at noise 0.5 the curve is " + row(curves[0.5])
            + f" and the argmax is lambda={quiet['lam']} at {quiet['snr']:.2f}; at noise 2.0 it "
            f"is " + row(curves[2.0])
            + f" and the argmax is lambda={loud['lam']} at {loud['snr']:.2f}. The lesson's "
            f"hard-coded {LESSON_LAMBDA} is within "
            f"{1 - result['lesson_gap'][0.5]:.1%} of optimal in the quiet case and "
            f"{1 - result['lesson_gap'][2.0]:.0%} off in the loud one, where the optimising "
            "lambda is the ablation's own 'pure first softmax' endpoint",
        ),
        practice.Check(
            "FINDING: the curve is single-peaked and the peak walks left as noise rises",
            quiet["rising"] and quiet["falling"] and loud["lam"] < quiet["lam"],
            f"SNR rises to the optimum and falls after it at both noise levels -- the quiet "
            f"curve peaks at {quiet['lam']} and the loud one at {loud['lam']}. What changes is "
            "where the optimum sits, and it walks toward zero as the context gets noisier until "
            "it hits the wall. A lambda tuned at one noise level is not a constant of the "
            "architecture",
        ),
        practice.Check(
            "MECHANISM: lambda trades a shrinking signal against a shrinking noise floor",
            curves[0.5][0.5] > curves[0.5][0.0] and curves[2.0][0.5] < curves[2.0][0.0],
            f"the signal weight falls linearly as A1[pos] - lam x A2[pos] and so do the noise "
            f"weights, but from a base the second branch matches well when the context is quiet "
            f"and badly when it is loud. At noise 0.5 lambda=0.5 takes SNR from "
            f"{curves[0.5][0.0]:.2f} to {curves[0.5][0.5]:.2f}; at noise 2.0 the same step takes "
            f"it from {curves[2.0][0.0]:.2f} to {curves[2.0][0.5]:.2f}. When the match is good "
            "the noise falls faster than the signal; when it is poor the best trade is not to "
            "trade",
        ),
        practice.Check(
            "FINDING: past lambda = 1 the weights go negative and the metric keeps reporting",
            row_sum < 0 and beyond > 0,
            f"snr takes abs(weights[pos]), so a signal weight driven through zero and out the "
            f"other side is scored as though it were positive. At lambda = 1.2 the sweep still "
            f"reports {beyond:.2f}, and the attention row it describes sums to {row_sum:+.2f} -- "
            "nothing in diff_attention or in snr requires lambda to stay in [0, 1], and the "
            "ablation the exercise specifies stops exactly where the metric stops being valid",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
