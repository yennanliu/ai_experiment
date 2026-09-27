"""Exercise 2 — heavy-tailed label noise never moves the peak; heavy-tailed proxy error makes best-of-N collapse.

    Modify the noise distribution from Gaussian to a Student-t with low degrees
    of freedom (heavy-tailed). Keep the proxy RM training setup unchanged. What
    changes about the peak location and post-peak collapse?

Reading of the exercise: "the noise" is the label noise in `train_proxy`,
which the reference already switches to its own `student_t(3.0)`; "low
degrees of freedom" is swept over df = 3, 2, 1 by swapping the df that
`train_proxy` passes, with n = 300 samples and least squares unchanged, 200
seeds per setting. Because exercise 1 shows this model cannot collapse, the
answer also builds the condition the lesson's "Catastrophic Goodhart" section
actually names -- heavy-tailed error in the proxy's score of each output --
and measures best-of-N selection under it.

**ANSWER: nothing changes about the peak, because there is none.** On the
shipped seed the Student-t(3) proxy prints its "peak" at sqrt(KL) = 2.828
with gold 6.332, against 6.374 for the Gaussian 300-sample proxy -- the same
grid point. Over 200 seeds the mean cosine between proxy and gold weights is
0.9952 (Gaussian), 0.9864 (df 3), 0.9519 (df 2) and 0.4156 (df 1), and no
seed at any df has an interior peak. What heavier tails do is make the fit
worse: at df = 1 (Cauchy) 40 of 200 proxies point away from gold, and their
gold "peaks" at the origin and falls in a straight line.

**FINDING: the main.py claim "Heavy-tailed noise moves the peak closer to
the origin" is false in its own model.** Label noise, however heavy, is
averaged by least squares into an error in the proxy's *slope*; the proxy's
error on any output is then linear in that output and Gaussian under the
policy, which is not the heavy-tailed condition.

**FINDING: put the heavy tail on the proxy's score of each output and
best-of-N peaks, then collapses.** With proxy = gold + error per candidate,
gold ~ N(0, |w_gold|^2) as for the reference's features, 600 trials: under
Gaussian error, gold of the chosen output rises at every n, to 4.42 at n =
1024. Under the reference's `student_t(3.0)` error it peaks at 2.01 at n =
64 (Gao's best-of-N KL: sqrt(KL) = 1.782) and falls to 0.86 at n = 1024.
The more the optimizer selects, the more it selects for error.

Structure: `label_noise()` reruns `train_proxy` under a swapped df with
`ref.random` seeded (both restored after); `best_of_n()` draws candidates
with the reference's own `gauss` and `student_t`.
"""

from __future__ import annotations

import math
import pathlib
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "02-reward-hacking-goodhart"
HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
SEEDS, N_LABELS, TRIALS = 200, 300, 600
NS = (1, 4, 16, 64, 256, 1024)


def label_noise(ref, df):
    """(mean cosine, proxies pointing away, gold peak positions seen) over SEEDS."""
    saved = ref.random, ref.student_t
    ref.random = random.Random(df or 0)
    if df:
        ref.student_t = lambda _df: saved[1](df)
    try:
        rms = [ref.train_proxy(N_LABELS, "student_t" if df else "gauss") for _ in range(SEEDS)]
    finally:
        ref.random, ref.student_t = saved
    cos = [EX01.cosine(ref, rm.w) for rm in rms]
    peaks = {round(max(ref.kl_constrained_policy_sweep(rm, EX01.BUDGETS), key=lambda r: r[2])[0], 3)
             for rm in rms}
    return round(sum(cos) / SEEDS, 4), sum(c < 0 for c in cos), sorted(peaks)


def best_of_n(ref, error):
    """Mean gold of the proxy-best of n candidates, proxy = gold + error()."""
    sd, saved, ref.random = math.sqrt(ref.dot(ref.GOLD_W, ref.GOLD_W)), ref.random, random.Random(0)
    try:
        curve = []
        for n in NS:
            picks = [max(((sd * ref.gauss(), error()) for _ in range(n)), key=sum)[0]
                     for _ in range(TRIALS)]
            curve.append(round(sum(picks) / TRIALS, 2))
        return curve
    finally:
        ref.random = saved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log = EX01.shipped(ref)
    peaks = EX01.parse(log)[0]
    heavy = best_of_n(ref, lambda: ref.student_t(3.0))
    top = heavy.index(max(heavy))
    return {
        "shipped": {"gauss_300": peaks[1][:2], "t3_300": peaks[4][:2]},
        "labels": {df or "gauss": label_noise(ref, df) for df in (None, 3, 2, 1)},
        "claim": "Heavy-tailed noise moves the peak closer to the origin" in " ".join(log.split()),
        "bon_gauss": best_of_n(ref, ref.gauss), "bon_t3": heavy,
        "t3_peak": (NS[top], round(math.sqrt(math.log(NS[top]) - 1 + 1 / NS[top]), 3)),
    }


def verify(result):
    labels, gauss, heavy = result["labels"], result["bon_gauss"], result["bon_t3"]
    return [
        practice.Check(
            "ANSWER: nothing changes about the peak, because there is none",
            (result["shipped"], labels) == (
                {"gauss_300": (2.828, 6.374), "t3_300": (2.828, 6.332)},
                {"gauss": (0.9952, 0, [2.828]), 3: (0.9864, 0, [2.828]),
                 2: (0.9519, 0, [2.828]), 1: (0.4156, 40, [0.0, 2.828])}),
            f"shipped (sqrt(KL), gold) at the printed peak: {result['shipped']}; over "
            f"{SEEDS} seeds (mean cos, proxies pointing away, peak positions): {labels}",
        ),
        practice.Check(
            "FINDING: main.py's 'heavy-tailed noise moves the peak closer' is false in its model",
            (result["shipped"]["t3_300"][0], labels[3][2], result["claim"])
            == (result["shipped"]["gauss_300"][0], [2.828], True),
            "Student-t(3) label noise: every seed's gold maximum is the last grid point",
        ),
        practice.Check(
            "FINDING: heavy-tailed error on each output makes best-of-N peak, then collapse",
            (sorted(gauss), gauss[-1], result["t3_peak"], max(heavy), heavy[-1])
            == (gauss, 4.42, (64, 1.782), 2.01, 0.86),
            f"gold of the chosen output at n = {NS}: Gaussian error {gauss}, Student-t(3) "
            f"error {heavy}; peak at (n, sqrt(KL)) = {result['t3_peak']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
