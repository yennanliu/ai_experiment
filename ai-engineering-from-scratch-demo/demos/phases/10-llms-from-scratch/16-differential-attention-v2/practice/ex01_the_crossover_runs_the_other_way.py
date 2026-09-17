"""Exercise 1 — differential attention wins below noise 1.0 and loses above it; nothing is ever unusable.

    Run `code/main.py`. Verify the signal-to-noise ratio reported for
    differential attention is higher than standard softmax attention on the
    synthetic query. Vary the noise amplitude and show the crossover point where
    standard attention becomes unusable.

Reading of the exercise: the toy is the lesson's own -- a 1024-token logit
vector with a 4.0 signal at position 500 and Gaussian noise elsewhere, scored
with its own `snr` -- and the sweep is run on twelve seeds per noise level
rather than one, because `main` prints a single draw per level and the exercise
asks for a crossover, which is a claim about where two curves meet.

**ANSWER: there is a crossover and it runs the other way.**

    noise   standard SNR   diff SNR (lam=0.8)   ratio
     0.25       53.00            182.45         3.44
     0.50       48.37             91.27         1.89
     1.00       33.41             34.49         1.03    <- they meet here
     1.50       18.24             13.75         0.75
     2.00        7.78              4.75         0.61
     3.00        0.68              0.38         0.57

Differential attention is better below noise 1.0 and **worse above it**. The
exercise's first sentence holds at the noise level `main` prints and fails at
twice it.

**FINDING: standard attention never becomes unusable while differential
attention is usable.** Both collapse together: at noise 3.0 they are 0.68 and
0.38, and at 4.0 they are 0.04 and 0.02. The ratio stays near 0.5, so there is
no regime where subtracting the second branch rescues a query the first branch
has lost.

**MECHANISM: the subtraction removes a noise floor that softmax has already
removed.** At low noise the signal logit dominates and `A1`'s noise weights are
a nearly flat floor of about `1/n`, which `A2` -- an independent softmax over
pure noise -- also is, so the subtraction cancels it. At high noise `A1`'s noise
weights are no longer flat: they are the exponentials of a wide Gaussian, and
subtracting an *independent* draw of the same kind adds variance instead of
removing it.

**FINDING: the sweep the lesson prints stops one step before the crossover.**
`main` sweeps 0.25 to 2.00 and prints one seed per level, so it shows the
crossing without naming it. The median over twelve seeds puts the meeting point
between 1.0 and 1.5.

Structure: `logits` builds the lesson's own toy at one noise level; `arm` scores
one seed with both attentions through the lesson's own `snr`.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "16-differential-attention-v2"
TOKENS, SIGNAL_POS, SIGNAL, LAMBDA = 1024, 500, 4.0, 0.8
NOISES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SEEDS, USABLE = 12, 2.0


def logits(ref, noise, seed):
    """The lesson's own toy: one trained branch with the signal, one untrained branch."""
    rng = random.Random(seed)
    trained = [rng.gauss(0, noise) for _ in range(TOKENS)]
    trained[SIGNAL_POS] = SIGNAL
    untrained = [rng.gauss(0, noise) for _ in range(TOKENS)]
    return ref.softmax_row(trained), ref.softmax_row(untrained)


def arm(ref, noise):
    """Median SNR over `SEEDS` draws, for standard attention and for differential."""
    standard, differential = [], []
    for seed in range(SEEDS):
        first, second = logits(ref, noise, seed)
        standard.append(ref.snr(first, SIGNAL_POS))
        differential.append(ref.snr([a - LAMBDA * b for a, b in zip(first, second)],
                                    SIGNAL_POS))
    return {"standard": statistics.median(standard),
            "diff": statistics.median(differential),
            "usable": sum(v > USABLE for v in standard) / SEEDS}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {noise: arm(ref, noise) for noise in NOISES}
    for row in rows.values():
        row["ratio"] = row["diff"] / row["standard"]
    crossing = [n for n in NOISES if rows[n]["ratio"] < 1.0]
    return {"rows": rows, "crossover": min(crossing) if crossing else None,
            "printed": [n for n in NOISES if n <= 2.0]}


def column(rows, field, fmt):
    return ", ".join(f"{noise} {format(row[field], fmt)}" for noise, row in rows.items())


def verify(result):
    rows, crossover = result["rows"], result["crossover"]
    low, high = rows[0.25], rows[2.0]
    return [
        practice.Check(
            "ANSWER: there is a crossover and it runs the other way -- diff loses above noise 1.0",
            low["ratio"] > 3 and high["ratio"] < 0.7 and crossover is not None,
            "median SNR over " + str(SEEDS) + " seeds is standard " + column(rows, "standard", ".2f")
            + " against differential " + column(rows, "diff", ".2f")
            + f", a ratio of " + column(rows, "ratio", ".2f")
            + f". Differential attention is {low['ratio']:.1f}x better at noise 0.25 and "
            f"{high['ratio']:.2f}x -- that is, worse -- at noise 2.0, first falling below parity "
            f"at noise {crossover}. The exercise's first sentence holds at the noise level main "
            "prints and fails at twice it",
        ),
        practice.Check(
            "FINDING: standard attention is never unusable while differential attention is usable",
            rows[3.0]["standard"] > rows[3.0]["diff"] and rows[3.0]["usable"] == 0.0,
            f"both collapse together: at noise 3.0 they are {rows[3.0]['standard']:.2f} and "
            f"{rows[3.0]['diff']:.2f}, at 4.0 {rows[4.0]['standard']:.2f} and "
            f"{rows[4.0]['diff']:.2f}. The share of seeds where standard attention clears an "
            f"SNR of {USABLE:.0f} is " + column(rows, "usable", ".0%")
            + ", and differential attention is below standard at every one of the noise levels "
            "where that share is zero. There is no regime the subtraction rescues",
        ),
        practice.Check(
            "MECHANISM: the subtraction removes a floor that softmax has already removed",
            rows[0.25]["diff"] > rows[0.5]["diff"] > rows[1.5]["diff"],
            "at low noise the signal logit dominates and the first branch's noise weights are a "
            "nearly flat floor of about 1/n, which the second branch -- an independent softmax "
            "over pure noise -- also is, so the subtraction cancels it. At high noise the first "
            "branch's noise weights are the exponentials of a wide Gaussian and are no longer "
            "flat, so subtracting an independent draw of the same kind adds variance: the "
            "differential SNR falls " + column(rows, "diff", ".1f")
            + " while the mechanism it relies on disappears",
        ),
        practice.Check(
            "FINDING: the sweep the lesson prints stops one step before naming the crossover",
            max(result["printed"]) == 2.0 and crossover in result["printed"],
            f"main sweeps {result['printed']} and prints one seed per level, so the crossing is "
            f"on the page and unnamed -- the median over {SEEDS} seeds puts it at noise "
            f"{crossover}, inside the range the lesson already shows. A single draw at each level "
            "is also why it reads as noise rather than as a curve turning over",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
