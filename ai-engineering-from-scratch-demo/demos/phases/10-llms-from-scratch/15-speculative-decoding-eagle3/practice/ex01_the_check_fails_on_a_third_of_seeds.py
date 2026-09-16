"""Exercise 1 — the statistic exceeds its critical value on 29% of seeds, and the sampler is not why.

    Run `code/main.py`. Confirm the chi-square statistic on the Leviathan
    distribution check stays below the 95% critical value on 50,000 samples.

Reading of the exercise: "confirm it stays below" describes a deterministic
property, and the statistic is a random variable, so it is run on 24 seeds
rather than one. The same 24 runs are then scored a second way -- against the
exact verifier distribution rather than against a second random sample -- because
`distribution_check` returns two sampled histograms and `chi_square` treats one
of them as the expectation.

**ANSWER: the check passes on the lesson's own seed and fails on 7 of 24.** The
statistic exceeds the 14.07 critical value **29%** of the time, with a mean of
**12.34** against the 7.0 a chi-square with 7 degrees of freedom should average.
Nothing is wrong with the sampler.

**MECHANISM: the test compares two random samples and calls one of them the
expectation.** `distribution_check` draws 50,000 speculative tokens *and* 50,000
direct ones; `chi_square` rescales the second to the first's total and treats it
as `E`. Both sides carry sampling noise, which roughly doubles the variance of
the statistic, so a nominal 5% test rejects far more often than 5%.

**FINDING: scored against the exact `q`, the same runs behave.** Mean **7.01**
against the theoretical 7.0, and **1 of 24** above the critical value -- 4%,
which is what a 95% threshold is supposed to give. The verifier distribution is
known exactly here; it is a list of eight floats at the top of `main`. Using it
costs nothing and makes the exercise's claim true.

**FINDING: Leviathan's guarantee is exact, and this is what confirming it should
look like.** Across the 24 seeds the speculative tokens match `q` to a mean
chi-square indistinguishable from its own degrees of freedom. The scheme in
`spec_step` -- accept with `min(1, q/p)`, otherwise draw from the normalised
positive part of `q - p` -- is unbiased by construction, and Exercise 3 shows
how little it takes to break that.

Structure: `chi_squares` runs the lesson's own `distribution_check` on one seed
and scores it both ways; `exceedance` is the fraction of seeds above the
critical value.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "15-speculative-decoding-eagle3"
Q = [0.30, 0.22, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03]
TRIALS, SEEDS, CRITICAL = 50_000, 24, 14.07
DOF = len(Q) - 1


def chi_squares(ref, draft, seed):
    """One run of the lesson's own check, scored against the sample and against q."""
    spec, direct = ref.distribution_check(Q, draft, TRIALS, random.Random(seed))
    return (ref.chi_square(spec, direct),
            ref.chi_square(spec, [TRIALS * qi for qi in Q]))


def exceedance(values):
    return sum(v > CRITICAL for v in values) / len(values)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    draft = ref.perturb(Q, 0.02, random.Random(2))
    scored = [chi_squares(ref, draft, seed) for seed in range(SEEDS)]
    against_sample = [pair[0] for pair in scored]
    against_exact = [pair[1] for pair in scored]
    return {
        "sample": {"values": against_sample, "mean": statistics.fmean(against_sample),
                   "over": exceedance(against_sample)},
        "exact": {"values": against_exact, "mean": statistics.fmean(against_exact),
                  "over": exceedance(against_exact)},
        "lesson_seed": chi_squares(ref, draft, 42)[0],
        "dof": DOF,
        "kl": ref.kl(Q, draft),
    }


def verify(result):
    sample, exact = result["sample"], result["exact"]
    return [
        practice.Check(
            "ANSWER: the check passes on the lesson's own seed and fails on 7 of 24",
            result["lesson_seed"] < CRITICAL and sample["over"] > 0.2,
            f"on seed 42 the statistic is {result['lesson_seed']:.2f}, below the "
            f"{CRITICAL} critical value, so the run the lesson prints does confirm what the "
            f"exercise asks. Across {SEEDS} seeds it exceeds that value "
            f"{sample['over']:.0%} of the time, with a mean of {sample['mean']:.2f} against the "
            f"{result['dof']}.0 a chi-square with {result['dof']} degrees of freedom should "
            "average. Nothing is wrong with the sampler",
        ),
        practice.Check(
            "MECHANISM: the test compares two random samples and calls one the expectation",
            sample["mean"] > 1.5 * result["dof"],
            f"distribution_check draws {TRIALS:,} speculative tokens and {TRIALS:,} direct ones, "
            f"and chi_square rescales the second to the first's total and treats it as E. Both "
            f"sides carry sampling noise, which roughly doubles the variance of the statistic: "
            f"the observed mean is {sample['mean']:.2f} where the distribution it is being "
            f"compared against has mean {result['dof']}.0, so a nominal 5% test rejects "
            f"{sample['over']:.0%} of the time",
        ),
        practice.Check(
            "FINDING: scored against the exact q, the same runs behave",
            abs(exact["mean"] - result["dof"]) < 1.0 and exact["over"] < 0.1,
            f"the same {SEEDS} speculative histograms scored against q itself give a mean of "
            f"{exact['mean']:.2f} against the theoretical {result['dof']}.0, and "
            f"{exact['over']:.0%} above the critical value -- which is what a 95% threshold is "
            "supposed to give. The verifier distribution is known exactly here: it is a list of "
            "eight floats at the top of main. Using it costs nothing and makes the exercise's "
            "claim true",
        ),
        practice.Check(
            "FINDING: Leviathan's guarantee is exact, and this is what confirming it looks like",
            max(exact["values"]) < 3 * CRITICAL and result["kl"] > 0,
            f"across {SEEDS} seeds the speculative tokens match q to a worst-case chi-square of "
            f"{max(exact['values']):.2f}, with a draft whose KL from the verifier is "
            f"{result['kl']:.4f} -- the draft is genuinely wrong and the output is not. Accept "
            "with min(1, q/p), otherwise draw from the normalised positive part of q - p: "
            "unbiased by construction, and Exercise 3 shows how little it takes to break it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
