"""Exercise 1 — the test it specifies rejects a theorem that is exactly true.

    **Easy.** Run `code/main.py`. Confirm the speculative token distribution
    matches the verifier's direct-sample distribution on 50,000 tokens within
    chi-square p > 0.05.

Reading of the exercise: the claim is exact, so it is checked exactly first --
the marginal of `spec_step_one_token` is written out in closed form and compared
with `q` -- and only then is the sampling test the exercise names run, 40 times,
to see what it does to a statement that is true.

**ANSWER: the theorem holds to 1.4e-17, with no sampling at all.**
`P(token i) = p_i * min(1, q_i/p_i) + P(reject) * res_i` comes to `q_i` for every
`i`, where `P(reject) = 1 - sum(min(p, q)) = 0.040306` -- the total variation
distance between the two distributions, exactly.

**FINDING: the statistic `main()` prints runs twice as hot as it should.**
`chi_square(spec_counts, direct_counts)` compares one sample against *another
sample*, so the "expected" counts carry their own multinomial noise and the
statistic carries both. Over 40 repeats its mean is **12.79** where a chi-square
on 7 degrees of freedom has mean 7. Against the true expectation `50000 * q` the
same draws average **6.31**, which is right.

**FINDING: so the exercise's own criterion fails 43% of the time on a true
theorem.** The `p > 0.05` critical value at 7 df is **14.067**. The printed
statistic exceeds it in **17 of 40** repeats; the correct statistic exceeds it in
**1 of 40**, the nominal 2.5%. A reader following the instruction would conclude
that Leviathan's theorem fails on almost half their runs.

**FINDING: and the threshold the lesson ships can never fire.** `main()` prints
PASS when the statistic is under **30** -- about p = 1e-4 at 7 df -- and across
40 repeats it is exceeded **0 times**. The test as written cannot fail, and the
test as specified fails half the time; neither is measuring the theorem.

Structure: `marginal` is the closed-form distribution of one speculative step;
`sweep` runs the lesson's own check 40 times under both statistics.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "16-speculative-decoding"
TARGET = (0.35, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02)
SAMPLES, REPEATS, CRITICAL, SHIPPED = 50_000, 40, 14.067, 30.0


def marginal(ref, q, p):
    """The exact distribution of spec_step_one_token: accept mass plus reject mass."""
    residual = ref.residual(list(q), p)
    reject = sum(pi * (1 - min(1.0, qi / pi)) for qi, pi in zip(q, p))
    return [pi * min(1.0, qi / pi) + reject * ri
            for qi, pi, ri in zip(q, p, residual)], reject


def sweep(ref, q, p, repeats=REPEATS, samples=SAMPLES):
    """(paired-sample statistics, against-expectation statistics) over fresh runs."""
    expected = [samples * qi for qi in q]
    paired, exact = [], []
    for seed in range(repeats):
        spec, direct = ref.run_distribution_check(list(q), p, samples, random.Random(seed))
        paired.append(ref.chi_square(spec, direct))
        exact.append(ref.chi_square(spec, expected))
    return paired, exact


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    p = ref.perturb(list(TARGET), 0.02, random.Random(7))
    exact, reject = marginal(ref, TARGET, p)
    paired, against = sweep(ref, TARGET, p)
    return {
        "gap": max(abs(a - b) for a, b in zip(exact, TARGET)),
        "reject": reject, "tv": 1 - sum(min(a, b) for a, b in zip(TARGET, p)),
        "paired": (statistics.fmean(paired), sum(v > CRITICAL for v in paired),
                   sum(v > SHIPPED for v in paired)),
        "against": (statistics.fmean(against), sum(v > CRITICAL for v in against)),
        "degrees": len(TARGET) - 1,
    }


def verify(result):
    paired, against = result["paired"], result["against"]
    return [
        practice.Check(
            "ANSWER: the theorem holds to 1.4e-17, with no sampling at all",
            result["gap"] < 1e-15 and abs(result["reject"] - result["tv"]) < 1e-12,
            f"p_i * min(1, q_i/p_i) + P(reject) * res_i comes to q_i for every i, worst "
            f"disagreement {result['gap']:.1e}. P(reject) is {result['reject']:.6f}, which is "
            f"1 - sum(min(p, q)) = {result['tv']:.6f} exactly -- the total variation distance. "
            "50,000 samples are not needed to establish any of this",
        ),
        practice.Check(
            "FINDING: the statistic main() prints runs twice as hot as it should",
            paired[0] > 1.6 * against[0],
            f"chi_square(spec_counts, direct_counts) compares one sample against another, so the "
            f"expected counts carry their own noise and the statistic carries both. Over "
            f"{REPEATS} repeats its mean is {paired[0]:.2f} where a chi-square on "
            f"{result['degrees']} degrees of freedom has mean {result['degrees']}; against the "
            f"true expectation the same draws average {against[0]:.2f}",
        ),
        practice.Check(
            "FINDING: so the exercise's own criterion fails 43% of the time on a true theorem",
            paired[1] > REPEATS // 4 and against[1] <= REPEATS // 10,
            f"the p > 0.05 critical value at {result['degrees']} df is {CRITICAL}. The printed "
            f"statistic exceeds it in {paired[1]} of {REPEATS} repeats, "
            f"{paired[1] / REPEATS:.0%}; the correct statistic exceeds it in "
            f"{against[1]} of {REPEATS}. A reader following the instruction would conclude the "
            "theorem fails on almost half their runs",
        ),
        practice.Check(
            "FINDING: and the threshold the lesson ships can never fire",
            paired[2] == 0,
            f"main() prints PASS below {SHIPPED:.0f}, about p = 1e-4 at {result['degrees']} df, "
            f"and across {REPEATS} repeats that is exceeded {paired[2]} times. The test as "
            "written cannot fail and the test as specified fails half the time -- neither of them "
            "is measuring the theorem",
        ),
        practice.Check(
            "CONTROL: the failures are the statistic, not the sampler",
            against[1] <= 2,
            f"the same 50,000-sample draws, scored against {SAMPLES:,} * q instead of against a "
            f"second sample, exceed the critical value {against[1]} time(s) in {REPEATS} -- the "
            "nominal rate. The speculative sampler is fine; what is broken is the comparison it "
            "is being held to",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
