"""Exercise 3 — the fixture does show the effect, and the bootstrap behind it has 8 samples.

    **Implement stratified analysis.** Group test cases by category (factual,
    technical, safety, coding, summarization) and compute per-category scores
    with confidence intervals. Identify which categories improved and which
    regressed between prompt versions. A system can improve overall while
    regressing on a specific category.

Reading of the exercise: the grouping is by `TestCase.category`, the scores are
every criterion of every case in the group, and the interval is the lesson's
own `bootstrap_confidence_interval`. The two prompt versions are baseline-v1
and baseline-v2, as exercise 2 uses.

**ANSWER: the effect the exercise describes is present in its own fixture.**
Overall improves 3.844 -> 3.875 while `factual` regresses 4.750 -> 4.375 and
`summarization` 4.000 -> 3.500. Two categories improve, two regress, one is
unchanged, and the aggregate hides all of it.

**FINDING: the intervals behind those numbers come from 8 distinct resamples.**
`bootstrap_confidence_interval` asks for 1,000 and gets 8, because its index is
`(seed + j * 31) % n` under an LCG with modulus 2**31. When `n` divides a power
of two -- 8 scores per category, 32 per criterion -- `x % n` depends only on
`seed % n`, so the LCG's low bits are its whole period. The 1,000 resamples
collapse to 8 distinct ones and 4 distinct means.

**MECHANISM: it is the classic low-bit failure, and n decides whether it
fires.** At n = 5 the same code produces 993 distinct resamples of 1,000; at
n = 8, 16 and 32 it produces exactly n. Every group size in this lesson is a
power of two.

**FINDING: the interval it reports is wider than a real one.** On the same 16
scores the lesson gives (3.500, 4.250, 4.750) and a genuine bootstrap gives
(3.875, 4.250, 4.625) -- the reported interval is 1.6x wider, so the analysis
errs toward "no significant change" on every comparison.

**FINDING: a category with one score reports an interval of zero.**
`bootstrap_confidence_interval` returns `(0.0, 0.0, 0.0)` for fewer than two
scores, which prints as a score of 0 rather than as a refusal. The suite
escapes it only because each case contributes four criteria.

Structure: `strata` groups and scores, `resamples` counts what the lesson's
bootstrap actually draws, and `honest` is a real bootstrap for comparison.
"""

from __future__ import annotations

import collections
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "10-evaluation"
VERSIONS = ("baseline-v1", "baseline-v2")
SIZES = (5, 8, 16, 32)


def strata(ref, suite, model):
    groups = collections.defaultdict(list)
    for case, result in zip(suite, ref.run_eval_suite(suite, model, model)):
        groups[case.category] += [score.score for score in result.scores]
    return groups


def resamples(scores, n_bootstrap=1000):
    """The lesson's inner sampling loop, kept rather than averaged away."""
    n = len(scores)
    base, out = int(sum(scores) * 1000) % 2 ** 31, []
    for i in range(n_bootstrap):
        seed, sample = (base + i * 7919) % 2 ** 31, []
        for j in range(n):
            sample.append(scores[(seed + j * 31) % n])
            seed = (seed * 1103515245 + 12345) % 2 ** 31
        out.append(tuple(sample))
    return out


def honest(scores, n_bootstrap=1000, seed=0):
    generator = random.Random(seed)
    means = sorted(statistics.mean(generator.choices(scores, k=len(scores)))
                   for _ in range(n_bootstrap))
    return (round(means[25], 4), round(statistics.mean(scores), 4), round(means[974], 4))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "eval_framework")
    suite = ref.build_test_suite()
    first, second = (strata(ref, suite, model) for model in VERSIONS)
    scores = first["factual"] + first["technical"]
    return {
        **movement(first, second), "sizes": {k: len(v) for k, v in sorted(first.items())},
        "distinct": {n: len(set(resamples([(i % 5) + 1 for i in range(n)]))) for n in SIZES},
        "distinct_means": len({round(statistics.mean(s), 9) for s in resamples(scores)}),
        "lesson_ci": ref.bootstrap_confidence_interval(scores),
        "honest_ci": honest(scores), "scores": len(scores),
        "single": ref.bootstrap_confidence_interval([4]),
    }


def movement(first, second):
    means = [{k: round(statistics.mean(v), 3) for k, v in group.items()}
             for group in (first, second)]
    return {"means": means, "categories": sorted(first),
            "overall": [round(statistics.mean(s for v in g.values() for s in v), 3)
                        for g in (first, second)],
            "regressed": sorted(k for k in means[0] if means[1][k] < means[0][k]),
            "improved": sorted(k for k in means[0] if means[1][k] > means[0][k])}


def verify(result):
    means, overall = result["means"], result["overall"]
    lesson, real = result["lesson_ci"], result["honest_ci"]
    widths = (lesson[2] - lesson[0], real[2] - real[0])
    return [
        practice.Check(
            "ANSWER: the effect the exercise describes is present in its own fixture",
            all([overall[1] > overall[0], result["regressed"] == ["factual", "summarization"],
                 result["improved"] == ["safety", "technical"]]),
            f"overall improves {overall[0]} -> {overall[1]} while {result['regressed']} "
            f"regress and {result['improved']} improve: {means[0]} becomes {means[1]}. Two "
            "up, two down, one unchanged, and the aggregate hides all of it",
        ),
        practice.Check(
            "FINDING: the bootstrap behind those intervals draws 8 distinct resamples",
            all([result["distinct"][8] == 8, result["distinct"][16] == 16,
                 result["distinct_means"] == 4]),
            f"`bootstrap_confidence_interval` asks for 1,000 resamples and gets "
            f"{result['distinct']} for n = {list(SIZES)}. Its index is (seed + j * 31) % n "
            f"under an LCG with modulus 2**31, so when n divides a power of two the index "
            f"depends only on seed % n. The {result['scores']} scores here collapse to "
            f"{result['distinct_means']} distinct means",
        ),
        practice.Check(
            "MECHANISM: n decides whether the low-bit failure fires",
            all([result["distinct"][5] > 900, result["distinct"][32] == 32,
                 all(size in (5, 8, 16, 32) for size in SIZES)]),
            f"at n = 5 the same code produces {result['distinct'][5]} distinct resamples of "
            f"1,000; at n = 8, 16 and 32 it produces exactly n. Every group size in this "
            f"lesson is a power of two -- {result['sizes']} scores per category, 32 per "
            "criterion -- so the failure fires everywhere it is used",
        ),
        practice.Check(
            "FINDING: the reported interval is wider than a real one",
            all([widths[0] > 1.4 * widths[1], lesson[1] == real[1]]),
            f"on the same {result['scores']} scores the lesson gives {lesson} and a real "
            f"bootstrap gives {real} -- same mean, {widths[0] / widths[1]:.1f}x the width. "
            "The analysis therefore errs toward 'no significant change' on every "
            "comparison it is used for",
        ),
        practice.Check(
            "FINDING: a group with one score reports an interval of zero",
            result["single"] == (0.0, 0.0, 0.0),
            f"`bootstrap_confidence_interval([4])` returns {result['single']}, which prints "
            "as a score of 0 rather than as a refusal. The suite escapes it only because "
            "each case contributes four criteria, so the smallest group is 4 and not 1",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
