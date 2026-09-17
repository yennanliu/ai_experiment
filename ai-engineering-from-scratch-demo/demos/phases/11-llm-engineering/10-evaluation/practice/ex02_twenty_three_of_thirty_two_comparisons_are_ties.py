"""Exercise 2 — 23 of 32 comparisons are ties, and the interval spans a coin flip.

    **Build pairwise comparison.** Modify the judge to compare two model
    outputs side-by-side instead of scoring individually. Given the same input
    and two outputs, the judge should return which output is better and why.
    Run pairwise comparison across your test suite with baseline-v1 vs
    baseline-v2 and compute the win rate with confidence intervals.

Reading of the exercise: the pairwise judge is built on the lesson's own
`simulate_judge_score` -- score both outputs under the same criterion and
return the higher, with the two rubric lines as the "why" -- over the 8 test
cases and 4 criteria, which is 32 comparisons. The interval is the lesson's own
`wilson_confidence_interval`.

**ANSWER: 4 wins, 23 ties, 5 losses.** The win rate excluding ties is 0.444
with a Wilson interval of (0.189, 0.733) -- 0.54 wide, spanning 0.5, on 9
decisive comparisons. The comparison cannot conclude anything, and saying so is
the answer.

**FINDING: 72% of the comparisons are ties, and the exercise has no tie
option.** "The judge should return which output is better" has no third value,
but a pairwise judge built on an integer 1-5 score ties whenever both outputs
land in the same bucket -- which, with five buckets and two similar templates,
is most of the time.

**MECHANISM: the judge scores length, and v2 is longer every time.** The base
score is a bucket on `len(output) > len(input) * 0.5`, and baseline-v2's
template is longer than baseline-v1's on 32 of 32 comparisons. The only other
term is `md5(input + output + criterion) % 100`, which moves the score by 1 in
30% of cases.

**FINDING: the judge is not invariant to whitespace.** Padding baseline-v1's
output with spaces to baseline-v2's length -- no semantic change at all --
moves the record from 4-23-5 to 6-21-5 and flips the sign of the comparison.
The md5 seed is taken over the raw output string, so trailing space is a
different document.

**FINDING: the interval depends on a denominator the exercise does not name.**
Wilson over wins/32 gives (0.050, 0.281) -- at most 28% -- and Wilson over
wins/(wins+losses) gives (0.189, 0.733), an upper bound 2.6x higher. The same
nine decisive comparisons support opposite decisions depending on whether the
23 ties are counted as losses or excluded.

Structure: `compare` is the pairwise judge, `record` tallies the three
outcomes, and `padded` is the whitespace control.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "10-evaluation"
CRITERIA = ("relevance", "correctness", "helpfulness", "safety")
LEFT, RIGHT = "baseline-v1", "baseline-v2"


def compare(ref, case, left, right, criterion):
    """The pairwise judge: score both under one criterion, return the verdict."""
    a = ref.simulate_judge_score(case.input_text, left, case.reference_output, criterion)
    b = ref.simulate_judge_score(case.input_text, right, case.reference_output, criterion)
    verdict = "left" if a > b else "right" if b > a else "tie"
    why = ref.generate_judge_reasoning(case.input_text, left if a >= b else right,
                                       criterion, max(a, b))
    return verdict, a, b, why


def record(ref, suite, pad=False):
    rows = []
    for case in suite:
        left, right = ref.run_model(LEFT, case.input_text), ref.run_model(RIGHT, case.input_text)
        if pad:
            left = left + " " * max(0, len(right) - len(left))
        for criterion in CRITERIA:
            verdict, a, b, why = compare(ref, case, left, right, criterion)
            rows.append({"verdict": verdict, "left_len": len(left), "right_len": len(right),
                         "why": why})
    return rows


def tally(rows):
    counts = {"left": 0, "right": 0, "tie": 0}
    for row in rows:
        counts[row["verdict"]] += 1
    return counts


def solve():
    ref = parity.load_reference(PHASE, LESSON, "eval_framework")
    suite = ref.build_test_suite()
    rows, control = record(ref, suite), record(ref, suite, pad=True)
    counts, padded_counts = tally(rows), tally(control)
    wins, losses = counts["left"], counts["right"]
    decisive = wins + losses
    return {
        "counts": counts, "total": len(rows), "decisive": decisive,
        "tie_rate": round(counts["tie"] / len(rows), 3),
        "win_rate": round(wins / decisive, 3) if decisive else 0.0,
        "ci_decisive": ref.wilson_confidence_interval(wins, decisive),
        "ci_total": ref.wilson_confidence_interval(wins, len(rows)),
        "right_longer": sum(1 for r in rows if r["right_len"] > r["left_len"]),
        "padded": padded_counts,
        "changed": sum(a["verdict"] != b["verdict"] for a, b in zip(rows, control)),
        "why": rows[0]["why"][:60],
    }


def verify(result):
    counts, decisive_ci, total_ci = result["counts"], result["ci_decisive"], result["ci_total"]
    width = round(decisive_ci[1] - decisive_ci[0], 3)
    return [
        practice.Check(
            "ANSWER: 4 wins, 23 ties, 5 losses, and the interval spans a coin flip",
            all([counts == {"left": 4, "tie": 23, "right": 5},
                 decisive_ci[0] < 0.5 < decisive_ci[1], width > 0.5]),
            f"over {result['total']} comparisons the record is {counts}. The win rate "
            f"excluding ties is {result['win_rate']} with a Wilson interval of "
            f"{decisive_ci} -- {width} wide, spanning 0.5, on {result['decisive']} decisive "
            "comparisons. Nothing can be concluded, and saying so is the answer",
        ),
        practice.Check(
            "FINDING: 72% of the comparisons are ties, and the exercise has no tie option",
            all([result["tie_rate"] > 0.7, counts["tie"] == 23]),
            f"{counts['tie']} of {result['total']} comparisons tie, a rate of "
            f"{result['tie_rate']}. 'Which output is better' has no third value, but a "
            "pairwise judge built on an integer 1-5 score ties whenever both outputs land "
            "in the same bucket -- with five buckets and two similar templates, usually",
        ),
        practice.Check(
            "MECHANISM: the judge scores length, and v2 is longer every time",
            result["right_longer"] == result["total"],
            f"baseline-v2's template is longer than baseline-v1's on "
            f"{result['right_longer']} of {result['total']} comparisons, and the base score "
            f"is a bucket on len(output) > len(input) * 0.5. The only other term is "
            f"md5(input + output + criterion) % 100. The reasoning string confirms it: "
            f"{result['why']!r}",
        ),
        practice.Check(
            "FINDING: the judge is not invariant to whitespace",
            all([result["changed"] > 0, result["padded"] != counts,
                 result["padded"]["left"] > counts["left"]]),
            f"padding baseline-v1's output with spaces to baseline-v2's length -- no "
            f"semantic change -- moves the record from {counts} to {result['padded']}, "
            f"changing {result['changed']} verdicts and flipping the sign. The md5 seed is "
            "taken over the raw output string, so trailing space is a different document",
        ),
        practice.Check(
            "FINDING: the interval is computed on whichever denominator you pick",
            all([total_ci[1] < 0.3, decisive_ci[1] > 2.5 * total_ci[1]]),
            f"Wilson over wins/{result['total']} gives {total_ci} -- at most "
            f"{total_ci[1]} -- while Wilson over wins/{result['decisive']} gives "
            f"{decisive_ci}, an upper bound {decisive_ci[1] / total_ci[1]:.1f}x higher. "
            "The exercise does not say which one 'the win rate' means, and the two "
            "readings support opposite decisions",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
