"""Exercise 4 — three raters give identical ratings, and on two criteria kappa is 0/0.

    **Add inter-rater reliability.** Run the LLM judge 3 times on each test
    case (simulating different judge "raters"). Compute Cohen's kappa or
    Krippendorff's alpha between the three runs. If agreement is below 0.7,
    your rubric is too ambiguous -- rewrite it.

Reading of the exercise: "run the judge 3 times" is taken literally first --
three calls to `score_with_llm_judge` on the same inputs -- and then two ways
of actually getting three raters are tried, because the literal reading has no
variance to measure.

**ANSWER: the three runs are byte-identical, so observed agreement is 1.000 on
every criterion.** `simulate_judge_score` is a pure function of
`(input_text, model_output, criterion)`: a length bucket, a reference-overlap
adjustment and `md5(...) % 100`. There is no sampling anywhere in it.

**FINDING: kappa is 1.000 on all four criteria, and says nothing about the
rubric.** The raters are the same call, so it measures the function. And the
rubric is barely exercised: `relevance` only ever emits 4 and 5, `correctness`
2, 3 and 4, `helpfulness` 3, 4 and 5, `safety` 1, 4 and 5 -- two or three of
five levels each.

**FINDING: the rule the exercise states can never fire.** "If agreement is
below 0.7, rewrite the rubric" requires disagreement, and the only disagreement
this judge can produce comes from changing its inputs. The rubric is never
tested by the test the exercise designs.

**FINDING: perturb the output and the judge moves, but on the wrong axis.**
Three "raters" made by appending 1, 2 and 3 spaces to the model output --
semantically identical documents -- disagree on 17 of the 32 case-criterion
cells, because the md5 seed is taken over the raw string. Observed agreement
falls to 0.469, which the exercise's rule would read as an ambiguous rubric.

**CONTROL: vary what a rater should be sensitive to, and it is not.** Three
raters made by paraphrasing the output agree on 29 of 32 -- more stable under a
rewrite than under a space. And 16 of the 17 whitespace disagreements are one
point apart, which the *nominal* kappa the exercise names charges exactly as
much as a 1-versus-5.

Structure: `ratings` runs one rater over the suite, `RATERS` holds the three
perturbations, and `kappa` is Cohen's on the nominal scale.
"""

from __future__ import annotations

import collections
import itertools
import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "10-evaluation"
CRITERIA = ("relevance", "correctness", "helpfulness", "safety")
MODEL = "baseline-v1"
WHITESPACE = (lambda t: t + " ", lambda t: t + "  ", lambda t: t + "   ")
PARAPHRASE = (lambda t: t, lambda t: t.replace(". ", ", and "),
              lambda t: t.replace("the ", "this "))


def ratings(ref, suite, perturb=lambda t: t):
    """One rater: the judge's score for every case and criterion."""
    rows = {}
    for case in suite:
        output = perturb(ref.run_model(MODEL, case.input_text))
        for criterion in CRITERIA:
            rows[(case.id, criterion)] = ref.simulate_judge_score(
                case.input_text, output, case.reference_output, criterion)
    return rows


def kappa(left, right):
    """Cohen's kappa on the nominal scale, or None when p_e is 1."""
    cells = list(left)
    observed = statistics.mean(left[c] == right[c] for c in cells)
    counts_l = collections.Counter(left[c] for c in cells)
    counts_r = collections.Counter(right[c] for c in cells)
    expected = sum(counts_l[v] * counts_r[v] for v in set(counts_l) | set(counts_r))
    expected /= len(cells) ** 2
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def agreement(raters):
    cells = list(raters[0])
    return round(statistics.mean(len({r[c] for r in raters}) == 1 for c in cells), 4)


def per_criterion(ref, raters):
    out = {}
    for criterion in CRITERIA:
        cells = [(case, c) for case, c in raters[0] if c == criterion]
        values = {r[cell] for r in raters for cell in cells}
        pair = [{cell: r[cell] for cell in cells} for r in raters[:2]]
        out[criterion] = {"values": sorted(values), "kappa": kappa(*pair)}
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "eval_framework")
    suite = ref.build_test_suite()
    literal = [ratings(ref, suite) for _ in range(3)]
    spaces = [ratings(ref, suite, p) for p in WHITESPACE]
    rewrites = [ratings(ref, suite, p) for p in PARAPHRASE]
    return {
        "cells": len(literal[0]), "identical": literal[0] == literal[1] == literal[2],
        "literal_agreement": agreement(literal),
        "per_criterion": per_criterion(ref, literal), "threshold": 0.7,
        **perturbed(literal[0], spaces, rewrites),
    }


def perturbed(cells, spaces, rewrites):
    return {"space_agreement": agreement(spaces),
            "space_disagreements": sum(len({r[c] for r in spaces}) > 1 for c in cells),
            "adjacent": sum(1 for c in cells
                            if max(r[c] for r in spaces) - min(r[c] for r in spaces) == 1),
            "rewrite_agreement": agreement(rewrites),
            "rewrite_agreements": sum(len({r[c] for r in rewrites}) == 1 for c in cells)}


def verify(result):
    per = result["per_criterion"]
    return [
        practice.Check(
            "ANSWER: the three runs are byte-identical",
            all([result["identical"], result["literal_agreement"] == 1.0]),
            f"three calls to the judge on the same inputs agree on all {result['cells']} "
            f"case-criterion cells, observed agreement {result['literal_agreement']}. "
            "`simulate_judge_score` is a pure function of (input, output, criterion): a "
            "length bucket, a reference-overlap adjustment and md5(...) % 100",
        ),
        practice.Check(
            "FINDING: kappa is 1.000 because the raters are the same call",
            all([{c: per[c]["kappa"] for c in CRITERIA} == dict.fromkeys(CRITERIA, 1.0),
                 max(len(per[c]["values"]) for c in CRITERIA) == 3]),
            f"kappa is 1.0 on all four criteria, which is a statement about the function "
            f"and not about the rubric -- the raters are the same call. And the rubric is "
            f"barely exercised: the levels each criterion actually emits are "
            f"{ {c: per[c]['values'] for c in CRITERIA} }, two or three of five",
        ),
        practice.Check(
            "FINDING: the rule the exercise states can never fire",
            all([result["literal_agreement"] > result["threshold"], result["identical"]]),
            f"'if agreement is below {result['threshold']}, rewrite the rubric' requires "
            f"disagreement, and observed agreement is {result['literal_agreement']}. The "
            "only disagreement this judge can produce comes from changing its inputs, so "
            "the rubric is never tested by the test the exercise designs",
        ),
        practice.Check(
            "FINDING: perturb the output and it moves, on the wrong axis",
            all([result["space_agreement"] < result["threshold"],
                 result["space_disagreements"] == 17]),
            f"three raters made by appending 1, 2 and 3 spaces to the output -- "
            f"semantically identical documents -- disagree on "
            f"{result['space_disagreements']} of {result['cells']} cells, agreement "
            f"{result['space_agreement']}. The md5 seed is taken over the raw string, so "
            "the exercise's rule would read whitespace as an ambiguous rubric",
        ),
        practice.Check(
            "CONTROL: it is more stable under a rewrite than under a space",
            all([result["rewrite_agreement"] > result["space_agreement"],
                 result["rewrite_agreements"] == 29,
                 result["adjacent"] == result["space_disagreements"] - 1]),
            f"paraphrasing the output instead -- reordering clauses, substituting words -- "
            f"agrees on {result['rewrite_agreements']} of {result['cells']} cells, "
            f"{result['rewrite_agreement']} against the whitespace raters' "
            f"{result['space_agreement']}: more stable under a rewrite than under a space. "
            f"And {result['adjacent']} of the {result['space_disagreements']} whitespace "
            "disagreements are one point apart, which the nominal kappa the exercise names "
            "charges exactly as much as a 1-versus-5",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
