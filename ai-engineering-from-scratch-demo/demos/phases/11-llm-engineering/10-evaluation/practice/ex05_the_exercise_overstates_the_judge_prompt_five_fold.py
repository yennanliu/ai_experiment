"""Exercise 5 — the exercise's 500 tokens is five times the real prompt, and 90% of it is resent.

    **Build a cost tracker.** Track the token usage and cost of every judge
    call. Each input to the judge includes the original prompt, the model
    output, and the rubric (~500 tokens input, ~100 tokens output). Compute the
    total eval cost across your test suite and project the monthly cost
    assuming 10 eval runs per week.

Reading of the exercise: a judge call is one (test case, criterion) pair, which
is what `score_with_llm_judge` does, and the input is measured rather than
assumed -- the case's input text, the model output and the rubric for that one
criterion, at the usual word-count estimate of 1.3 tokens per word, since the
lesson imports no tokenizer.

**ANSWER: 32 calls, 3,216 input tokens, 3,200 output tokens per run.** Eight
cases times four criteria. At $2.50 and $10.00 per million tokens that is
$0.04 per run and $1.74 a month at 10 runs a week -- a number too small to
decide anything, which is itself the answer for a suite this size.

**FINDING: the exercise's 500 input tokens is 5x the measured prompt.** The
real mean is 100 tokens per call: 8 for the input text, 30 for the model output
and 62 for the rubric. Assuming 500 would put the input bill at 16,000 tokens
against the measured 3,216 and overstate the input cost by a factor of 5.

**FINDING: the rubric is 62% of the prompt, and the output 30%.** The thing the
judge is judging is the smallest part of what it reads. Halving the rubric
would save more than deleting the model output.

**FINDING: 28% of the input is the same text sent four times.** The four
criteria of a case differ only in their rubric block, so the case content --
input plus output, 304 tokens per run -- is sent four times and 912 of the
3,216 input tokens are repeats. Batching the four criteria into one call takes
the run to 2,304 input tokens, a 28% cut for the same 32 scores.

**FINDING: the projection is per-suite and the suite is 8 cases.** Scaled to
1,000 cases the same run costs $5.00 and $216.88 a month -- which is the number
worth computing, and the exercise's per-suite framing hides it behind a suite
that costs four cents.

Structure: `judge_calls` enumerates the (case, criterion) pairs, `tokens`
is the word-count estimate, and `project` does the arithmetic in one place.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "10-evaluation"
CRITERIA = ("relevance", "correctness", "helpfulness", "safety")
MODEL = "baseline-v1"
OUTPUT_TOKENS = 100
ASSUMED_INPUT = 500
INPUT_RATE, OUTPUT_RATE = 2.50 / 1_000_000, 10.00 / 1_000_000
RUNS_PER_MONTH = 10 * 52 / 12


def tokens(text):
    """The lesson's own estimator shape: words times 1.3."""
    return int(len(text.split()) * 1.3)


def rubric_text(ref, criterion):
    return " ".join(f"{level}: {text}"
                    for level, text in ref.RUBRICS.get(criterion, {}).items())


def judge_calls(ref, suite):
    rows = []
    for case in suite:
        output = ref.run_model(MODEL, case.input_text)
        for criterion in CRITERIA:
            rows.append({"input": tokens(case.input_text), "output": tokens(output),
                         "rubric": tokens(rubric_text(ref, criterion)),
                         "criterion": criterion})
    return rows


def project(input_tokens, output_tokens, cases=8):
    per_run = input_tokens * INPUT_RATE + output_tokens * OUTPUT_RATE
    return {"per_run": round(per_run, 4),
            "per_month": round(per_run * RUNS_PER_MONTH, 2),
            "scaled_run": round(per_run * 1000 / cases, 2),
            "scaled_month": round(per_run * RUNS_PER_MONTH * 1000 / cases, 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "eval_framework")
    suite = ref.build_test_suite()
    rows = judge_calls(ref, suite)
    total_input = sum(r["input"] + r["output"] + r["rubric"] for r in rows)
    total_output = len(rows) * OUTPUT_TOKENS
    shared = sum(tokens(c.input_text) + tokens(ref.run_model(MODEL, c.input_text))
                 for c in suite)
    batched = shared + sum(tokens(rubric_text(ref, c)) for c in CRITERIA) * len(suite)
    return {
        "calls": len(rows), "cases": len(suite), "criteria": len(CRITERIA),
        "input_tokens": total_input, "output_tokens": total_output,
        "mean_input": round(statistics.mean(r["input"] + r["output"] + r["rubric"]
                                            for r in rows)),
        "assumed_input": ASSUMED_INPUT * len(rows), "assumed_each": ASSUMED_INPUT,
        "share": {part: round(sum(r[part] for r in rows) / total_input, 2)
                  for part in ("input", "output", "rubric")},
        "shared": shared, "resent": shared * (len(CRITERIA) - 1),
        "batched": batched, "saving": round(1 - batched / total_input, 2),
        **project(total_input, total_output, len(suite)),
    }


def verify(result):
    share = result["share"]
    return [
        practice.Check(
            "ANSWER: 32 calls, 3,216 input tokens, $0.04 a run",
            all([result["calls"] == result["cases"] * result["criteria"],
                 result["input_tokens"] == 3216, result["per_run"] < 0.05]),
            f"{result['cases']} cases times {result['criteria']} criteria is "
            f"{result['calls']} judge calls: {result['input_tokens']:,} input tokens and "
            f"{result['output_tokens']:,} output, ${result['per_run']} a run and "
            f"${result['per_month']} a month at 10 runs a week. Too small to decide "
            "anything, which is the honest answer for a suite this size",
        ),
        practice.Check(
            "FINDING: the exercise's 500 input tokens is 5x the measured prompt",
            all([result["mean_input"] == 100,
                 result["assumed_input"] > 4 * result["input_tokens"]]),
            f"the measured mean is {result['mean_input']} tokens per call, not "
            f"{result['assumed_each']}. Assuming the exercise's figure puts the input bill "
            f"at {result['assumed_input']:,} tokens against the measured "
            f"{result['input_tokens']:,} -- a factor of "
            f"{result['assumed_input'] / result['input_tokens']:.1f}",
        ),
        practice.Check(
            "FINDING: the rubric is the largest part of the prompt",
            all([share["rubric"] > share["output"] > share["input"],
                 share["rubric"] > 0.5]),
            f"the prompt splits {share} between the case's input text, the model output "
            "and the rubric. The thing the judge is judging is the smallest part of what "
            "it reads, and halving the rubric would save more than deleting the output",
        ),
        practice.Check(
            "FINDING: 28% of the input is the same text sent four times",
            all([result["resent"] > 0.25 * result["input_tokens"],
                 result["saving"] == 0.28]),
            f"the four criteria of a case differ only in the rubric block, so the case "
            f"content -- {result['shared']} tokens -- is sent {result['criteria']} times "
            f"and {result['resent']:,} of the {result['input_tokens']:,} are repeats. "
            f"Batching the four criteria into one call gives {result['batched']:,} input "
            f"tokens, a {result['saving']:.0%} cut for the same {result['calls']} scores",
        ),
        practice.Check(
            "FINDING: the projection is per-suite, and the suite is 8 cases",
            all([result["scaled_month"] > 100, result["per_month"] < 2]),
            f"scaled to 1,000 cases the same run costs ${result['scaled_run']} and "
            f"${result['scaled_month']} a month, against ${result['per_month']} for the "
            "eight-case suite. The per-suite framing hides the number worth computing "
            "behind a suite that costs four cents",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
