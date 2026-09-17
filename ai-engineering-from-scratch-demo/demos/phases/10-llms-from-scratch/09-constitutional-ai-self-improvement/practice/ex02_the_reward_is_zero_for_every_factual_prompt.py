"""Exercise 2 — a factuality principle has no reward channel: capitals and dates both score 0.0.

    Add a third constitutional principle about factuality. Run the pipeline on
    prompts that require factual claims (capitals, dates) and measure how many
    revisions remove factual errors versus introduce new ones.

Reading of the exercise: the principle is added to `CONSTITUTION` as written and
the pipeline is run on the prompts the exercise names -- capitals and dates --
using the lesson's own `cai_stage_one` and `combined_reward`. "Measure how many
revisions remove factual errors versus introduce new ones" requires the pipeline
to (a) revise something and (b) score factuality, and both are checked.

**ANSWER: 0 and 0, and neither is a measurement.** No revision occurs on any
factual prompt, so there is nothing to classify -- and even if one did,
`combined_reward` returns **0.0** for the right answer and **0.0** for the wrong
one, so "removes an error" and "introduces one" are the same number.

**MECHANISM: `reward_math` evaluates the prompt as Python.** It strips
`"What is "` and `"?"` and calls `eval`, so `"the capital of France"` raises and
the function returns 0.0 from its `except`. Every non-arithmetic prompt scores
zero regardless of the response, which is the whole class of prompts this
exercise is about.

**FINDING: a fourth principle is one in five, not one more.** `cai_stage_one`
calls `random.choice(CONSTITUTION)`, so adding a principle *reduces* the chance
that any existing one is applied, from 1 in 4 to 1 in 5. Adding a principle in
this pipeline dilutes the others rather than accumulating with them.

**FINDING: the new principle could not fire anyway.** `critique` dispatches on
four hard-coded conditions and has no branch keyed to a factuality principle, so
appending the string to `CONSTITUTION` adds a 20% chance of running a critic
that checks length, refusals, commas and hedging -- none of which is a factual
claim. The principle is data the code does not read.

Structure: `factual_reward` is the lesson's own scorer on the exercise's prompt
class; `dilution` is the probability an existing principle is drawn before and
after the addition.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "09-constitutional-ai-self-improvement"
SEED = 0
FACTUAL = (
    ("What is the capital of France?", "Paris.", "Lyon."),
    ("What year did World War II end?", "1945.", "1918."),
    ("What is the capital of Japan?", "Tokyo.", "Kyoto."),
    ("What year did the Apollo 11 landing happen?", "1969.", "1972."),
)
PRINCIPLE = "The response must not state a fact that is false."


def factual_reward(ref, prompt, response):
    return ref.reward_math(prompt, response), ref.combined_reward(prompt, response)


def run_pipeline(ref, principles):
    """`cai_stage_one` over the factual prompts, with `CONSTITUTION` as given."""
    original = ref.CONSTITUTION[:]
    ref.CONSTITUTION[:] = principles
    try:
        random.seed(SEED)
        return ref.cai_stage_one([(prompt, right) for prompt, right, _ in FACTUAL])
    finally:
        ref.CONSTITUTION[:] = original


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before = len(ref.CONSTITUTION)
    rows = run_pipeline(ref, ref.CONSTITUTION + [PRINCIPLE])
    scores = [(prompt, factual_reward(ref, prompt, right), factual_reward(ref, prompt, wrong))
              for prompt, right, wrong in FACTUAL]
    arithmetic = factual_reward(ref, "What is 3+4?", "The answer is 7.")
    return {
        "revisions": sum(row["changed"] for row in rows),
        "problems": sum(1 for row in rows if row["problems"]),
        "prompts": len(FACTUAL),
        "scores": scores,
        "right_equals_wrong": all(right == wrong for _, right, wrong in scores),
        "arithmetic": arithmetic,
        "dilution": (1 / before, 1 / (before + 1)),
        "principles": (before, before + 1),
    }


def verify(result):
    scores, prompts = result["scores"], result["prompts"]
    old_odds, new_odds = result["dilution"]
    before, after = result["principles"]
    return [
        practice.Check(
            "ANSWER: 0 revisions remove an error and 0 introduce one",
            result["revisions"] == 0 and result["problems"] == 0,
            f"cai_stage_one finds problems in {result['problems']} of the {prompts} factual "
            f"prompts and performs {result['revisions']} revisions, so there is nothing to "
            "classify. Exercise 1 shows why: none of the critic's four conditions can fire on a "
            "short factual answer any more than on an arithmetic one",
        ),
        practice.Check(
            "MECHANISM: reward_math evaluates the prompt as Python, so every factual prompt is 0.0",
            result["right_equals_wrong"] and result["arithmetic"][0] == 1.0,
            "reward_math strips 'What is ' and '?' and calls eval, so 'the capital of France' "
            "raises and the except returns 0.0. Right and wrong answers score identically: "
            + "; ".join(f"{p[:26]!r} {right[1]:.1f} vs {wrong[1]:.1f}"
                        for p, right, wrong in scores[:2])
            + f". The same scorer gives {result['arithmetic'][1]:.1f} on an arithmetic prompt, "
            "so the machinery works and this prompt class is outside it",
        ),
        practice.Check(
            "FINDING: 'removes an error' and 'introduces one' are the same number here",
            all(right[1] == wrong[1] == 0.0 for _, right, wrong in scores),
            f"all {prompts} prompts score {scores[0][1][1]:.1f} for the correct answer and "
            f"{scores[0][2][1]:.1f} for the wrong one. The exercise asks for a count of "
            "revisions that remove factual errors versus introduce new ones, and the reward it "
            "would be counted with cannot distinguish the two outcomes",
        ),
        practice.Check(
            "FINDING: a fifth principle is a dilution, not an addition",
            new_odds < old_odds and after == before + 1,
            f"cai_stage_one calls random.choice(CONSTITUTION), so going from {before} principles "
            f"to {after} takes the chance that any existing one is applied from "
            f"{old_odds:.0%} to {new_odds:.0%}. And critique() dispatches on four hard-coded "
            "conditions with no branch keyed to factuality, so the new string adds a 20% chance "
            "of running a critic that checks length, refusals, commas and hedging. The "
            "principle is data the code does not read",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
