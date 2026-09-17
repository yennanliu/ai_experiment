"""Exercise 1 — the critic revises 0 of 100, because no rule can reach the sampler's output.

    Replace the handwritten critic in Step 2 with an LLM call. Use any local chat
    model. Measure how often the critique and revision actually improve the
    response versus leaving it unchanged.

Reading of the exercise: no local chat model is available, so the substitution
cannot be made -- but the measurement can, and it is the half of the exercise
that has an answer. The handwritten critic is run over 100 responses from the
lesson's own `mock_sampler` on its own arithmetic prompts, and "improve versus
leave unchanged" is scored with the lesson's own `combined_reward`.

**ANSWER: 0 improved, 0 worsened, 100 unchanged.** `cai_stage_one` finds zero
problems and performs zero revisions on 100 responses.

**MECHANISM: the critic's four thresholds do not intersect the sampler's output
distribution.** Over 58 distinct outputs, the longest is **26 words** against a
rule needing more than 40; the most commas in one is **3** against a rule needing
more than 4; none begins with a refusal, and none contains "maybe" or "i think".
Every branch of `critique` is unreachable from every branch of `mock_sampler`.

**FINDING: the length rule is also gated on a coin flip.** It fires only when
`"plainly" in principle`, and `cai_stage_one` picks a principle with
`random.choice(CONSTITUTION)` -- so even a 41-word response would be critiqued
**25%** of the time. The same response gets a different verdict on different
runs, and nothing in the pipeline records which principle it was judged under.

**FINDING: two smaller defects sit behind the unreachable ones.** The guard
`"hedging" not in problems` is a list-membership test against entries like
`"too much hedging"`, so it is always true and never guards anything. And the
refusal revision is `"Here is the answer: " + response.split(":")[-1]`, which on
a refusal with no colon returns the refusal with a prefix:
`"I can't help with that"` becomes `"Here is the answer: I can't help with that"`.

Structure: `outputs` collects the sampler's reachable output set and `reach`
measures the four quantities the critic thresholds on; `scored` runs
`cai_stage_one` and `tally` compares rewards before and after.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "09-constitutional-ai-self-improvement"
SEED, SAMPLES = 0, 25
PROMPTS = ("What is 3+4?", "What is 12*7?", "What is 100-37?", "What is (3+4)*5?")
REFUSALS = ("i can't", "i cannot", "as an ai")


def outputs(ref, draws=400):
    """Every distinct string `mock_sampler` can produce on the lesson's prompts."""
    sampler = ref.mock_sampler(random.Random(SEED))
    return {sampler(prompt) for _ in range(draws) for prompt in PROMPTS}


def scored(ref):
    """`cai_stage_one` over `SAMPLES` responses per prompt, with rewards either side."""
    sampler = ref.mock_sampler(random.Random(SEED))
    random.seed(SEED)
    pairs = [(prompt, sampler(prompt)) for prompt in PROMPTS for _ in range(SAMPLES)]
    rows = ref.cai_stage_one(pairs)
    return [(row, ref.combined_reward(row["prompt"], row["initial"]),
             ref.combined_reward(row["prompt"], row["revised"])) for row in rows]


def reach(reachable):
    """The four quantities the critic's four conditions are thresholds on."""
    return {
        "distinct": len(reachable),
        "max_words": max(len(text.split()) for text in reachable),
        "max_commas": max(text.count(",") for text in reachable),
        "refuses": any(text.lower().startswith(REFUSALS) for text in reachable),
        "hedges": any("maybe" in text.lower() or "i think" in text.lower()
                      for text in reachable),
    }


def tally(rows):
    """How many rows were revised, and how the reward moved when they were."""
    return {
        "total": len(rows),
        "changed": sum(row["changed"] for row, _, _ in rows),
        "problems": sum(1 for row, _, _ in rows if row["problems"]),
        "better": sum(1 for _, before, after in rows if after > before),
        "worse": sum(1 for _, before, after in rows if after < before),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    refusal = "I can't help with that"
    return dict(
        tally(scored(ref)),
        **reach(outputs(ref)),
        gated=(len([p for p in ref.CONSTITUTION if "plainly" in p]), len(ref.CONSTITUTION)),
        prefixed=ref.revise(refusal, {"problems": ["unwarranted refusal"]}),
        refusal=refusal,
    )


def verify(result):
    total, changed = result["total"], result["changed"]
    gated, principles = result["gated"]
    return [
        practice.Check(
            f"ANSWER: 0 improved, 0 worsened, {total} unchanged",
            changed == 0 and result["better"] == result["worse"] == 0,
            f"cai_stage_one finds problems in {result['problems']} of {total} responses and "
            f"performs {changed} revisions, so combined_reward is identical before and after for "
            "every one. The exercise asks how often the critique improves the response versus "
            "leaving it unchanged, and on the lesson's own data the answer is never and always",
        ),
        practice.Check(
            "MECHANISM: the four thresholds do not intersect the sampler's output distribution",
            result["max_words"] <= 40 and result["max_commas"] <= 4
            and not result["refuses"] and not result["hedges"],
            f"over {result['distinct']} distinct sampler outputs the longest is "
            f"{result['max_words']} words against a rule needing more than 40, the most commas "
            f"in one is {result['max_commas']} against a rule needing more than 4, none begins "
            "with a refusal and none contains 'maybe' or 'i think'. Every branch of critique() "
            "is unreachable from every branch of mock_sampler()",
        ),
        practice.Check(
            "FINDING: the length rule is gated on a coin flip as well as on a threshold",
            gated == 1 and principles > 1,
            f"it fires only when 'plainly' in principle, which is true of {gated} of the "
            f"{principles} entries in CONSTITUTION, and cai_stage_one picks one with "
            f"random.choice. So even a 41-word response would be critiqued "
            f"{100 * gated / principles:.0f}% of the time -- the same response gets a different "
            "verdict on different runs, and nothing downstream records which principle it was "
            "judged under",
        ),
        practice.Check(
            "FINDING: the refusal revision prefixes the refusal instead of replacing it",
            result["prefixed"].endswith(result["refusal"]),
            f"revise() returns 'Here is the answer: ' + response.split(':')[-1], and a refusal "
            f"with no colon splits into itself: {result['refusal']!r} becomes "
            f"{result['prefixed']!r}. The guard beside it, `\"hedging\" not in problems`, is a "
            "list-membership test against entries like 'too much hedging', so it is always true "
            "and never guards anything",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
