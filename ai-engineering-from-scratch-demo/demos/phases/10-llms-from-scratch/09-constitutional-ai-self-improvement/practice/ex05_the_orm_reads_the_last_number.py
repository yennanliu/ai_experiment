"""Exercise 5 — the ORM reads the last number, so showing your working can cost you the reward.

    Build a process reward scorer for a two-step arithmetic problem. Given "What
    is (3+4)*5?", the model must show the intermediate 3+4=7 step. Grade the
    intermediate step separately from the final answer and compare PRM-weighted
    GRPO to pure ORM-weighted GRPO over 10 rounds.

Reading of the exercise: the ORM is the lesson's own `combined_reward`, and the
PRM is written beside it -- 0.5 for stating the intermediate 7 and 0.5 for
ending on 35 -- so the two differ only in what they can see. The prompt is the
one the exercise names, and the comparison over 10 rounds uses the lesson's own
`group_relative_advantage`, which is what GRPO would weight.

**FINDING: `reward_math` reads the *last* number in the response.** So a
response that gets the answer right and then shows its working scores **0.0**:
`"The answer is 35. Note 3+4=7."` ends on 7 and is marked wrong. The same
content in the other order -- `"First 3+4=7, then 7*5=35."` -- scores **1.0**.
The ORM is sensitive to the position of the working, not to whether it is there.

**ANSWER: both responses that show their working rank the wrong way under the
ORM.** Taking the bare `"35"` as a baseline, the PRM ranks both working
responses above it -- 1.0 against 0.5 -- and the ORM ranks neither above it: one
ties at 1.0 and the other falls to 0.0. The two cases the exercise wants
rewarded are exactly the two the scorers order differently.

**FINDING: the exercise's own requirement is unrewarded by the ORM.** "The model
must show the intermediate 3+4=7 step" is worth exactly **0.0** extra under
`combined_reward`: `"35"` and `"3+4=7 so 35"` both score 1.0. There is no ORM
setting that pays for the step, which is why a separate process reward has to
exist -- and the exercise asks you to compare against an ORM that is indifferent
to the thing being compared.

**FINDING: ten rounds cannot separate them, because nothing is trained.**
`self_improvement_round` returns `['per_prompt', 'overall_mean']` and no model
(Exercise 4), so a PRM-weighted and an ORM-weighted run over 10 rounds draw from
the same fixed sampler and discard both sets of advantages -- which at
`group_size=4` are non-degenerate under both scorers, and so informative and
thrown away either way.

Structure: `process_reward` is the PRM the exercise asks for; `CASES` is the
four-way table the two scorers are compared on.
"""

from __future__ import annotations

import random
import re

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "09-constitutional-ai-self-improvement"
PROMPT, GROUP, SEEDS = "What is (3+4)*5?", 4, range(10)
INTERMEDIATE, FINAL = 7, 35
CASES = {
    "working then answer": "First 3+4=7, then 7*5=35.",
    "answer then working": "The answer is 35. Note 3+4=7.",
    "answer only": "35",
    "working only": "3+4=7",
}


def process_reward(response):
    """The PRM: half for the intermediate step, half for the final answer."""
    numbers = [int(n) for n in re.findall(r"-?\d+", response)]
    step = 0.5 if INTERMEDIATE in numbers else 0.0
    answer = 0.5 if FINAL in numbers else 0.0
    return step + answer


def degenerate(ref, scorer, group_size):
    """Groups whose advantages are all zero under this scorer."""
    zero = total = 0
    for seed in SEEDS:
        sampler = ref.mock_sampler(random.Random(seed))
        rewards = [scorer(sampler(PROMPT)) for _ in range(group_size)]
        zero += all(abs(a) < 1e-9 for a in ref.group_relative_advantage(rewards))
        total += 1
    return zero, total


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = {name: (ref.combined_reward(PROMPT, text), process_reward(text))
             for name, text in CASES.items()}
    orm_zero, orm_total = degenerate(ref, lambda r: ref.combined_reward(PROMPT, r), GROUP)
    prm_zero, _ = degenerate(ref, process_reward, GROUP)
    return {
        "table": table,
        "orm_values": sorted({orm for orm, _ in table.values()}),
        "prm_values": sorted({prm for _, prm in table.values()}),
        "degenerate": ((orm_zero, orm_total), (prm_zero, orm_total)),
        "step_pays": (ref.combined_reward(PROMPT, "3+4=7 so the answer is 35")
                      - ref.combined_reward(PROMPT, "35")),
        "inverted": [name for name in CASES
                     if (table[name][0] > table["answer only"][0])
                     != (table[name][1] > table["answer only"][1])],
        "keys": list(ref.self_improvement_round(
            [PROMPT], ref.mock_sampler(random.Random(0)), group_size=GROUP)),
    }


def verify(result):
    table = result["table"]
    (orm_zero, total), (prm_zero, _) = result["degenerate"]
    ordered, reversed_order = table["working then answer"], table["answer then working"]
    return [
        practice.Check(
            "FINDING: reward_math reads the last number, so word order decides the reward",
            ordered[0] == 1.0 and reversed_order[0] == 0.0 and ordered[1] == reversed_order[1],
            f"{CASES['working then answer']!r} scores {ordered[0]:.1f} and "
            f"{CASES['answer then working']!r} scores {reversed_order[0]:.1f} -- the same "
            f"content, the same PRM score of {ordered[1]:.1f}, opposite ORM verdicts. "
            "reward_math takes re.findall(...)[-1], so showing your working after the answer "
            "is marked wrong",
        ),
        practice.Check(
            "ANSWER: both responses that show their working rank the wrong way under the ORM",
            result["inverted"] == ["working then answer", "answer then working"],
            "the four cases score "
            + ", ".join(f"{name} ORM {orm:.1f} PRM {prm:.1f}"
                        for name, (orm, prm) in table.items())
            + ". Against the bare answer '35' as a baseline, the PRM ranks both working "
            f"responses above it and the ORM ranks neither above it -- {result['inverted']} are "
            "the two the scorers invert, and they are the two the exercise wants rewarded",
        ),
        practice.Check(
            "FINDING: the exercise's own requirement is worth 0.0 under the ORM",
            result["step_pays"] == 0.0,
            "'the model must show the intermediate 3+4=7 step' pays "
            f"{result['step_pays']:.1f} extra under combined_reward: '35' and "
            f"'3+4=7 so the answer is 35' both score {table['answer only'][0]:.1f}. There is no "
            "ORM setting that pays for the step, which is why a process reward has to exist -- "
            "and the exercise asks you to compare against an ORM indifferent to the comparison",
        ),
        practice.Check(
            "FINDING: ten rounds cannot separate them, because nothing is trained",
            set(result["keys"]) == {"per_prompt", "overall_mean"},
            f"self_improvement_round returns {result['keys']} -- statistics and no model -- so a "
            "PRM-weighted run and an ORM-weighted run over 10 rounds draw from the same fixed "
            f"sampler and discard both sets of advantages. At group_size={GROUP} neither scorer "
            f"even produces a degenerate group on this prompt ({orm_zero} and {prm_zero} of "
            f"{total}), so the advantages are informative and thrown away either way",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
