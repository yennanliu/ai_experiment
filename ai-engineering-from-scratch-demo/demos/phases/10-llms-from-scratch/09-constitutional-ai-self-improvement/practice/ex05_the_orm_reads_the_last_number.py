"""Exercise 5 — the ORM reads the last number, so showing your working can cost you the reward.

    Build a process reward scorer for a two-step arithmetic problem. Given "What
    is (3+4)*5?", the model must show the intermediate 3+4=7 step. Grade the
    intermediate step separately from the final answer and compare PRM-weighted
    GRPO to pure ORM-weighted GRPO over 10 rounds.

Reading of the exercise: the ORM is the lesson's own `combined_reward`, and the
PRM is written beside it in two versions -- `loose_prm`, which is the exercise's
own phrasing (half if 7 appears, half if 35 appears), and `strict_prm`, which
requires the equation `3+4=7` to be stated and 35 to be the last number -- so
that the cost of closing the loose version's holes is visible rather than
assumed. The prompt is the one the exercise names, and the comparison over 10
rounds uses the lesson's own `group_relative_advantage`, which is what GRPO
would weight.

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

**FINDING: a PRM that only looks for the numbers can be gamed.** `"7 is lucky;
35 is mentioned; final answer 36"` contains no `3+4=7` step and ends on the
wrong answer, and the exercise's own phrasing of the PRM gives it **1.0**.
Requiring the equation and the final position takes it to **0.0**, which is what
the ORM gives it too.

**FINDING: closing that hole costs the PRM its independence from word order.**
The strict PRM scores `"The answer is 35. Note 3+4=7."` **0.5** against the loose
PRM's 1.0, because its answer half is the same positional rule the ORM uses.
"Which number is the answer" has no non-positional test in free text, so a PRM
that cannot be gamed by a decoy is a PRM that inherits the defect the exercise
is asking you to measure.

**FINDING: ten rounds cannot separate them, because nothing is trained.**
`self_improvement_round` returns `['per_prompt', 'overall_mean']` and no model
(Exercise 4), so a PRM-weighted and an ORM-weighted run over 10 rounds draw from
the same fixed sampler and discard both sets of advantages -- which at
`group_size=4` are non-degenerate under both scorers, and so informative and
thrown away either way.

Structure: `loose_prm` is the PRM the exercise asks for and `strict_prm` the
same with its holes closed; `CASES` is the table the three scorers are compared
on.
"""

from __future__ import annotations

import random
import re

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "09-constitutional-ai-self-improvement"
PROMPT, GROUP, SEEDS = "What is (3+4)*5?", 4, range(10)
INTERMEDIATE, FINAL = 7, 35
STEP_EQUATION = re.compile(r"3\s*\+\s*4\s*=\s*7")
CASES = {
    "working then answer": "First 3+4=7, then 7*5=35.",
    "answer then working": "The answer is 35. Note 3+4=7.",
    "answer only": "35",
    "working only": "3+4=7",
    "decoy": "7 is lucky; 35 is mentioned; final answer 36",
}


def numbers_in(response):
    return [int(n) for n in re.findall(r"-?\d+", response)]


def loose_prm(response):
    """The PRM as the exercise phrases it: half if 7 appears, half if 35 appears."""
    numbers = numbers_in(response)
    return (0.5 if INTERMEDIATE in numbers else 0.0) + (0.5 if FINAL in numbers else 0.0)


def strict_prm(response):
    """The same PRM with the holes closed: the equation must be stated, 35 must be last."""
    numbers = numbers_in(response)
    return ((0.5 if STEP_EQUATION.search(response) else 0.0)
            + (0.5 if numbers and numbers[-1] == FINAL else 0.0))


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
    table = {name: (ref.combined_reward(PROMPT, text), loose_prm(text), strict_prm(text))
             for name, text in CASES.items()}
    orm_zero, orm_total = degenerate(ref, lambda r: ref.combined_reward(PROMPT, r), GROUP)
    prm_zero, _ = degenerate(ref, strict_prm, GROUP)
    baseline = table["answer only"]
    return {
        "table": table,
        "degenerate": ((orm_zero, orm_total), (prm_zero, orm_total)),
        "step_pays": (ref.combined_reward(PROMPT, "3+4=7 so the answer is 35")
                      - ref.combined_reward(PROMPT, "35")),
        "inverted": [name for name in CASES if name != "decoy"
                     and (table[name][0] > baseline[0]) != (table[name][1] > baseline[1])],
        "keys": list(ref.self_improvement_round(
            [PROMPT], ref.mock_sampler(random.Random(0)), group_size=GROUP)),
    }


def scores(table, index):
    return ", ".join(f"{name} {row[index]:.1f}" for name, row in table.items())


def verify(result):
    table = result["table"]
    (orm_zero, total), (prm_zero, _) = result["degenerate"]
    ordered, flipped = table["working then answer"], table["answer then working"]
    decoy = table["decoy"]
    return [
        practice.Check(
            "FINDING: reward_math reads the last number, so word order decides the reward",
            ordered[0] == 1.0 and flipped[0] == 0.0 and ordered[1] == flipped[1],
            f"{CASES['working then answer']!r} scores {ordered[0]:.1f} and "
            f"{CASES['answer then working']!r} scores {flipped[0]:.1f} -- the same content, the "
            f"same PRM score of {ordered[1]:.1f}, opposite ORM verdicts. reward_math takes "
            "re.findall(...)[-1], so showing your working after the answer is marked wrong",
        ),
        practice.Check(
            "ANSWER: both responses that show their working rank the wrong way under the ORM",
            result["inverted"] == ["working then answer", "answer then working"],
            "against the bare answer '35' as a baseline the ORM scores " + scores(table, 0)
            + " and the loose PRM " + scores(table, 1)
            + f". The PRM ranks both working responses above the baseline and the ORM ranks "
            f"neither above it -- {result['inverted']} are the two the scorers invert, and they "
            "are the two the exercise wants rewarded",
        ),
        practice.Check(
            "FINDING: a PRM that only looks for the numbers pays full marks for neither step",
            decoy[1] == 1.0 and decoy[2] == 0.0 and decoy[0] == 0.0,
            f"{CASES['decoy']!r} contains no 3+4=7 step and ends on the wrong answer, and the "
            f"PRM as the exercise phrases it -- half if 7 appears, half if 35 appears -- gives it "
            f"{decoy[1]:.1f}. Requiring the equation to be stated and 35 to be the last number "
            f"takes it to {decoy[2]:.1f}, which is what the ORM also gives it",
        ),
        practice.Check(
            "FINDING: closing that hole costs the PRM its independence from word order",
            flipped[2] < flipped[1] and ordered[2] == ordered[1],
            f"the strict PRM scores {CASES['answer then working']!r} {flipped[2]:.1f} against the "
            f"loose PRM's {flipped[1]:.1f}, because its answer half is the same positional rule "
            f"the ORM uses. 'Which number is the answer' has no non-positional test in free text, "
            "so a PRM that cannot be gamed by a decoy is a PRM that inherits the defect the "
            "exercise is asking you to measure",
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
