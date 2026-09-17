"""Exercise 4 — entropy regularisation has nothing to regularise: the sampler is stateless.

    Add entropy regularization to the GRPO objective. The term `-alpha *
    entropy(policy)` with alpha=0.01 encourages diverse sampling. Measure whether
    it delays mode collapse across 5 rounds of self-improvement.

Reading of the exercise: five rounds are run with the lesson's own
`self_improvement_round` on its own `mock_sampler`, over 40 seeds rather than one
so a trend can be told from noise. Mode collapse is measured as the entropy of
the sampler's five output modes -- the quantity `-alpha * entropy(policy)` is
meant to defend.

**FINDING: `self_improvement_round` takes a sampler and returns statistics.**
No model goes in and no model comes out. It draws `group_size` responses, scores
them, computes advantages -- and discards them. Five calls are five independent
draws from one fixed distribution.

**ANSWER: mode entropy sits at its ceiling in round 1 and in round 5.** Against
`ln(5) = 1.6094` for the sampler's five branches, the measured entropy is within
a few hundredths at both ends and unchanged between them: `mock_sampler` picks
uniformly from `["correct", "off_by_one", "wrong", "formatted", "verbose"]` with
`rng.choice` and nothing updates the odds. Mode collapse cannot occur, so
entropy regularisation cannot delay it.

**FINDING: the reward mean does not trend either.** Over 40 seeds the five
rounds average 0.6114, 0.6401, 0.6504, 0.6105, 0.6070, with a per-round spread
of **0.0842** and a fitted slope of **-3.8e-03** per round -- smaller than a
twentieth of the noise. A single-seed run that happens to decline, which the
lesson's own demo can produce, is that spread and not a measurement.

**FINDING: the advantages are discarded even when they are informative.** At
`group_size=8` none of the groups is degenerate -- every one has a non-zero
advantage vector that a policy gradient could use. The loop computes them
correctly and then returns a summary dictionary. The missing piece is not the
entropy term.

Structure: `rounds` runs the lesson's loop and records what it reports;
`mode_entropy` measures the quantity the exercise's regulariser targets.
"""

from __future__ import annotations

import collections
import math
import random
import re
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "09-constitutional-ai-self-improvement"
ROUNDS, GROUP, SEEDS = 5, 8, range(40)
PROMPTS = ("What is 3+4?", "What is 12*7?", "What is 100-37?", "What is (3+4)*5?")
MODES = ("correct", "off_by_one", "wrong", "formatted", "verbose")


def classify(ref, prompt, response):
    """Which of `mock_sampler`'s five modes produced this response."""
    if response.startswith("<answer>"):
        return "formatted"
    if response.startswith("Let me think"):
        return "verbose"
    truth = eval(prompt.replace("What is ", "").replace("?", "").strip(),  # noqa: S307
                 {"__builtins__": {}}, {})
    stated = int(re.findall(r"-?\d+", response)[-1])
    if stated == truth:
        return "correct"
    return "off_by_one" if abs(stated - truth) == 1 else "wrong"


def mode_entropy(ref, seed, draws=200):
    """Entropy of the sampler's output modes, in nats, over one round's worth of draws."""
    sampler = ref.mock_sampler(random.Random(seed))
    counts = collections.Counter(classify(ref, prompt, sampler(prompt))
                                 for _ in range(draws) for prompt in PROMPTS)
    total = sum(counts.values())
    return -sum((n / total) * math.log(n / total) for n in counts.values())


def rounds(ref, seed):
    """`ROUNDS` calls of the lesson's own self-improvement loop on one sampler."""
    sampler = ref.mock_sampler(random.Random(seed))
    return [ref.self_improvement_round(list(PROMPTS), sampler, group_size=GROUP)
            for _ in range(ROUNDS)]


def trend(traces):
    """Per-round mean, the spread across seeds, and the least-squares slope."""
    per_round = [statistics.fmean(trace[i] for trace in traces) for i in range(ROUNDS)]
    spread = statistics.fmean(statistics.pstdev(trace[i] for trace in traces)
                              for i in range(ROUNDS))
    xs = list(range(ROUNDS))
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(per_round)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, per_round))
             / sum((x - mean_x) ** 2 for x in xs))
    return per_round, spread, slope


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    per_round, spread, slope = trend(
        [[r["overall_mean"] for r in rounds(ref, seed)] for seed in SEEDS])
    sample = rounds(ref, 1)
    groups = [p["advantages"] for r in sample for p in r["per_prompt"]]
    return {
        "per_round": per_round,
        "spread": spread,
        "slope": slope,
        "entropy": (mode_entropy(ref, 1), mode_entropy(ref, 5)),
        "ceiling": math.log(len(MODES)),
        "modes": len(MODES),
        "groups": len(groups),
        "degenerate": sum(all(abs(a) < 1e-9 for a in g) for g in groups),
        "returns_model": any(hasattr(v, "forward") for v in sample[0].values()),
    }


def verify(result):
    first, last = result["entropy"]
    per_round, spread, slope = result["per_round"], result["spread"], result["slope"]
    return [
        practice.Check(
            "FINDING: self_improvement_round takes a sampler and returns statistics",
            not result["returns_model"],
            "no model goes in and no model comes out. The loop draws group_size responses, "
            "scores them with combined_reward, computes group_relative_advantage -- and then "
            f"returns a dictionary of means. {ROUNDS} calls are {ROUNDS} independent draws from "
            "one fixed distribution, which is what 'rounds of self-improvement' names here",
        ),
        practice.Check(
            "ANSWER: mode entropy is at its ceiling in round 1 and in round 5",
            abs(first - last) < 0.05 and result["ceiling"] - first < 0.05,
            f"the sampler's output modes carry {first:.4f} nats at the first round and "
            f"{last:.4f} at the last, against a ceiling of ln({result['modes']}) = "
            f"{result['ceiling']:.4f} for its {result['modes']} branches -- within "
            f"{result['ceiling'] - first:.4f} of the maximum, and unchanged between the rounds. "
            "mock_sampler picks uniformly with rng.choice and nothing updates the odds, so mode "
            "collapse cannot occur and entropy regularisation cannot delay it",
        ),
        practice.Check(
            "FINDING: the reward mean does not trend either",
            abs(slope) < spread / 20,
            f"over {len(SEEDS)} seeds the {ROUNDS} rounds average "
            + ", ".join(f"{value:.4f}" for value in per_round)
            + f", with a per-round spread of {spread:.4f} and a fitted slope of {slope:+.2e} -- "
            "less than a twentieth of the noise. A single-seed run that happens to decline is "
            "that spread and not a measurement",
        ),
        practice.Check(
            "FINDING: the advantages are discarded even when they are informative",
            result["degenerate"] == 0 and result["groups"] == ROUNDS * len(PROMPTS),
            f"at group_size={GROUP}, {result['degenerate']} of the {result['groups']} groups are "
            "degenerate -- every one has a non-zero advantage vector that a policy gradient "
            "could consume. The loop computes them correctly and returns a summary dictionary. "
            "What is missing from this self-improvement loop is not the entropy term",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
