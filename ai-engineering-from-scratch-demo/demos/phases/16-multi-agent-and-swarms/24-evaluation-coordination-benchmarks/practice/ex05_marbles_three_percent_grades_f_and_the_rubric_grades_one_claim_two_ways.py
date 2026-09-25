"""Exercise 5 — MARBLE's 3% grades F, and the rubric grades one claim two ways.

    Apply the benchmark-claims checklist to one recent multi-agent paper's
    headline result. What grade would you give the claim?

Reading of the exercise: the checklist is the lesson's own grader,
`outputs/skill-benchmark-reader.md` -- six checks, letter grades by weakness
count, and hard rejects -- so it is encoded as a function and run on the
headline it is most naturally aimed at, MultiAgentBench's "cognitive planning
improves milestone achievement rates by 3%" (arXiv:2503.01935, abstract),
with the lesson's own scorecard graded alongside as a control.

The facts graded, each checked against the paper: the comparison is against
vanilla, chain-of-thought and group-discussion planning (§3.1.1) -- variants
of the same system; it is measured in the research scenario only (§4.3); no
run count, deviation, interval or test is reported for it; token cost is
reported for the protocols (Figure 5) but not for the planning strategies
(Figure 6); and the model behind Figure 6 is not identified, so its training
cutoff cannot be set against the benchmark's release.

**ANSWER: F.** Five of six checks fail -- only benchmark and split pass -- which
alone is a D, and "no statistics" is a disqualifier, which makes it F. The
paper is ACL 2025, so the preprint downgrade does not apply. It is a
suggestive ablation, not evidence that cognitive planning helps.

**FINDING: the lesson's own scorecard fails its own rubric -- D by count, F
by the hard rejects.** The contamination check and cost column pass and the
held split is named, but there are 40 seen tasks with no interval, the random
"baseline" is the literal 0.15, and there is one task family: three
weaknesses, two of them disqualifying.

**FINDING: the rubric grades a claim with only its baseline missing as both
B and F.** Grades are counted -- "B: one weakness" -- while "claims without
baseline comparison" is a hard reject. The same claim lands on either letter
depending on which rule is read first; this encoding applies the hard rejects
first, which is the only order under which they mean anything.

Structure: `grade()` is the rubric; `CLAIMS` holds the facts per claim,
with the scorecard's facts read from the reference module itself.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "24-evaluation-coordination-benchmarks"
CHECKS = ("split", "contamination", "baseline", "statistics", "diversity", "cost")
DISQUALIFY = ("statistics", "baseline")
MARBLE = {"split": True, "contamination": False, "baseline": False,
          "statistics": False, "diversity": False, "cost": False, "preprint": False}


def grade(facts):
    """The skill's letters: hard rejects first, then count the weaknesses."""
    weak = [c for c in CHECKS if not facts[c]]
    if any(c in weak for c in DISQUALIFY):
        letter = "F"
    else:
        letter = "ABCD"[min(len(weak), 3)]
        if facts.get("preprint") and letter != "D":
            letter = "ABCDF"["ABCD".index(letter) + 1]
    return letter, weak


def count_only(facts):
    return "ABCD"[min(sum(not facts[c] for c in CHECKS), 3)]


def scorecard_facts(ref):
    """What the lesson's own scorecard discloses, read from its code."""
    source = inspect.getsource(ref.format_scorecard)
    constant = {ref.random_baseline(random.Random(s)) for s in range(20)} == {0.15}
    return {"split": "acc(held)" in source, "contamination": "contamination" in source,
            "baseline": not constant, "statistics": False,
            "diversity": False,  # every task is run_task on one toy task family
            "cost": "cost/t" in source, "n_seen": "n_seen=40" in source, "preprint": False}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    card = scorecard_facts(ref)
    baseline_only = dict.fromkeys(CHECKS, True) | {"baseline": False}
    return {
        "marble": grade(MARBLE), "marble_count": count_only(MARBLE),
        "card": grade(card), "card_facts": card, "card_count": count_only(card),
        "baseline_only": (grade(baseline_only)[0], count_only(baseline_only)),
    }


def verify(result):
    card = result["card_facts"]
    return [
        practice.Check(
            "ANSWER: F",
            result["marble"][0] == "F" and result["marble_count"] == "D"
            and result["marble"][1] == list(CHECKS[1:]),
            f"failing checks {result['marble'][1]}: five weaknesses is a "
            f"{result['marble_count']} by count, and no statistics disqualifies",
        ),
        practice.Check(
            "FINDING: the lesson's own scorecard fails its own rubric",
            result["card_count"] == "D" and result["card"][0] == "F"
            and card["n_seen"] and not card["baseline"],
            f"the scorecard runs n_seen=40 with no interval, a random baseline that is "
            f"the literal 0.15, and one task family -- {result['card_count']} by count, "
            f"{result['card'][0]} once the constant baseline is read as missing",
        ),
        practice.Check(
            "FINDING: the rubric grades a baseline-only gap as both B and F",
            result["baseline_only"] == ("F", "B"),
            f"a claim missing only its baseline is {result['baseline_only'][1]} by "
            f"weakness count and {result['baseline_only'][0]} by the hard reject",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
