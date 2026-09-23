"""Exercise 5 — ninety percent recall costs eighty-one percent of a full read.

    Propose a human-review protocol for research-agent outputs that scales
    better than "a PhD reads every paper." Identify the bottleneck and design
    around it.

Reading of the exercise: "scales better" needs a curve, not an adjective, so
the protocol is written as a two-arm policy with one free parameter and then
priced -- reviewer effort against flaws caught -- on the same simulator the
rest of this lesson measures. The bottleneck is taken to be reviewer-hours
per submission, because that is the only quantity in the system that does not
move when compute does.

**ANSWER: triage on provenance, sample the rest.** Read every submission
whose experiment was retry-recovered; sample the others at rate s. At
**s = 0** that is **28.5%** of the reading for **61.7%** of the flaws; at
**s = 0.25**, **46.4%** for **71.3%**; at **s = 0.5**, **64.2%** for
**80.9%**. The first two thirds of recall cost under a third of the effort.

**FINDING: the last third of recall costs nearly all of the budget.** To
reach **90%** recall the sampling rate has to be **0.739**, which is
**81.3%** of a full read -- the protocol saves **18.7%** and no more.
Everything past the flagged pile is uniform in the flaw rate, so the second
arm buys recall at exactly the base rate and there is no better place to
point it.

**FINDING: the bottleneck is a ratio, and only one side of it moves.** The
loop submits **1** paper per **2.91** runs, and runs are compute. Reviewers
are not. So the protocol's parameter is not really `s` -- it is
`reviewer-hours / submission-rate`, and any honest version of it pins the
sampling rate to a fixed review budget and lets recall fall out, rather than
the reverse.

**FINDING: improving the agent makes the cheap arm expensive.** Raising
`retry_recovery` from 0.55 to **1.0** takes the flagged share from **28.5%**
to **41.7%** of submissions, so the same protocol's zero-sampling cost rises
by half while its recall rises to **74.2%**. A better agent does not shrink
the review queue here; it moves the queue into the arm that has to be read in
full.

Structure: `arms()` splits submissions on the provenance flag; `price()`
turns a sampling rate into (effort, recall).
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "05-ai-scientist-v2"

TRIALS, SEED = 20000, 42
RATES = (0.0, 0.25, 0.5, 1.0)
TARGET_RECALL = 0.90


def submissions(ref, config):
    random.seed(SEED)
    rows = [ref.run_one(config) for _ in range(TRIALS)]
    return [row for row in rows if row.submitted], len(rows)


def arms(rows):
    """(flagged share, flaws in the flagged arm, flaws in the sampled arm)."""
    flagged = [row for row in rows if row.has_experiment_flaw]
    rest = [row for row in rows if not row.has_experiment_flaw]
    return (len(flagged) / len(rows),
            sum(row.polished_but_flawed for row in flagged),
            sum(row.polished_but_flawed for row in rest))


def price(rows, rate):
    flagged_share, caught, remaining = arms(rows)
    effort = flagged_share + (1 - flagged_share) * rate
    total = caught + remaining
    return round(effort, 3), round((caught + remaining * rate) / total, 3)


def rate_for(rows, recall):
    """The sampling rate that reaches a target recall."""
    _flagged_share, caught, remaining = arms(rows)
    total = caught + remaining
    return round((recall * total - caught) / remaining, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, trials = submissions(ref, ref.LoopConfig())
    needed = rate_for(rows, TARGET_RECALL)
    recovered, _trials = submissions(ref, ref.LoopConfig(retry_recovery=1.0))
    return {
        "curve": [price(rows, rate) for rate in RATES],
        "rates": list(RATES),
        "needed_rate": needed,
        "needed_effort": price(rows, needed)[0],
        "saved": round(1 - price(rows, needed)[0], 3),
        "runs_per_submission": round(trials / len(rows), 2),
        "flagged_share": round(arms(rows)[0], 3),
        "recovered_flagged": round(arms(recovered)[0], 3),
        "recovered_curve": price(recovered, 0.0),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 28.5% of the reading buys 61.7% of the flaws",
            all([result["curve"] == [(0.285, 0.617), (0.464, 0.713),
                                     (0.642, 0.809), (1.0, 1.0)],
                 result["rates"] == [0.0, 0.25, 0.5, 1.0]]),
            f"reading every retry-recovered paper and sampling the rest at "
            f"{result['rates']} costs and catches {result['curve']} -- the first two "
            "thirds of recall for under a third of the effort",
        ),
        practice.Check(
            "FINDING: the last third of recall costs nearly all of the budget",
            all([result["needed_rate"] == 0.739, result["needed_effort"] == 0.813,
                 result["saved"] == 0.187]),
            f"{TARGET_RECALL:.0%} recall needs a sampling rate of "
            f"{result['needed_rate']}, which is {result['needed_effort']:.1%} of a full "
            f"read -- the protocol saves {result['saved']:.1%} and no more, because the "
            "sampled arm is uniform in the flaw rate",
        ),
        practice.Check(
            "FINDING: the bottleneck is a ratio, and only one side of it moves",
            all([result["runs_per_submission"] == 2.91,
                 result["flagged_share"] == 0.285]),
            f"the loop submits one paper per {result['runs_per_submission']} runs and "
            f"runs are compute, so the real parameter is reviewer-hours per submission "
            f"and the flagged arm -- {result['flagged_share']:.1%} -- is the part that "
            "cannot be sampled",
        ),
        practice.Check(
            "FINDING: improving the agent makes the cheap arm expensive",
            all([result["recovered_flagged"] == 0.417,
                 result["recovered_curve"] == (0.417, 0.742),
                 result["recovered_flagged"] > result["flagged_share"]]),
            f"raising retry_recovery to 1.0 takes the flagged share "
            f"{result['flagged_share']} to {result['recovered_flagged']}, so the "
            f"zero-sampling arm costs {result['recovered_curve'][0]} for "
            f"{result['recovered_curve'][1]} recall -- the queue moves into the arm "
            "that must be read in full",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
