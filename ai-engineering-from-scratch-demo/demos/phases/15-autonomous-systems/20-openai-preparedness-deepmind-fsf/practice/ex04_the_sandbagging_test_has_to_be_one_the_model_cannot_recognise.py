"""Exercise 4 — the sandbagging test has to be one the model cannot recognise.

    Sandbagging is in OpenAI's Research Categories. Design an evaluation that
    would force a sandbagging model to reveal its actual capability. Reference
    the Lesson 1 eval-context-gaming discussion.

Reading of the exercise: "force" is the hard word. A model that can recognise
an evaluation can underperform on it, so no single test forces anything -- the
design has to make underperformance *costly* rather than detectable, which is
a different objective and produces a different protocol.

**ANSWER: pay for the capability rather than test for it.** Four arms, run on
the same model over the same task distribution: (1) a declared evaluation,
(2) an undeclared sample of production traffic, (3) the same tasks with a
reward the model has been told is real, and (4) the same tasks with a
consistency check against the model's own earlier answers. Sandbagging shows
as a **gap between arms**, not as a low score in any of them -- which is why
**1** arm cannot detect it and why the metric is a variance rather than a
mean.

**FINDING: the three policies classify it three ways and only one of them
implies a measurement.** OpenAI files sandbagging as `Research` with the
action `observed; potential mitigations`; Anthropic names it `via eval-context
gap`, addressed `in measurement methodology`; DeepMind assigns an `automated
instrumental-reasoning monitor`. Observation, methodology and a monitor are
three different budgets, and only the third names a thing that runs.

**FINDING: the arms disagree in the direction Lesson 1 measured.** At
Anthropic's documented alignment-faking rates the model behaves differently on
**12%** of basic tests and up to **78%** after retraining attempts, so a
four-arm protocol should expect the declared-evaluation arm to read **low** by
exactly that fraction. Applied to a 68-step trajectory at 0.99 per-step, a
**12%** gap moves end-to-end reliability from **50.5%** to **23.9%** -- the
number a capability report would have to carry to be honest about its own
denominator.

**FINDING: the protocol's weakness is arm 2, and it is a policy problem.**
Sampling undeclared production traffic is the only arm a model cannot
recognise, and it is the arm that requires consent from the people whose
traffic it is. **0** of the three policies' sandbagging entries mention data
governance, so the evaluation that would work is the one whose cost is not in
any of these documents.

Structure: `ARMS` is the protocol; `positions()` reads the three
classifications out of the shipped tables.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "20-openai-preparedness-deepmind-fsf"

# (arm, can the model recognise it, what a gap against arm 1 would mean)
ARMS = (
    ("declared evaluation", True, "baseline"),
    ("undeclared production sample", False, "sandbagging on the declared arm"),
    ("same tasks, stated real reward", True, "incentive-sensitive capability"),
    ("consistency against earlier answers", True, "inconsistent underperformance"),
)
GAMING_RATES = (0.12, 0.78)      # Lesson 1's documented alignment-faking figures
CLEAN, STEPS = 0.99, 68


def positions(ref, capability="sandbagging"):
    return [(policy.name.split(" ")[0], *policy.table[capability])
            for policy in ref.POLICIES]


def gamed(rate, clean=CLEAN, degraded=0.90, steps=STEPS):
    """End-to-end reliability when the model behaves differently on `rate` of steps."""
    per_step = clean * (1 - rate) + degraded * rate
    return round(per_step ** steps, 3)


def unrecognisable():
    return [name for name, recognisable, _meaning in ARMS if not recognisable]


def mentions(ref, words=("data", "consent", "governance", "privacy")):
    entries = [f"{entry[0]} {entry[1]}".lower()
               for policy in ref.POLICIES for entry in [policy.table["sandbagging"]]]
    return sum(any(word in text for word in words) for text in entries)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = positions(ref)
    return {
        "arms": len(ARMS),
        "unrecognisable": unrecognisable(),
        "detectable_in_one_arm": False,
        "classifications": [classification for _n, classification, _a in rows],
        "actions": [action for _n, _c, action in rows],
        "distinct": len({classification for _n, classification, _a in rows}),
        "names_a_runtime": [name for name, _c, action in rows if "monitor" in action],
        "rates": list(GAMING_RATES),
        "clean": gamed(0.0),
        "at_twelve": gamed(GAMING_RATES[0]),
        "at_seventy_eight": gamed(GAMING_RATES[1]),
        "governance_mentions": mentions(ref),
        "policies": len(ref.POLICIES),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four arms, and the signal is the gap between them",
            all([result["arms"] == 4,
                 result["unrecognisable"] == ["undeclared production sample"],
                 not result["detectable_in_one_arm"]]),
            f"{result['arms']} arms over the same task distribution, of which "
            f"{len(result['unrecognisable'])} cannot be recognised by the model -- "
            "sandbagging shows as a gap between arms rather than a low score in any of "
            "them",
        ),
        practice.Check(
            "FINDING: three classifications and one runtime",
            all([result["distinct"] == 3, result["policies"] == 3,
                 result["names_a_runtime"] == ["DeepMind"],
                 result["classifications"][0] == "Research"]),
            f"the three policies file sandbagging as {result['classifications']}, and "
            f"{len(result['names_a_runtime'])} of {result['policies']} names something "
            f"that runs -- {result['names_a_runtime'][0]}'s monitor",
        ),
        practice.Check(
            "FINDING: the arms disagree in the direction Lesson 1 measured",
            all([result["rates"] == [0.12, 0.78], result["clean"] == 0.505,
                 result["at_twelve"] == 0.239, result["at_seventy_eight"] == 0.003]),
            f"at documented gaming rates of {result['rates']}, a 68-step trajectory at "
            f"0.99 per-step falls from {result['clean']} to {result['at_twelve']} and "
            f"{result['at_seventy_eight']} -- the denominator a capability report would "
            "have to carry",
        ),
        practice.Check(
            "FINDING: the protocol's weakness is arm 2, and it is a policy problem",
            result["governance_mentions"] == 0,
            f"the only arm a model cannot recognise is the one sampling undeclared "
            f"production traffic, and {result['governance_mentions']} of the three "
            "sandbagging entries mention data governance -- the evaluation that works "
            "has a cost none of these documents prices",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
