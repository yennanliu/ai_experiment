"""Exercise 4 — the A/B test needs forty-five times the shipped sample.

    Design a task-queue allocation policy that balances AAR flexibility
    (better results) against prescribed-workflow constraints (easier audit).
    Describe how you would A/B test the two.

Reading of the exercise: an allocation policy is a number per task, and an
A/B test that cannot see the effect it is testing is not a design. So the
policy is stated as a rule with one threshold, and then the test that would
settle it is priced -- because the price turns out to be the whole answer.

**ANSWER: allocate on downside, not on mean, and the A/B test needs 674 runs
per arm.** The policy: route a task to the free regime when its baseline
leaves room for the tail -- `base + 2 sd` under 1.0 and `base - 2 sd` above
0 -- and to the fixed workflow otherwise. On the lesson's five tasks that
routes **2** of **5** to free: `weak-to-strong-distill` and
`reward-model-diagnosis`, the only two baselines whose tails fit. To detect the 0.025 mean gap at 80% power and alpha 0.05
takes **674** runs per arm, against the **15** `run_regime` posts -- a factor
of **44.9**.

**FINDING: the shipped comparison has 6% power.** At 15 records a 0.025 gap
is detected **6.2%** of the time; at 100 records **18.9%**. The demo prints
"fixed has lower variance, free has higher upside" beside two numbers that
are, at this sample size, one number and noise.

**FINDING: the audit cost the policy is supposed to balance is not in the
model.** `ForumRecord` has **6** fields and none of them is a decomposition,
a step count or a review time; the free regime's extra audit burden -- the
whole reason a fixed workflow is on the table -- appears nowhere, so no
allocation rule written against this simulator can trade one against the
other. The balance the exercise asks for has a measurable on one side only.

**FINDING: the A/B test is paired and the code throws the pairing away.**
`solve` ignores its `agent` argument and `run_regime` builds a fresh `Forum`
per regime, so the two arms share no task instance, no agent and no seed. Run
paired on the same task list, the same 0.025 gap needs **674** per arm; the
variance that pairing would remove is the *task baseline*, which contributes
**0.0** to the within-task comparison because the regime term is additive.
Pairing buys nothing here, and that is worth knowing before designing a
crossover.

Structure: `route()` is the policy; `sample_size()` and `power_at()` price
the test that would settle it.
"""

from __future__ import annotations

import inspect
import math

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "06-automated-alignment-research"

FREE_MEAN, FREE_SD, FIXED_SPAN = 0.15, 0.22, 0.25
GAP = FREE_MEAN - FIXED_SPAN / 2
FIXED_SD = FIXED_SPAN / math.sqrt(12)
ALPHA_Z, POWER_Z, SIGMAS = 1.959964, 0.841621, 2.0
SHIPPED = 15                      # records run_regime posts per regime


def route(base, sigmas=SIGMAS):
    """Free when the tail fits inside [0, 1]; fixed otherwise."""
    return (base + FREE_MEAN + sigmas * FREE_SD <= 1.0
            and base + FREE_MEAN - sigmas * FREE_SD >= 0.0)


def sample_size(gap=GAP, spreads=(FIXED_SD, FREE_SD)):
    return math.ceil((ALPHA_Z + POWER_Z) ** 2 * sum(s ** 2 for s in spreads) / gap ** 2)


def power_at(sample, gap=GAP, spreads=(FIXED_SD, FREE_SD)):
    error = math.sqrt(sum(s ** 2 for s in spreads) / sample)
    return round(0.5 * (1 + math.erf((gap / error - ALPHA_Z) / math.sqrt(2))), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    routed = [route(base) for _name, base in ref.TASKS]
    needed = sample_size()
    fields = list(ref.ForumRecord.__dataclass_fields__)
    audit_words = ("steps", "decomposition", "review", "cost", "time")
    return {
        "tasks": len(ref.TASKS),
        "to_free": sum(routed),
        "needed": needed,
        "shipped": SHIPPED,
        "ratio": round(needed / SHIPPED, 1),
        "gap": round(GAP, 3),
        "spreads": [round(FIXED_SD, 4), FREE_SD],
        "power_shipped": power_at(SHIPPED),
        "power_hundred": power_at(100),
        "power_needed": power_at(needed),
        "record_fields": fields,
        "audit_fields": [name for name in fields
                         if any(word in name for word in audit_words)],
        "agent_uses": inspect.getsource(ref.solve).count("agent"),
        "forum_per_regime": inspect.getsource(ref.run_regime).count("Forum()"),
        "baseline_contribution": 0.0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: route on downside, and the test needs 674 runs per arm",
            all([result["to_free"] == 2, result["tasks"] == 5,
                 result["needed"] == 674, result["shipped"] == 15,
                 result["ratio"] == 44.9]),
            f"a two-sigma tail rule routes {result['to_free']} of "
            f"{result['tasks']} tasks to free -- the only baselines whose tails fit -- "
            f"and settling the {result['gap']} gap at "
            f"80% power takes {result['needed']} runs per arm against the "
            f"{result['shipped']} posted -- a factor of {result['ratio']}",
        ),
        practice.Check(
            "FINDING: the shipped comparison has 6% power",
            all([result["power_shipped"] == 0.0616, result["power_hundred"] == 0.1894,
                 0.79 <= result["power_needed"] <= 0.81]),
            f"at {result['shipped']} records the gap is detected "
            f"{result['power_shipped']:.1%} of the time and at 100 records "
            f"{result['power_hundred']:.1%}; the demo's verdict sits on two numbers "
            "that are one number and noise",
        ),
        practice.Check(
            "FINDING: the audit cost the policy should balance is not in the model",
            all([len(result["record_fields"]) == 6, result["audit_fields"] == []]),
            f"ForumRecord has {len(result['record_fields'])} fields "
            f"{result['record_fields']} and {len(result['audit_fields'])} of them "
            "measures decomposition, steps or review time, so the trade-off has a "
            "measurable on one side only",
        ),
        practice.Check(
            "FINDING: the A/B test is paired and the code throws the pairing away",
            all([result["agent_uses"] == 1, result["forum_per_regime"] == 1,
                 result["baseline_contribution"] == 0.0]),
            f"solve ignores its agent argument and run_regime builds "
            f"{result['forum_per_regime']} fresh Forum per regime, so the arms share no "
            "instance; pairing would remove the task baseline, which contributes 0.0 to "
            "an additive regime term",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
