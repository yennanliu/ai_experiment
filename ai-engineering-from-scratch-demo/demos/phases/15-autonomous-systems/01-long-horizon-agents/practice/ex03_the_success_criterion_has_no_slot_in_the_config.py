"""Exercise 3 — the success criterion has no slot in the config.

    Read METR's Time Horizon 1.1 blog post. Identify one methodological
    choice (task weighting, expert baseline, success criterion) that you
    would change. Write one paragraph explaining why.

Reading of the exercise: the paragraph is prose and lives in the README; what
belongs in a file is the part of the claim that can be measured. So this
solution takes the third of the three offered choices -- the success criterion
-- and prices the change in the lesson's own units instead of asserting it.

**ANSWER: report the 80% horizon beside the 50% one.** The 50% mark is the
median of a logistic fit, which is the point at which the model fails as often
as it succeeds; nobody deploys at the median. The lesson quotes "50%" **8**
times and "80%" **0**, and its config has **3** fields -- baseline hours,
baseline month, doubling months -- of which **0** is a success criterion.

**FINDING: exactly one of the five numeric functions takes a criterion, and it
is on the wrong half.** `max_steps_for_target` accepts `target`;
`horizon_at`, `months_to_cross`, `end_to_end_reliability` and `fmt_hours` do
not. The reliability half can be asked about any bar and the horizon half
cannot be asked about any, so the simulator can price a 95% trajectory but not
an 80% horizon.

**FINDING: the change costs a year and a bit, in the lesson's own unit.** If
the 80% horizon sits a factor k below the 50% one, today's 80% figure is the
50% figure from `7 * log2(k)` months ago: **7.0**, **14.0**, **16.25** and
**21.0** months of lag for k of 2, 4, 5 and 8. That is the number a
deployment argument needs, and it is one call to `months_to_cross` away.

**FINDING: that call reaches a branch the lesson never prints.** Its **4**
shipped targets -- 24, 48, 168 and 720 hours -- all sit above the 14-hour
baseline and return positive months. A target *below* the baseline returns a
negative month, which is exactly the "how far behind is the strict criterion"
question, and **0** of the shipped calls asks it.

Structure: `lag_months()` runs the criterion change through the lesson's own
projection; `criterion_slots()` counts where a success bar can be supplied.
"""

from __future__ import annotations

import dataclasses
import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "01-long-horizon-agents"

BASELINE_HOURS, DOUBLING_MONTHS = 14.0, 7.0
NUMERIC = ("horizon_at", "months_to_cross", "end_to_end_reliability",
           "max_steps_for_target", "fmt_hours")
SHIPPED_TARGETS = (24.0, 48.0, 168.0, 720.0)
FACTORS = (2, 4, 5, 8)          # candidate ratios between the 50% and 80% horizon


def criterion_slots(ref):
    """Which numeric functions can be handed a success probability at all."""
    return [name for name in NUMERIC
            if "target" in inspect.signature(getattr(ref, name)).parameters]


def lag_months(ref, factors=FACTORS):
    """Months between today's 50% horizon and today's stricter-criterion horizon."""
    cfg = ref.HorizonConfig(BASELINE_HOURS, 0, DOUBLING_MONTHS)
    return [-ref.months_to_cross(cfg, BASELINE_HOURS / k) for k in factors]


def config_fields(ref):
    return [field.name for field in dataclasses.fields(ref.HorizonConfig)]


def shipped_months(ref):
    """The month each of the lesson's own crossing targets is reached."""
    cfg = ref.HorizonConfig(BASELINE_HOURS, 0, DOUBLING_MONTHS)
    return [round(ref.months_to_cross(cfg, target), 2) for target in SHIPPED_TARGETS]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    docs = parity.doc_text(PHASE, LESSON)
    fields, shipped = config_fields(ref), shipped_months(ref)
    return {
        "says_50": docs.count("50%"),
        "says_80": docs.count("80%"),
        "config_fields": fields,
        "criterion_in_config": [name for name in fields if "success" in name],
        "numeric": list(NUMERIC),
        "slots": criterion_slots(ref),
        "lag": [round(value, 2) for value in lag_months(ref)],
        "factors": list(FACTORS),
        "shipped_targets": list(SHIPPED_TARGETS),
        "shipped_months": shipped,
        "negative_shipped": sum(value < 0 for value in shipped),
        "below_baseline": sum(t < BASELINE_HOURS for t in SHIPPED_TARGETS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the criterion is stated everywhere and parameterised nowhere",
            all([result["says_50"] == 8, result["says_80"] == 0,
                 len(result["config_fields"]) == 3,
                 result["criterion_in_config"] == []]),
            f"the lesson writes '50%' {result['says_50']} times and '80%' "
            f"{result['says_80']}, and HorizonConfig's {len(result['config_fields'])} "
            f"fields {result['config_fields']} contain no success criterion",
        ),
        practice.Check(
            "FINDING: one of the five numeric functions takes a criterion",
            all([result["slots"] == ["max_steps_for_target"],
                 len(result["numeric"]) == 5]),
            f"{len(result['slots'])} of {len(result['numeric'])} numeric functions "
            f"accepts a success bar -- {result['slots'][0]} -- so the reliability half "
            "can be asked about any threshold and the horizon half about none",
        ),
        practice.Check(
            "FINDING: the criterion change costs a year and a bit of horizon",
            all([result["lag"] == [7.0, 14.0, 16.25, 21.0],
                 result["factors"] == [2, 4, 5, 8]]),
            f"a stricter horizon a factor {result['factors']} below the median one is "
            f"the median horizon from {result['lag']} months ago, priced by the "
            "lesson's own 7-month doubling",
        ),
        practice.Check(
            "FINDING: that lag uses a branch the shipped calls never reach",
            all([result["negative_shipped"] == 0, result["below_baseline"] == 0,
                 result["shipped_months"] == [5.44, 12.44, 25.09, 39.79]]),
            f"all {len(result['shipped_targets'])} shipped targets sit above the "
            f"14-hour baseline and return {result['shipped_months']}, so "
            f"{result['negative_shipped']} of them exercises the negative-month branch "
            "the stricter criterion needs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
