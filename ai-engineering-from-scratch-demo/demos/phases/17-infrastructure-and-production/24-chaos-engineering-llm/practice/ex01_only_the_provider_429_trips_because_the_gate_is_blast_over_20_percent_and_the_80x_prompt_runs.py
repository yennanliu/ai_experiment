"""Exercise 1 — only the provider 429 trips, because the gate is blast over 20 percent and the 80x prompt runs.

    Run `code/main.py`. Which experiment trips the burn-rate gate and why?

Reading of the exercise: "which" is read off the shipped run, and "why" is
answered from the gate's own expression -- then held against the rule the
lesson states for it ("pause experiment if daily error-budget burn exceeds 2x
expected"), and against the skill file it ships, since a gate that trips for a
reason the lesson never names is part of the answer.

**ANSWER: only "provider 429 fallback" trips, at 30x burn and 30% blast.**
The gate is `burn_rate > 2.0 and blast_radius_pct > 0.2`. All three
experiments clear the burn half -- 4x, 30x, 80x the expected 0.05% error rate
-- so the verdict is decided by the blast half alone: 5%, 30%, 10%. Rerun at
20% blast, none trips; at 21%, all three do.

**FINDING: the experiment that burns fastest runs to completion.** The
malformed-prompt stall burns 80x per affected request; weighted by its blast
radius it takes the whole service to 8.9x expected against the aborted
experiment's 9.7x. It completes on the blast condition, which the lesson text
never states: its rule (Guardrails, Key Terms, Numbers) is burn over 2x alone,
and under that rule all three experiments pause. The 20% appears nowhere in
docs/en.md; the lesson's own skill file sets the blast cap at "< 30% of
fleet", which the aborted experiment's 30% already breaks.

**FINDING: the "burn rate" is measured against the baseline, not the budget,
and ignores duration.** `ERROR_BUDGET_PER_DAY` (0.1%) is printed and never
read; burn is divided by `EXPECTED_ERROR_RATE`. Against the budget, as SRE
burn rate is usually defined, the pod kill is exactly 2.0x -- not over 2 --
and the other two are 15x and 40x. `duration_min` is echoed only: a 500-minute
pod kill gets the same verdict as a 5-minute one.

**FINDING: read as a daily budget, nothing trips.** The provider experiment
spends 1.56% of one day's error budget (1.5% errors x 30% of traffic x 5 of
1440 minutes); on top of the expected 50% day that is 1.03x expected, far from
2x. The per-request multiples (4x-80x) and the daily burn the rule names are
different quantities.

Structure: `verdicts()` reruns the reference `run_experiment` on its own
`EXPERIMENTS` with one field replaced; the rest is arithmetic on those fields.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "24-chaos-engineering-llm"
MINUTES_PER_DAY = 1440


def verdicts(ref, **field):
    """Paused flags for every shipped experiment with one field overridden."""
    return [ref.run_experiment(dataclasses.replace(e, **field))["paused_by_safety_plane"]
            for e in ref.EXPERIMENTS]


def service_burn(ref, e):
    """Whole-service error rate over expected: blast share at the induced rate."""
    rate = e.induced_error_rate * e.blast_radius_pct
    rate += ref.EXPECTED_ERROR_RATE * (1 - e.blast_radius_pct)
    return rate / ref.EXPECTED_ERROR_RATE


def daily_share(ref, e):
    """Fraction of one day's error budget the experiment's extra errors spend."""
    extra = e.induced_error_rate * e.blast_radius_pct * e.duration_min / MINUTES_PER_DAY
    return extra / ref.ERROR_BUDGET_PER_DAY


def columns(ref):
    """One pass over the shipped experiments: the reference verdict and the arithmetic."""
    expected_day = ref.EXPECTED_ERROR_RATE / ref.ERROR_BUDGET_PER_DAY
    cols = {k: [] for k in ("tripped", "burn", "blast", "service", "vs_budget", "share")}
    for e in ref.EXPERIMENTS:
        run = ref.run_experiment(e)
        if run["paused_by_safety_plane"]:
            cols["tripped"].append(e.name)
        cols["burn"].append(run["burn_rate_x"])
        cols["blast"].append(e.blast_radius_pct)
        cols["service"].append(round(service_burn(ref, e), 2))
        cols["vs_budget"].append(round(e.induced_error_rate / ref.ERROR_BUDGET_PER_DAY, 2))
        cols["share"].append(daily_share(ref, e))
    cols["daily_x"] = [(expected_day + share) / expected_day for share in cols["share"]]
    return cols


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON, "en")
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-chaos-plan.md").read_text()
    return {
        **columns(ref),
        "at_20": verdicts(ref, blast_radius_pct=0.20),
        "at_21": verdicts(ref, blast_radius_pct=0.21),
        "long": verdicts(ref, duration_min=500),
        "doc_20": "20%" in doc,
        "skill_cap": "< 30% of fleet" in skill,
    }


def verify(result):
    burn, service, share = result["burn"], result["service"], result["share"]
    return [
        practice.Check(
            "ANSWER: only the provider 429 fallback trips, at 30x burn and 30% blast",
            all([result["tripped"] == ["provider 429 fallback"], burn == [4.0, 30.0, 80.0],
                 result["at_20"] == [False] * 3, result["at_21"] == [True] * 3]),
            f"burn {burn}x, blast {result['blast']}; every experiment clears 2x, so the "
            "blast half decides: none trips at 20% blast, all three at 21%",
        ),
        practice.Check(
            "FINDING: the experiment that burns fastest runs to completion",
            all([service == [1.15, 9.7, 8.9], min(burn) > 2.0,
                 not result["doc_20"], result["skill_cap"]]),
            f"service-wide burn {service}x; the lesson's burn-over-2x rule pauses all "
            "three, the 20% blast condition is not in docs/en.md, and the skill caps "
            "blast at < 30% of fleet",
        ),
        practice.Check(
            "FINDING: the burn rate is measured against the baseline, not the budget, "
            "and ignores duration",
            all([result["vs_budget"] == [2.0, 15.0, 40.0],
                 result["long"] == [False, True, False]]),
            f"against the 0.1% budget burn is {result['vs_budget']}x (pod kill exactly 2.0, "
            f"not over 2); a 500-minute run gets verdicts {result['long']}",
        ),
        practice.Check(
            "FINDING: read as a daily budget, nothing trips",
            all([round(share[1], 5) == 0.01562, max(result["daily_x"]) < 1.04]),
            f"daily budget spent {[round(s, 5) for s in share]}, daily burn "
            f"{[round(x, 3) for x in result['daily_x']]}x expected, against a 2x pause",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
