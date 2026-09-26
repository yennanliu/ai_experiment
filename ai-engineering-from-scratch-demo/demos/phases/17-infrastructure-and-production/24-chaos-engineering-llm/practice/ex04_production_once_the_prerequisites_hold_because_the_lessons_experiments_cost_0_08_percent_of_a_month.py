"""Exercise 4 — production once the prerequisites hold, because the lesson's experiments cost 0.08 percent of a month.

    Argue whether chaos should run in production or only staging. When is
    production the right answer?

Reading of the exercise: the argument against production is that it spends
users' error budget, so it is made in the lesson's own units -- its 99.9% SLO,
its 0.05% expected error rate, its three experiments -- over a 30-day budget.
Two costs matter: what the experiment is planned to spend, and what it spends
if it goes wrong (everything in the blast radius fails until the kill switch
fires). "When" then becomes a rule, applied to four team profiles.

**ANSWER: production, once the five prerequisites hold -- the planned cost is
negligible.** At the lesson's rates the three experiments together spend
0.081% of a month's error budget, 0.16% of the half the baseline leaves. The
malformed prompt would have to run 5,400 minutes -- 3.75 days -- and the
provider 429 4,800 to spend that half. Staging cannot give what the lesson's
experiments are for: the real provider's 429s, the real traffic mix behind a
tokenizer stall, production concurrency behind a KV storm.

**FINDING: the worst case, not the plan, sets the blast cap and the abort
time.** If the blast radius fails outright, the remaining budget lasts 432
minutes at 5% blast and 72 at 30%. With a 5-minute abort the 30% experiment
risks 6.9% of it, the 5% one 1.2%. Production is the right answer when
blast x time-to-abort fits the budget left -- here, a worst case of at most
10% of it. A team whose abort is a person paged (15 minutes) passes at 5%
blast and fails at 30%: the 30% experiment goes to staging.

**FINDING: nothing in the lesson's code knows where an experiment runs.**
`Experiment` has no environment field and `run_experiment` reads no
prerequisite, budget remaining or incident count. The skill's refusal rules --
five prerequisites, one green staging run first, stabilise above 2 incidents a
week -- live only in prose; `where()` below is them plus the worst-case test.

Structure: arithmetic on the reference constants; `where()` is the decision.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "24-chaos-engineering-llm"
DAYS, MIN_PER_DAY, ABORT_MIN = 30, 1440, 5
PREREQS = ("slo", "observability", "rollback", "runbooks", "on_call")
PROFILES = {  # prerequisites held, staging green, incidents/week, budget left, abort min
    "mature": (PREREQS, True, 1, 0.50, 5),
    "no rollback": (PREREQS[:2] + PREREQS[3:], True, 1, 0.50, 5),
    "firefighting": (PREREQS, True, 3, 0.50, 5),
    "manual abort": (PREREQS, True, 1, 0.50, 15),
}


def month_share(ref, rate, blast, minutes):
    """Fraction of a 30-day error budget spent by `rate` errors on `blast` traffic."""
    return rate * blast * minutes / MIN_PER_DAY / (ref.ERROR_BUDGET_PER_DAY * DAYS)


def where(ref, profile, blast):
    held, staging_green, incidents, left, abort = profile
    if set(held) != set(PREREQS) or not staging_green:
        return "staging"
    if incidents > 2:
        return "stabilise first"
    worst = month_share(ref, 1.0, blast, abort)
    return "production" if worst <= 0.1 * left else "staging"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    exps = ref.EXPERIMENTS
    left = 1 - ref.EXPECTED_ERROR_RATE / ref.ERROR_BUDGET_PER_DAY
    planned = sum(month_share(ref, e.induced_error_rate, e.blast_radius_pct, e.duration_min)
                  for e in exps)
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-chaos-plan.md").read_text()
    return {
        "left": left, "planned": planned,
        "minutes_to_left": {e.name: left / month_share(ref, e.induced_error_rate,
                                                        e.blast_radius_pct, 1) for e in exps},
        "worst_minutes": {b: left / month_share(ref, 1.0, b, 1) for b in (0.05, 0.30)},
        "worst_abort": {b: month_share(ref, 1.0, b, ABORT_MIN) / left for b in (0.05, 0.30)},
        "where": {name: (where(ref, p, 0.30), where(ref, p, 0.05)) for name, p in PROFILES.items()},
        "fields": [f.name for f in dataclasses.fields(ref.Experiment)],
        "skill_rules": all(s in skill for s in ("green in staging", ">2/week", "five prerequisites")),
    }


def verify(result):
    mins, worst, abort = result["minutes_to_left"], result["worst_minutes"], result["worst_abort"]
    return [
        practice.Check(
            "ANSWER: production, once the five prerequisites hold -- the planned cost is negligible",
            all([round(result["planned"], 5) == 0.00081, result["left"] == 0.5,
                 round(mins["malformed prompt tokenizer stall"]) == 5400,
                 round(mins["provider 429 fallback"]) == 4800]),
            f"planned cost {result['planned']:.5f} of a month's budget, "
            f"{result['planned'] / result['left']:.4f} of what is left; minutes to spend "
            f"the rest {({k: round(v) for k, v in mins.items()})}",
        ),
        practice.Check(
            "FINDING: the worst case, not the plan, sets the blast cap and the abort time",
            all([[round(v) for v in worst.values()] == [432, 72],
                 [round(v, 3) for v in abort.values()] == [0.012, 0.069],
                 result["where"]["mature"] == ("production", "production"),
                 result["where"]["manual abort"] == ("staging", "production")]),
            f"a failed blast radius spends what is left in {worst} minutes; a 5-minute "
            f"abort risks {abort}; (30% blast, 5% blast) by profile {result['where']}",
        ),
        practice.Check(
            "FINDING: nothing in the lesson's code knows where an experiment runs",
            all([not [f for f in result["fields"] if "env" in f], result["skill_rules"]]),
            f"Experiment fields {result['fields']}; the skill's refusal rules are prose only",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
