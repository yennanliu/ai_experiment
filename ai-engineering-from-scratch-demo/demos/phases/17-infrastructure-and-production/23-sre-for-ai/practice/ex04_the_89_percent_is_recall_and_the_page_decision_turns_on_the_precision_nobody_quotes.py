"""Exercise 4 — the 89% is recall, and the page decision turns on the precision nobody quotes.

    Predictive detection fires at 12 min lead. What's your policy — pager,
    pre-drain, or both?

Reading of the exercise: a policy is chosen by what it costs, so each option
is scored in closed form over a month: outage-minutes left, pages sent and
drains run. The lesson supplies recall 0.89, the 12-minute lead, and the
30-minute investigation from its 3 a.m. story. The rest is assumed and named:
4 outages a month, a 5-minute page acknowledgement, half the causes pod-local
(like the lesson's KV-cache OOM, which a drain clears), a 3-minute drain, and
one prediction per minute of the month. `code/main.py` has no predictor, so
the model is built from the lesson's prose.

**ANSWER: both, but asymmetrically.** Drain on every fire, within exercise 2's
restart guard, and page only once precision is known to be high enough.

| policy | outage-min / month | predictive pages / month |
|---|---:|---:|
| none | 140.0 | 0 |
| pager | 97.3 | 3.56 + false alarms |
| pre-drain | 77.7 | 0 |
| both | 56.3 | 3.56 + false alarms |

Pre-drain alone beats pager alone, and it spends a pod-drain rather than a
human wakeup.

**FINDING: 89% is recall, and the page half of the policy is decided by
precision, which the lesson never states.** At one prediction a minute there
are 43,200 chances a month to fire wrongly. At a false-alarm rate of 1e-3 per
minute that is 43.2 false pages against 3.56 true ones, precision 7.6%. At
1e-4 it is 4.32, precision 45.2%. Only at 1e-5 does precision reach 89.2%. A
3-person rotation cannot absorb 1.5 false pages a night, so the pager is
added only after the false-alarm rate has been measured at or below 1e-4.

**FINDING: a 12-minute lead does not cover the lesson's own 30-minute
investigation.** Paged 12 minutes early, the engineer still needs 5 + 30
minutes, so a caught outage runs 23 minutes instead of 35. The page buys
exactly the lead time. With the lesson's "first 20 minutes" automated by AI
triage, 5 + 10 - 12 leaves 3 minutes. Prediction pays for paging only
alongside triage.

Structure: `outage_minutes()` scores each policy per predicted, missed,
pod-local and non-local outage; `precision()` sweeps the false-alarm rate.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "23-sre-for-ai"
RECALL, LEAD, DIAGNOSE = 0.89, 12, 30  # the lesson's numbers
OUTAGES, ACK, LOCAL, DRAIN = 4, 5, 0.5, 3  # assumptions, stated in the docstring
MINUTES = 30 * 24 * 60
POLICIES = ("none", "pager", "pre-drain", "both")


def minutes_left(policy, local, diagnose=DIAGNOSE):
    """Outage-minutes for one predicted outage under a policy."""
    unpredicted = ACK + diagnose
    drained = policy in ("pre-drain", "both") and local and DRAIN <= LEAD
    if drained:
        return 0
    paged = policy in ("pager", "both")
    return max(0, unpredicted - LEAD) if paged else unpredicted


def outage_minutes(policy, diagnose=DIAGNOSE):
    caught = OUTAGES * RECALL
    missed = OUTAGES * (1 - RECALL) * (ACK + diagnose)
    per = LOCAL * minutes_left(policy, True, diagnose)
    per += (1 - LOCAL) * minutes_left(policy, False, diagnose)
    return round(caught * per + missed, 2)


def precision(false_rate):
    true, false = OUTAGES * RECALL, false_rate * MINUTES
    return round(false, 2), round(true / (true + false), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "ref_predict": [n for n in dir(ref) if "predict" in n.lower()],
        "table": {p: outage_minutes(p) for p in POLICIES},
        "true_pages": round(OUTAGES * RECALL, 2),
        "precision": {f: precision(f) for f in (1e-3, 1e-4, 1e-5)},
        "pager_only": (ACK + DIAGNOSE, minutes_left("pager", False)),
        "with_triage": minutes_left("pager", False, diagnose=DIAGNOSE - 20),
    }


def verify(result):
    t, prec = result["table"], result["precision"]
    return [
        practice.Check(
            "ANSWER: both, asymmetrically -- drain on every fire, page once precision is known",
            t == {"none": 140.0, "pager": 97.28, "pre-drain": 77.7, "both": 56.34}
            and t["both"] < t["pre-drain"] < t["pager"] < t["none"] and not result["ref_predict"],
            f"outage-minutes a month {t}; pager and both add {result['true_pages']} "
            "true pages plus every false alarm; code/main.py has no predictor to run",
        ),
        practice.Check(
            "FINDING: 89% is recall; the page decision turns on precision",
            prec == {1e-3: (43.2, 0.076), 1e-4: (4.32, 0.452), 1e-5: (0.43, 0.892)},
            f"(false pages / month, precision) by per-minute false-alarm rate: {prec}",
        ),
        practice.Check(
            "FINDING: a 12-minute lead does not cover the lesson's 30-minute investigation",
            result["pager_only"] == (35, 23) and result["with_triage"] == 3,
            f"paged early, a caught outage runs {result['pager_only'][1]} of "
            f"{result['pager_only'][0]} minutes; with the first 20 minutes automated, "
            f"{result['with_triage']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
