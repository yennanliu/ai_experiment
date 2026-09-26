"""Exercise 2 — the lanes split cleanly until "next hour", which the lesson calls batch and its SLA cannot meet.

    Pick three features in a real product you know. Triage each into
    interactive/semi/batch.

Reading of the exercise: the product is GitHub, and each feature is reduced
to the one number the lesson says decides the lane, how long the user will
wait. The lane rules are taken from the lesson in two ways, since it states
them twice. The Concept text is "user waits" -> interactive, "minutes" ->
semi, "next hour" or "by morning" -> batch. The SLA rules are the skill's
refusal of batch under a 60 s P99 and the 24-hour batch promise. Each lane
is then priced with the reference's cost functions.

**ANSWER: inline completion is interactive, code review semi, the weekly
digest batch.** Copilot inline completion, Copilot code review on a pull
request and the weekly repository digest email are the three. Completion is typed against, so its tolerance is under a second.
Review is requested and checked back on within about 15 minutes. The digest
is read days later. Both rule sets agree on all three.

**FINDING: the lesson's batch lane includes "next hour", which a 24-hour
SLA cannot promise.** A feature that tolerates one hour, such as labelling
newly opened issues, is batch by the Concept text and semi by the SLA rule.
The two readings disagree on every tolerance from 1 hour up to 24 hours. The
lesson puts typical P50 at 2-6 hours, so more than half of such batches would
miss that hour.

**FINDING: once caching is on, the lane is worth exactly 2x, not ~90%.**
`cost_batch_cache` is `cost_sync_cache * 0.5`, so moving a cached feature
from interactive to batch saves exactly 50% on all three reference
workloads. The skill's "~90% leaked spend" needs a sync run that was also
uncached, and even then the reference workloads leak 75.7%, 82.9% and 58.7%.
The semi lane has no price of its own: `code/main.py` has four
configurations and none is an async queue, so semi costs what interactive
costs.

Structure: `concept_lane()` and `sla_lane()` encode the two rule sets;
`leak()` prices one workload with the reference.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "15-batch-apis"
MINUTE, HOUR, DAY = 60, 3600, 86_400
FEATURES = {  # feature -> seconds the user will wait
    "Copilot inline completion": 0.5,
    "Copilot code review on a pull request": 15 * MINUTE,
    "weekly repository digest email": 7 * DAY,
}
PROBE = ("label newly opened issues within the hour", HOUR)
REF_RUNS = (
    (50_000, 4000, 2000, 200),
    (200_000, 1500, 300, 50),
    (1_000, 6000, 15_000, 2000),
)


def concept_lane(wait_s):
    """The Concept text: user waits / minutes / 'next hour' or 'by morning'."""
    return "interactive" if wait_s < MINUTE else "semi" if wait_s < HOUR else "batch"


def sla_lane(wait_s):
    """The skill: no batch under a 60 s P99, and batch only promises 24 hours."""
    return "interactive" if wait_s < MINUTE else "semi" if wait_s < DAY else "batch"


def leak(ref, run):
    """(sync+cache / batch+cache, share of an uncached sync bill batch+cache removes)."""
    sync, cached, stacked = (
        f(*run) for f in (ref.cost_sync, ref.cost_sync_cache, ref.cost_batch_cache)
    )
    return round(cached / stacked, 6), round(1 - stacked / sync, 3)


def triage():
    """Both rule sets on the three features, and where the rule sets disagree."""
    lanes = {f: (concept_lane(w), sla_lane(w)) for f, w in FEATURES.items()}
    split = [h for h in range(0, 49) if concept_lane(h * HOUR) != sla_lane(h * HOUR)]
    return {
        "lanes": lanes,
        "picked": [pair[0] for pair in lanes.values()],
        "agree": all(a == b for a, b in lanes.values()),
        "probe": (concept_lane(PROBE[1]), sla_lane(PROBE[1])),
        "split": (float(min(split)), float(max(split))),
    }


def pricing(ref):
    leaks = [leak(ref, run) for run in REF_RUNS]
    configs = sorted(n for n in dir(ref) if n.startswith("cost_"))
    return {
        "ratios": [r for r, _ in leaks],
        "shares": [s for _, s in leaks],
        "configs": configs,
        "queue_priced": any("queue" in n or "semi" in n for n in configs),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {**triage(), **pricing(ref)}


def verify(result):
    lanes, ratios, shares = result["lanes"], result["ratios"], result["shares"]
    return [
        practice.Check(
            "ANSWER: inline completion is interactive, code review semi, the weekly digest batch",
            result["picked"] == ["interactive", "semi", "batch"] and result["agree"],
            f"(concept lane, SLA lane) by feature {lanes}",
        ),
        practice.Check(
            "FINDING: the lesson's batch lane includes 'next hour', which a 24-hour SLA "
            "cannot promise",
            result["probe"] == ("batch", "semi") and result["split"] == (1.0, 23.0),
            f"'{PROBE[0]}' is {result['probe'][0]} by the Concept text and "
            f"{result['probe'][1]} by the SLA; the two disagree from {result['split'][0]}h "
            f"to {result['split'][1]}h of tolerance",
        ),
        practice.Check(
            "FINDING: once caching is on, the lane is worth exactly 2x, not ~90%",
            ratios == [2.0, 2.0, 2.0]
            and shares == [0.757, 0.829, 0.587]
            and not result["queue_priced"],
            f"sync+cache / batch+cache on the three reference workloads "
            f"{ratios}; against uncached sync batch+cache removes "
            f"{shares}; the only priced configurations are {result['configs']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
