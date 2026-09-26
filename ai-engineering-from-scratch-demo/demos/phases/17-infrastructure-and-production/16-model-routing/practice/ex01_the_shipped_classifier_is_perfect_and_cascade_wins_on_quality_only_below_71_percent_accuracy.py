"""Exercise 1 — the shipped classifier is perfect, and cascade wins on quality only below 71% accuracy.

    Run `code/main.py`. At what accuracy floor does cascade beat pre-route?

Reading of the exercise: "accuracy" is the pre-route classifier's accuracy,
and "beat" is read on both of the axes the module reports -- cost and mean
quality. The shipped PRE_ROUTE has no accuracy knob, so one is added: each
request is routed correctly with probability a and to the other model
otherwise. Cost and quality are then exact expectations over the reference's
own 1000-request workload, `cost_of` and `quality`, so both are linear in a
and the crossovers are solved, not searched.

**ANSWER: below 71.3% classifier accuracy cascade has the higher quality,
and it is cheaper at every accuracy above 64.1%.** The shipped run gives
NO_ROUTE $7.52, PRE_ROUTE $5.38 at 99.37% quality, CASCADE $4.42 at 98.24%.
A noisy pre-route falls to cascade's quality at a = 0.713 and to its cost
at a = 0.641, because misrouting sends hard queries to the cheap model and
that is cheaper too. So cascade beats pre-route on both axes only for
classifiers between 64.1% and 71.3% accurate; above that it is a trade
(cheaper, worse), below it pre-route is the cheaper, worse one.

**FINDING: the shipped pre-route classifier is perfect, and the shipped
cascade has the lower quality.** PRE_ROUTE reads `q.difficulty` directly, so
the printout is pre-route at a = 1.0. The module's closing line says CASCADE
"guarantees quality floor" and the lesson calls it the "Best quality floor",
yet it scores 98.24% against pre-route's 99.37%: its "confidence" also reads
the label -- simple always kept, hard always escalated, medium on a 50% coin
-- so it keeps 142 of 276 medium queries on the cheap model at 0.92.

**FINDING: the cascade is that same routing plus a wasted cheap call on
every escalation.** Of its $4.42, $0.29 (6.6%) is the cheap run on the 229
requests it then escalates; a pre-route making the identical choices costs
$4.13 at the identical quality.

**FINDING: the ensemble the lesson says the code simulates is not there.**
"Use It" says `main.py` simulates pre-route, cascade and ensemble; the
module has NO_ROUTE instead, and `simulate("ENSEMBLE", ...)` returns cost 0
and quality 0 without an error.

Structure: `noisy_pre_route()` is the expectation at accuracy a;
`crossover()` solves the linear equation; `cascade_split()` replays the
reference's cascade rule with its seed to price the wasted calls.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "16-model-routing"


def load_ref():
    return parity.load_reference(PHASE, LESSON, "main")


def noisy_pre_route(ref, reqs, a):
    """(cost, mean quality) when each request is routed correctly with probability a."""
    cost = qual = 0.0
    for q in reqs:
        right = "cheap" if q.difficulty == "simple" else "frontier"
        wrong = "frontier" if right == "cheap" else "cheap"
        cost += a * ref.cost_of(right, q) + (1 - a) * ref.cost_of(wrong, q)
        qual += a * ref.quality(right, q) + (1 - a) * ref.quality(wrong, q)
    return cost, qual / len(reqs)


def crossover(at_one, at_zero, target):
    """a at which a*at_one + (1-a)*at_zero equals target."""
    return (target - at_zero) / (at_one - at_zero)


def cascade_split(ref, reqs):
    """The reference cascade's decisions, replayed with its own seed: (escalated, kept)."""
    rng, escalated, kept = random.Random(11), [], []
    for q in reqs:
        medium_kept = q.difficulty == "medium" and rng.random() < 0.5
        (kept if q.difficulty == "simple" or medium_kept else escalated).append(q)
    return escalated, kept


def solve():
    ref = load_ref()
    reqs = ref.make_workload()
    rows = {
        p: ref.simulate(p, reqs)
        for p in ("NO_ROUTE", "PRE_ROUTE", "CASCADE", "ENSEMBLE")
    }
    one, zero, cas = (
        noisy_pre_route(ref, reqs, 1.0),
        noisy_pre_route(ref, reqs, 0.0),
        rows["CASCADE"],
    )
    escalated, kept = cascade_split(ref, reqs)
    waste = sum(ref.cost_of("cheap", q) for q in escalated)
    return {
        "rows": {
            p: (round(r["cost"], 2), round(r["mean_quality"], 4), r["escalated"])
            for p, r in rows.items()
        },
        "a_quality": round(crossover(one[1], zero[1], cas["mean_quality"]), 3),
        "a_cost": round(crossover(one[0], zero[0], cas["cost"]), 3),
        "perfect": round(one[0], 6) == round(rows["PRE_ROUTE"]["cost"], 6),
        "medium_kept": sum(q.difficulty == "medium" for q in kept),
        "medium": sum(q.difficulty == "medium" for q in reqs),
        "escalated": len(escalated),
        "waste": round(waste, 2),
        "same_routing": round(cas["cost"] - waste, 2),
        "claims": all(
            s in parity.doc_text(PHASE, LESSON)
            for s in ("Best quality floor", "ensemble")
        ),
    }


def verify(result):
    rows = result["rows"]
    return [
        practice.Check(
            "ANSWER: below 71.3% classifier accuracy cascade has the higher quality",
            result["a_quality"] == 0.713 and result["a_cost"] == 0.641,
            f"shipped rows {rows}; a noisy pre-route meets cascade's quality at "
            f"a={result['a_quality']} and its cost at a={result['a_cost']}, so cascade "
            "wins on both axes only between those accuracies",
        ),
        practice.Check(
            "FINDING: the shipped pre-route classifier is perfect, and the shipped cascade "
            "has the lower quality",
            result["perfect"]
            and rows["CASCADE"][1] < rows["PRE_ROUTE"][1]
            and result["claims"],
            f"PRE_ROUTE equals a=1.0; CASCADE {rows['CASCADE'][1]} < PRE_ROUTE "
            f"{rows['PRE_ROUTE'][1]} because {result['medium_kept']} of {result['medium']} "
            "medium queries stay on the cheap model",
        ),
        practice.Check(
            "FINDING: the cascade is that same routing plus a wasted cheap call on every escalation",
            result["escalated"] == rows["CASCADE"][2] == 229
            and result["waste"] == 0.29,
            f"${result['waste']} of ${rows['CASCADE'][0]} is the cheap run on "
            f"{result['escalated']} escalations; the same routing as a pre-route costs "
            f"${result['same_routing']}",
        ),
        practice.Check(
            "FINDING: the ensemble the lesson says the code simulates is not there",
            rows["ENSEMBLE"] == (0.0, 0.0, 0),
            f"simulate('ENSEMBLE') returns {rows['ENSEMBLE']} (cost, quality, escalated) silently",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
