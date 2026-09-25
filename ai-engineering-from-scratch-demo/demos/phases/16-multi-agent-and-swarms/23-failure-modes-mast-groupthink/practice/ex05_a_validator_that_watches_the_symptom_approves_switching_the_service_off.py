"""Exercise 5 — a validator that watches the symptom approves switching the service off.

    Design a STRATUS-style detection-diagnosis-validation trio for a specific
    multi-agent system you know. Which symptoms does detection watch for? What
    mitigations does diagnosis recommend? How does validation confirm they work?

Reading of the exercise: the system is the lesson's own order -> payment
pipeline, so the trio can be run rather than described. Detection watches
calls per request; diagnosis uses the reference's `categorize_incident` and
`detect_groupthink`; validation is built twice -- once on the symptom, once on
the objective -- because "confirm they work" depends on which one it reads.

**ANSWER: detection watches retry amplification, diagnosis names the
cascade, and validation must check the goal, not the symptom.** Detection
fires at 1.64 calls per request against a 1.5 budget. Diagnosis gets
`cascade` from the Groupthink table and "unknown" from MAST -- the taxonomy
the lesson calls the reference has no category for the incident it demos --
and recommends a circuit breaker. A symptom validator then passes it:
amplification falls to 0.70. An objective validator in STRATUS's
transactional-no-regression spirit rejects it: success rate falls from 0.945
to 0.565.

**FINDING: the symptom validator's favourite fix is turning traffic off.**
Across the lesson's threshold sweep, amplification is lowest at threshold
0.05 -- 26 calls for 200 requests, 0.13 per request -- which serves 25 of 200.
Every setting passes the symptom check except 0.9 and above; the one it
ranks best is the one that serves least.

**FINDING: whether the breaker is a fix depends on the load model, and only
the objective validator can tell.** On the shipped lifetime-counter load it
is rejected (189 -> 113 successes). With load over the last 20 requests the
unprotected run collapses to 71 successes and the breaker lifts it to 176, so
the same validator accepts it. The symptom validator accepts both.

**FINDING: STRATUS is not what the lesson describes.** arXiv:2506.02009's
abstract names detection, diagnosis and *mitigation* agents plus a transactional
no-regression safety property, and reports "at least 1.5 times" the
mitigation success of state-of-the-art SRE agents on AIOpsLab and ITBench --
cloud incidents, not multi-agent LLM failures. The lesson's "validation
agent" does not appear in that list.

Structure: `detect()`, `diagnose()` and the two validators are small
functions over run records from the reference simulator and exercise 1's
`storm()`, which varies only the load model.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "23-failure-modes-mast-groupthink"
HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
BUDGET, REQUESTS = 1.5, 200


def amplification(run, requests=REQUESTS):
    return round(run[0] / requests, 2)


def detect(run):
    return {"retry_amplification": amplification(run) > BUDGET}


def diagnose(ref, symptoms):
    groupthink = [code for code, _ in ref.detect_groupthink(symptoms)]
    remedy = "circuit breaker" if "cascade" in groupthink else None
    return ref.categorize_incident(symptoms)[0], groupthink, remedy


def by_symptom(after, requests=REQUESTS):
    return amplification(after, requests) <= BUDGET


def by_objective(before, after):
    """No-regression: the mitigation may not serve fewer requests than the baseline."""
    return after[1] >= before[1]


def solve():
    ref = EX01.load_ref()
    off = ref.simulate_retry_storm(REQUESTS, use_breaker=False, seed=0)
    on = ref.simulate_retry_storm(REQUESTS, use_breaker=True, seed=0)
    sweep = EX01.sweep(ref)
    storm_off, storm_on = (EX01.storm(ref, 1000, b, 20) for b in (False, True))
    symptoms = detect(off)
    return {
        "amp": (amplification(off), amplification(on)), "detected": symptoms,
        "diagnosis": diagnose(ref, symptoms),
        "symptom_ok": by_symptom(on), "objective_ok": by_objective(off, on),
        "rates": (off[1] / REQUESTS, on[1] / REQUESTS),
        "passing": sorted(t for t, run in sweep.items() if by_symptom(run)),
        "best": min(sweep, key=lambda t: amplification(sweep[t])), "sweep": sweep,
        "storm": (storm_off[1], storm_on[1]),
        "storm_ok": (by_symptom(storm_on, 1000),
                     by_objective(storm_off, storm_on)),
        "lesson_validation": "Validation agent" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    best = result["sweep"][result["best"]]
    return [
        practice.Check(
            "ANSWER: detect amplification, diagnose the cascade, validate the goal",
            all([result["detected"] == {"retry_amplification": True},
                 result["diagnosis"] == ("unknown", ["cascade"], "circuit breaker"),
                 result["symptom_ok"], not result["objective_ok"]]),
            f"detection fires at {result['amp'][0]} calls/request; diagnosis "
            f"{result['diagnosis']}; the symptom validator passes the breaker at "
            f"{result['amp'][1]} while the no-regression validator rejects it, success "
            f"{result['rates'][0]:.3f} -> {result['rates'][1]:.3f}",
        ),
        practice.Check(
            "FINDING: the symptom validator's favourite fix is turning traffic off",
            result["best"] == 0.05 and best == (26, 25)
            and result["passing"] == [0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
            f"lowest amplification at threshold {result['best']}: {best[0]} calls serving "
            f"{best[1]} of {REQUESTS}; passing thresholds {result['passing']}",
        ),
        practice.Check(
            "FINDING: whether the breaker is a fix depends on the load model",
            result["storm"][1] > 2 * result["storm"][0] and result["storm_ok"] == (True, True),
            f"with load over the last 20 requests successes go {result['storm'][0]} -> "
            f"{result['storm'][1]} and the objective validator accepts; on the shipped "
            "load it rejects -- the symptom validator accepts both",
        ),
        practice.Check(
            "FINDING: STRATUS's abstract names mitigation, not validation",
            result["lesson_validation"],
            "the lesson lists a 'Validation agent'; arXiv:2506.02009 names detection, "
            "diagnosis and mitigation agents with transactional no-regression, measured on "
            "cloud SRE benchmarks (AIOpsLab, ITBench)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
