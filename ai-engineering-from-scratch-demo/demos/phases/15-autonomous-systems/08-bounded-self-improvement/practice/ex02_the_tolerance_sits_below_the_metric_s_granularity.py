"""Exercise 2 — the tolerance sits below the metric's granularity.

    Disable regression detection. Construct an input where this leads to
    silent capability loss being accepted.

Reading of the exercise: "construct an input" suggests the loss needs a
special case to provoke. It does not -- it happens at the lesson's own seed,
in most accepted cycles -- so the construction here is the *measurement* that
makes it visible, plus the smallest input that shows the gate's threshold is
decorative.

**ANSWER: 80 of 116 accepted cycles sit below the running best.** With the
regression gate off, the loop accepts **116** of 200 cycles and **80** of
them have a perf strictly lower than the best already seen. With it on the
same stream accepts **96** and **0** are below the best. Final perf is
**1.00** either way, which is exactly why the loss is silent: the headline
number recovers and the dip is never printed.

**FINDING: the tolerance is below the metric's granularity.** `perf_score`
divides by **4** cases, so it takes values in steps of **0.25** and the
smallest possible regression is **0.25**. `gate_regression` defaults to
`tol=0.2`. Every regression the metric can express is larger than the slack,
so the gate is strictly monotonic and its docstring's "accept noise" describes
a behaviour it cannot have.

**FINDING: the smallest input that shows it is one case.** Against a history
whose best is **1.00**, a candidate at **0.75** -- one of four cases lost --
is rejected at `tol=0.2` and accepted at `tol=0.25`. The gate's stated slack
and its effective slack differ by one case, and the metric has only four.

**FINDING: the dips are absorbed, not recovered from.** Of the **80** below-
best acceptances, the deepest is **0.75** points below the running best and
the loop still finishes at the ceiling. So the failure this gate prevents is
not a worse final score -- it is that nothing in the run records that the
capability was ever lost, which is what makes "silent" the operative word.

Structure: `replay()` is `run`'s loop with the below-best acceptances
counted; `granularity()` reads the metric's step size off the case list.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "08-bounded-self-improvement"

SEED, CYCLES = 5, 200
ALL_ON = dict(invariant=True, anchor=True, multi=True, regress=True)
SHIPPED_TOL = 0.2


GATES = ("invariant", "anchor", "multi", "regress")


def failing_gate(ref, gates, candidate, history):
    """The first gate a candidate fails, in the order `run` applies them."""
    perf, safety = ref.perf_score(candidate), ref.safety_score(candidate)
    checks = (("invariant", ref.gate_invariant(candidate)),
              ("anchor", ref.gate_anchor(candidate)),
              ("multi", ref.gate_multi(perf, safety)),
              ("regress", ref.gate_regression(history, perf)))
    for name, passed in checks:
        if gates[name] and not passed:
            return name
    return None


def replay(ref, gates, edit=True):
    """`run`'s loop, returning the state it prints and the state it does not."""
    random.seed(SEED)
    agent, accepted = ref.Agent(), []
    history, rejects = [ref.perf_score(agent)], dict.fromkeys(GATES, 0)
    scores = []
    for _cycle in range(CYCLES):
        candidate = ref.mutate(agent, edit)
        gate = failing_gate(ref, gates, candidate, history)
        scores.append((gate, ref.perf_score(candidate), ref.safety_score(candidate)))
        if gate:
            rejects[gate] += 1
            continue
        agent = candidate
        accepted.append(ref.perf_score(agent))
        history.append(accepted[-1])
    return {"accepted": len(accepted), "rejects": rejects, "agent": agent,
            "history": history, "scores": scores, "perfs": accepted}


def multi_reasons(scores):
    """Why each multi rejection fired, by axis."""
    counts = dict.fromkeys(("perf", "safety", "both"), 0)
    for gate, perf, safety in scores:
        if gate == "multi":
            counts["both" if perf < 0.25 and safety < 1.0
                   else ("perf" if perf < 0.25 else "safety")] += 1
    return counts


def below_best(run):
    """Accepted cycles whose perf is under the best already seen, and the deepest drop."""
    history, below, deepest = run["history"], 0, 0.0
    for index, perf in enumerate(run["perfs"]):
        best = max(history[:index + 1])
        below += perf < best
        deepest = max(deepest, best - perf)
    return below, round(deepest, 2)


def granularity(ref):
    return round(1 / len(ref.CASES_PERF), 2)


def smallest_regression(ref, best=1.0, step=0.25):
    """The one-case loss, judged at the shipped tolerance and at the metric's step."""
    return (ref.gate_regression([best], best - step, SHIPPED_TOL),
            ref.gate_regression([best], best - step, step))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    guarded = replay(ref, ALL_ON)
    unguarded = replay(ref, dict(ALL_ON, regress=False))
    step = granularity(ref)
    return {
        "guarded": (guarded["accepted"], below_best(guarded)[0]),
        "unguarded": (unguarded["accepted"], below_best(unguarded)[0]),
        "guarded_final": ref.perf_score(guarded["agent"]),
        "unguarded_final": ref.perf_score(unguarded["agent"]),
        "deepest": below_best(unguarded)[1],
        "cases": len(ref.CASES_PERF),
        "step": step,
        "tolerance": SHIPPED_TOL,
        "tolerance_below_step": SHIPPED_TOL < step,
        "one_case": smallest_regression(ref),
        "claims_noise": "accept noise" in (ref.gate_regression.__doc__ or ""),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 80 of 116 accepted cycles sit below the running best",
            all([result["unguarded"] == (116, 80), result["guarded"] == (96, 0),
                 result["guarded_final"] == result["unguarded_final"] == 1.0]),
            f"with the gate off the loop accepts {result['unguarded'][0]} cycles of "
            f"which {result['unguarded'][1]} are below the best already seen; with it "
            f"on, {result['guarded'][1]} of {result['guarded'][0]} are -- and both runs "
            f"finish at perf {result['guarded_final']}, which is what makes it silent",
        ),
        practice.Check(
            "FINDING: the tolerance is below the metric's granularity",
            all([result["cases"] == 4, result["step"] == 0.25,
                 result["tolerance"] == 0.2, result["tolerance_below_step"],
                 result["claims_noise"]]),
            f"perf_score divides by {result['cases']} cases, so it moves in steps of "
            f"{result['step']} and the smallest possible regression exceeds the "
            f"{result['tolerance']} tolerance -- the gate is strictly monotonic and its "
            "docstring says it accepts noise",
        ),
        practice.Check(
            "FINDING: the smallest input that shows it is one case",
            result["one_case"] == (False, True),
            f"against a best of 1.00 a candidate at 0.75 -- one case of "
            f"{result['cases']} lost -- is rejected at tol={result['tolerance']} and "
            f"accepted at tol={result['step']}: the stated slack and the effective "
            "slack differ by one case",
        ),
        practice.Check(
            "FINDING: the dips are absorbed, not recovered from",
            all([result["deepest"] == 0.75, result["unguarded_final"] == 1.0]),
            f"the deepest below-best acceptance is {result['deepest']} points down and "
            f"the run still finishes at {result['unguarded_final']}, so what the gate "
            "prevents is not a worse final score but an unrecorded loss",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
