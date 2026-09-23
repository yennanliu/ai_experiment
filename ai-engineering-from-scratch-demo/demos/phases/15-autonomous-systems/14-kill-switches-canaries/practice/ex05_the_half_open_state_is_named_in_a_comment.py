"""Exercise 5 — the half-open state is named in a comment.

    Define a re-enable procedure for a kill-switched agent. Who can
    re-enable? What must be documented? What must change about the agent
    before re-enable?

Reading of the exercise: three questions about a procedure, and a procedure
is only real where the code has somewhere to record it. So each answer is
paired with the field or call site it would need, and the module is asked
whether that place exists.

**ANSWER: a second person, the trajectory that tripped it, and the
condition that fired.** Re-enable requires someone other than the operator
who ran the agent -- the switch exists because the run could not be trusted to
stop itself, and the same asymmetry applies to starting it. What must be
documented is the window that tripped the detector, not a summary of it. And
what must change is the condition: a breaker re-enabled against an unchanged
trajectory opens again on the same turn, which is the definition of a stop
that was not a decision.

**FINDING: the recovery state exists in a comment.** `CircuitBreaker.state`
is annotated `closed | open | half_open` and **1** of the three is ever
assigned -- `open`. There is no transition back, so "re-enable" for the
breaker means constructing a new object, which discards `recent` and
therefore the evidence.

**FINDING: the switch the agent cannot write is a module global.**
`EXTERNAL_KILL_SWITCH` is a dict at module scope with the comment "agent
cannot write", and **1** line of the agent's own process changes it -- `main`
does exactly that, twice. The externality is documentation.

**FINDING: nothing records who or why.** `CircuitBreaker` has **3** fields
and `Canary` **1**; **0** of the **4** hold an operator, a timestamp or a
reason. A re-enable procedure needs a durable record that outlives the
process, and Lesson 12's event log is the shape of it -- which is also why the
two lessons' artifacts do not compose: the tripwires here keep state in
objects the next process will not see.

Structure: `states()` reads which breaker transitions exist; `fields()`
counts what the detectors can record about a stop.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "14-kill-switches-canaries"

STATES = ("closed", "open", "half_open")
PROVENANCE = ("who", "operator", "reason", "timestamp", "cleared", "when")


def assigned_states(ref):
    source = inspect.getsource(ref.CircuitBreaker)
    return [name for name in STATES if f'self.state = "{name}"' in source]


def mentioned_states(ref):
    source = inspect.getsource(ref.CircuitBreaker)
    return [name for name in STATES if name in source]


def fields(ref):
    return (list(ref.CircuitBreaker.__dataclass_fields__),
            list(ref.Canary.__dataclass_fields__))


def reopens(ref):
    """A breaker re-enabled against the same trajectory: the turn it opens again."""
    burst = ["tool:read:logs/app.log"] * 6
    breaker = ref.CircuitBreaker(threshold=5)
    for turn, key in enumerate(burst, 1):
        if not breaker.record(key):
            return turn
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    breaker_fields, canary_fields = fields(ref)
    main_source = inspect.getsource(ref.main)
    return {
        "states_mentioned": mentioned_states(ref),
        "states_assigned": assigned_states(ref),
        "has_reset": any(name in dir(ref.CircuitBreaker)
                         for name in ("reset", "close", "half_open")),
        "reopens_at": reopens(ref),
        "switch_is_global": isinstance(ref.EXTERNAL_KILL_SWITCH, dict),
        "writes_in_module": main_source.count('EXTERNAL_KILL_SWITCH["enabled"]'),
        "claims_external": "agent cannot write" in inspect.getsource(ref).split(
            "EXTERNAL_KILL_SWITCH")[0][-200:],
        "breaker_fields": breaker_fields,
        "canary_fields": canary_fields,
        "total_fields": len(breaker_fields) + len(canary_fields),
        "provenance_fields": [name for name in breaker_fields + canary_fields
                              if any(word in name for word in PROVENANCE)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a second person, the trajectory, and a changed condition",
            all([result["reopens_at"] == 5, not result["has_reset"]]),
            f"a breaker re-enabled against the same trajectory opens again on turn "
            f"{result['reopens_at']}, so the condition is what has to change; and the "
            "class exposes no reset, so re-enable is not a transition the code has",
        ),
        practice.Check(
            "FINDING: the recovery state exists in a comment",
            all([result["states_mentioned"] == ["closed", "open", "half_open"],
                 result["states_assigned"] == ["open"]]),
            f"state is annotated {result['states_mentioned']} and "
            f"{len(result['states_assigned'])} of the three is ever assigned -- "
            f"{result['states_assigned'][0]} -- so re-enable means a new object, which "
            "discards the evidence",
        ),
        practice.Check(
            "FINDING: the switch the agent cannot write is a module global",
            all([result["switch_is_global"], result["writes_in_module"] == 2,
                 result["claims_external"]]),
            f"EXTERNAL_KILL_SWITCH is a module-scope dict under a comment saying the "
            f"agent cannot write it, and main changes it {result['writes_in_module']} "
            "times from inside the same process",
        ),
        practice.Check(
            "FINDING: nothing records who or why",
            all([result["total_fields"] == 4, result["provenance_fields"] == []]),
            f"the two detectors hold {result['total_fields']} fields between them -- "
            f"{result['breaker_fields']} and {result['canary_fields']} -- and "
            f"{len(result['provenance_fields'])} record an operator, a time or a "
            "reason",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
