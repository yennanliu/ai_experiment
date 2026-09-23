"""Exercise 1 — the breaker fires on turn seven and the canary never fires.

    Run `code/main.py`. Confirm the circuit breaker fires on turn 5 (fifth
    identical call) and the canary fires on turn 9 (fake-key read).

Reading of the exercise: "confirm" is an instruction to check two claims, and
both are wrong -- for different reasons, which is what makes the pair worth
separating rather than just correcting.

**ANSWER: turn 7, and never.** The identical burst starts at action **3**, so
the fifth identical call is turn **7**, not 5. And `run_trajectory` breaks out
of the loop when the breaker opens, so action **9** is never reached:
`canary_hits` is **0** in both shipped scenarios.

**FINDING: the two claims fail for unrelated reasons.** "Turn 5" is an
off-by-two in counting the burst; "turn 9" is unreachable because the loop
terminates two actions earlier. Reordering the same nine actions to put the
canary read first makes it fire on turn **1** and the breaker on turn **8** --
so the detectors are not independent, they are sequential, and which one you
observe is a property of the trajectory order.

**FINDING: a canary hit is not a stop.** `check_read` records and returns
True, and the caller `continue`s -- the read is allowed and the trajectory
proceeds. Of the three detectors, **2** break the loop and **1** logs, which
is the right design and the opposite of what the shipped summary line
suggests by printing all three as peers.

**FINDING: the kill-switch scenario demonstrates nothing about the others.**
It is checked before every action, so it fires on turn **1** and **1** of the
**9** actions executes. Both shipped runs therefore end with
`canary_hits=0`, and the lesson's three-detector headline is supported by a
demonstration in which one detector never runs.

Structure: `trace()` captures a trajectory's stdout and parses the summary;
`shipped()` rebuilds the nine actions `main` uses.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "14-kill-switches-canaries"


def shipped(ref):
    """The nine actions `main` builds, read back out of its source."""
    body = inspect.getsource(ref.main)
    pairs = re.findall(r'Action\("(\w+)",\s*"([^"]+)"\)', body)
    return [ref.Action(kind, payload) for kind, payload in pairs]


def trace(ref, actions, enabled=False):
    """(turn the run stopped, why, canary hits) -- a canary hit is not a stop."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.run_trajectory(actions, {"enabled": enabled})
    text = buffer.getvalue()
    stop = re.search(r"\s*(\d+)\. \[(CIRCUIT BREAKER|KILL SWITCH)", text)
    hits = int(re.search(r"canary_hits=(\d+)", text).group(1))
    return [int(stop.group(1)), stop.group(2), hits] if stop else [None, "", hits]


def reordered(actions):
    """The same actions with the canary read moved to the front."""
    canary = [a for a in actions if a.payload.endswith(".env.canary")]
    return canary + [a for a in actions if a not in canary]


def burst_start(actions):
    keys = [f"{a.kind}:{a.payload}" for a in actions]
    for index in range(len(keys) - 4):
        if len(set(keys[index:index + 5])) == 1:
            return index + 1
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    actions = shipped(ref)
    body = ast.parse(inspect.getsource(ref.run_trajectory))
    breaker = inspect.getsource(ref.CircuitBreaker)
    return {
        "actions": len(actions),
        "burst_start": burst_start(actions),
        "threshold": ref.CircuitBreaker().threshold,
        "off": trace(ref, actions),
        "on": trace(ref, actions, enabled=True),
        "reordered": trace(ref, reordered(actions)),
        "breaks": sum(isinstance(node, ast.Break) for node in ast.walk(body)),
        "continues": sum(isinstance(node, ast.Continue) for node in ast.walk(body)),
        "canary_returns_true": "return True" in inspect.getsource(ref.Canary.check_read),
        "states_assigned": [name for name in ("closed", "open", "half_open")
                            if f'self.state = "{name}"' in breaker],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: turn 7, and never",
            all([result["off"] == [7, "CIRCUIT BREAKER", 0],
                 result["burst_start"] == 3, result["threshold"] == 5,
                 result["actions"] == 9]),
            f"the burst starts at action {result['burst_start']} and the threshold is "
            f"{result['threshold']}, so the fifth identical call is turn "
            f"{result['off'][0]}; the loop breaks there and action "
            f"{result['actions']} is never reached, leaving {result['off'][2]} canary "
            "hits",
        ),
        practice.Check(
            "FINDING: the two claims fail for unrelated reasons",
            all([result["reordered"] == [8, "CIRCUIT BREAKER", 1],
                 result["off"][2] == 0]),
            f"moving the canary read to the front makes it fire once and the breaker on "
            f"turn {result['reordered'][0]} -- the detectors are sequential, not "
            "independent, and which one you observe is a property of the order",
        ),
        practice.Check(
            "FINDING: a canary hit is not a stop",
            all([result["canary_returns_true"], result["breaks"] == 2,
                 result["continues"] == 1]),
            f"check_read records and returns True and the caller continues, so "
            f"{result['breaks']} of the three detectors break the loop and "
            f"{result['continues']} logs -- the right design, and not what the summary "
            "line suggests",
        ),
        practice.Check(
            "FINDING: the kill-switch scenario demonstrates nothing about the others",
            all([result["on"] == [1, "KILL SWITCH", 0],
                 result["states_assigned"] == ["open"]]),
            f"the switch is checked before every action, so it fires on turn "
            f"{result['on'][0]} and 1 of {result['actions']} actions executes; both "
            "shipped runs end with zero canary hits",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
