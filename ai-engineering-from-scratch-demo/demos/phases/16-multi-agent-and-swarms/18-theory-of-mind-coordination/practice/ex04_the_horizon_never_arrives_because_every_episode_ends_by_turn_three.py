"""Exercise 4 — the horizon never arrives, because every episode ends by turn three.

    Read Li et al. (arXiv:2310.10701). Reproduce the "long-horizon degradation"
    finding: as turns grow from 10 to 30, how does your first-order ToM
    performance change?

Reading of the exercise: "turns" is `max_turns`, the only horizon knob the
module has; it is swept 10, 20, 30 for zeroth-order, primed first-order and
first-order with one hallucinated belief per turn (exercise 3), at 3x3 and
5x5, and compared trial by trial rather than by average.

**ANSWER: it does not change at all.** At 10, 20 and 30 turns every
condition gives the same completions, duplications and turns on every one of
its 400 trials. There is nothing to degrade: the longest episode in any
condition is 3 turns, so a budget of 10 is already never reached.

**The paper does not contain a 10-to-30 finding either.** Li et
al. run a bomb-defusal game -- 3 agents, 5 rooms, 5 bombs -- that ends when
all bombs are defused, a deadlock occurs, or 30 rounds pass; no experiment
varies the horizon. Their long-horizon failure is about *where* information
sits in the context: agents emit invalid actions although room connectivity
"is included in the initial prompts", because it is far away (§6.4.1). An
explicit belief state cut invalid actions 50.7% and raised first-order ToM
accuracy from 60.0% to 80.1% for GPT-4 (Table 2).

**FINDING: the mechanism is in the code; the task ends before it bites.**
The avoidance rule reads only the last (boxes remaining + 2) observations --
at most n + 2 -- and every turn adds `n - 1` more, so even at the widest
window the prime -- the only belief that matters -- falls
out of the window after 3 turns at 3x3 and 2 at 5x5. That is Li et al.'s
failure in miniature, a fixed context that pushes old facts out; it cannot
show here because the prime is spent on turn 0.

**FINDING: with more agents than boxes, a longer horizon only adds empty
turns.** 4 agents on 3 boxes can never all collect: 0 full completions and
1.85 duplications per trial at both 10 and 30 turns, with every trial using
the whole budget -- 10.0 turns, then 30.0.

Structure: `sweep()` runs ex01's `trial()` at each horizon; `evicted_after()`
counts the turns until the prime leaves the observation window.
"""

from __future__ import annotations

import pathlib

from harness import practice

HERE = pathlib.Path(__file__).resolve().parent
SIM = practice.load_module(next(HERE.glob("ex01_*.py")))
HORIZONS = (10, 20, 30)
CONDITIONS = {"zeroth": (False, {}), "first": (True, {}), "hallucinated": (True, {"flips": 1})}


def sweep(ref, n, tom, kw):
    """Per-horizon tuple of all trial results, so equality is trial by trial."""
    return {h: tuple(SIM.trial(ref, n, n, tom, s, max_turns=h, **kw) for s in range(SIM.TRIALS))
            for h in HORIZONS}


def evicted_after(ref, n):
    """Turns of (n - 1) new observations before no prime entry is in the window."""
    agents = [ref.Agent(f"agent-{i}", tom=True) for i in range(n)]
    SIM.default_prime(agents, n)
    agent, primed, window = agents[0], set(agents[0].observations), n + 2
    for turn in range(1, 10):
        agent.observations.extend(("other", -1) for _ in range(n - 1))
        if not primed & set(agent.observations[-window:]):
            return turn
    return None


def surplus(ref, horizon):
    runs = [SIM.trial(ref, 4, 3, True, s, max_turns=horizon) for s in range(SIM.TRIALS)]
    return (sum(c == 4 for c, _, _ in runs), round(sum(d for _, d, _ in runs) / SIM.TRIALS, 3),
            round(sum(t for _, _, t in runs) / SIM.TRIALS, 3))


def solve():
    ref = SIM.reference()
    flat, longest = True, 0
    for n in (3, 5):
        for tom, kw in CONDITIONS.values():
            runs = sweep(ref, n, tom, kw)
            flat &= runs[10] == runs[20] == runs[30]
            longest = max(longest, max(t for _, _, t in runs[30]))
    return {"flat": flat, "longest": longest,
            "evicted": {n: evicted_after(ref, n) for n in (3, 5)},
            "surplus": {h: surplus(ref, h) for h in (10, 30)}}


def verify(result):
    s = result["surplus"]
    return [
        practice.Check(
            "ANSWER: it does not change at all",
            result["flat"] and result["longest"] == 3,
            f"all {len(CONDITIONS)} conditions at 3x3 and 5x5 give identical results on "
            f"every trial at {HORIZONS} turns; the longest episode is "
            f"{result['longest']} turns",
        ),
        practice.Check(
            "FINDING: the mechanism is in the code; the task ends before it bites",
            result["evicted"] == {3: 3, 5: 2},
            f"the window is at most n + 2 observations and each turn adds n - 1, so the "
            f"prime is gone after {result['evicted'][3]} turns at 3x3 and "
            f"{result['evicted'][5]} at 5x5 -- but it is spent on turn 0",
        ),
        practice.Check(
            "FINDING: with more agents than boxes, a longer horizon only adds empty turns",
            s[10][:2] == s[30][:2] == (0, 1.85) and (s[10][2], s[30][2]) == (10.0, 30.0),
            f"4 agents on 3 boxes: {s[10]} at 10 turns and {s[30]} at 30 as (full "
            "completions, duplications/trial, turns)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
