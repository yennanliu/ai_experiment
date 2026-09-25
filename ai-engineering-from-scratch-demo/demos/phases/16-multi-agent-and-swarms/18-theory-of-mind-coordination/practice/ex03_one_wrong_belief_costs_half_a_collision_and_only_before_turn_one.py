"""Exercise 3 — one wrong belief costs half a collision, and only before turn one.

    Inject a **hallucination** into the ToM state: randomly flip one belief per
    turn. How much does this degrade first-order performance?

Reading of the exercise: a "belief" is one (agent, box) entry in an agent's
observation list -- the whole ToM state -- and a flip rewrites its box to a
uniformly random one, once per turn, before anyone chooses. Zeroth-order
agents get the same flips as a control, since they hold the same list and
never read it.

**ANSWER: from 0.00 to 0.505 duplications per trial at 3x3, and to 0.47 at
5x5** -- half of the 0.965 zeroth-order gap at 3x3 and a quarter of the
1.97 gap at 5x5. Completion stays 200 of 200 and turns rise from 1.0 to
1.505 and 1.47. The zeroth-order control moves 0.965 to 0.975: the flips
only perturb its random stream, so the ToM loss is belief damage.

**FINDING: only the turn-0 flip does anything.** 101 of 101 duplications at
3x3 and 94 of 94 at 5x5 happen on turn 0. A flip lands on the prime, which is
the only belief that ever removes an option (exercise 1); every later flip
hits a list that the avoidance rule reads and cannot act on. "Per turn" is
one hallucination per episode here, because the episode is one turn long.

**FINDING: the worst single wrong belief is a certain collision.**
Enumerating every single-belief rewrite of the prime: 18 at 3x3, and the 6
that cost nothing are exactly the 6 that rewrite a box to itself -- every
real rewrite causes duplications. The worst gives a collision on 200 of 200
seeds, and the mean over all 18 is 0.491. At 5x5 it is 100 rewrites, the 20
harmless ones again exactly the no-ops, mean 0.49, worst again 200 of 200. An agent that thinks
agent 1 wants box 0 is left with box 1 and walks into agent 1.

Structure: ex01's `trial()` with `flips=1` is the random injection;
`rewrite()` builds the one-belief-wrong primes for the enumeration.
"""

from __future__ import annotations

import pathlib

from harness import practice

HERE = pathlib.Path(__file__).resolve().parent
SIM = practice.load_module(next(HERE.glob("ex01_*.py")))


def turn0_share(ref, n):
    """(duplications on turn 0, all duplications) under one flip per turn."""
    first = total = 0
    for seed in range(SIM.TRIALS):
        log = []
        total += SIM.trial(ref, n, n, True, seed, flips=1, log=log)[1]
        picks = [box for box in log[0] if box >= 0]
        first += len(picks) - len(set(picks))
    return first, total


def rewrite(agent_index, slot, box):
    def prime(agents, n_boxes):
        SIM.default_prime(agents, n_boxes)
        holder = agents[agent_index]
        holder.observations[slot] = (holder.observations[slot][0], box)
    return prime


def enumerate_flips(ref, n):
    outcomes = []
    for a in range(n):
        for slot in range(n - 1):
            for box in range(n):
                runs = [SIM.trial(ref, n, n, True, s, prime=rewrite(a, slot, box))[1]
                        for s in range(SIM.TRIALS)]
                outcomes.append(sum(runs) / SIM.TRIALS)
    return {"rewrites": len(outcomes), "harmless": sum(o == 0 for o in outcomes),
            "worst": max(outcomes), "mean": round(sum(outcomes) / len(outcomes), 3)}


def solve():
    ref = SIM.reference()
    sizes = (3, 5)
    return {
        "tom": {n: SIM.bench(ref, n, True, flips=1) for n in sizes},
        "clean": {n: SIM.bench(ref, n, True) for n in sizes},
        "zeroth": {n: SIM.bench(ref, n, False) for n in sizes},
        "zeroth_flipped": {n: SIM.bench(ref, n, False, flips=1) for n in sizes},
        "turn0": {n: turn0_share(ref, n) for n in sizes},
        "enum": {n: enumerate_flips(ref, n) for n in sizes},
    }


def verify(result):
    tom, zeroth, flipped = result["tom"], result["zeroth"], result["zeroth_flipped"]
    enum = result["enum"]
    return [
        practice.Check(
            "ANSWER: 0.00 -> 0.505 duplications per trial at 3x3, 0.47 at 5x5",
            all([tom[3] == (200, 0.505, 1.505), tom[5] == (200, 0.47, 1.47),
                 result["clean"][3][1] == 0.0, abs(flipped[3][1] - zeroth[3][1]) < 0.05]),
            f"one flip per turn: ToM {tom[3]} and {tom[5]} against clean "
            f"{result['clean'][3]}; the zeroth-order control moves {zeroth[3][1]} -> "
            f"{flipped[3][1]}, so the loss is belief damage, not a changed random stream",
        ),
        practice.Check(
            "FINDING: only the turn-0 flip does anything",
            all(first == total > 0 for first, total in result["turn0"].values()),
            f"duplications on turn 0 / all: {result['turn0'][3]} at 3x3 and "
            f"{result['turn0'][5]} at 5x5 -- the prime is the only belief that removes an option",
        ),
        practice.Check(
            "FINDING: the worst single wrong belief is a certain collision",
            all([enum[3] == {"rewrites": 18, "harmless": 6, "worst": 1.0, "mean": 0.491},
                 enum[5]["rewrites"] == 100, enum[5]["harmless"] == 20,
                 enum[5]["worst"] == 1.0]),
            f"every one-belief rewrite of the prime: 3x3 {enum[3]}, 5x5 {enum[5]} -- "
            "worst 1.0 means a collision on every seed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
