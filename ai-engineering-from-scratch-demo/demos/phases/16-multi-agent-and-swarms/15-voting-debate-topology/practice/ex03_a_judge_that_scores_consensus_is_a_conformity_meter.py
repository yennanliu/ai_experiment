"""Exercise 3 — a judge that scores consensus is a conformity meter.

    Add a "judge" role to the graph topology that does not vote, only scores
    the final consensus. Does this change the emergent conformity behavior?

Reading of the exercise: "scores the final consensus" is read as the only
score available without ground truth -- the fraction of agents agreeing with
the final answer -- and the judge is tried twice: as a pure observer, and as
a gate that sends the debate round again until the score clears 0.8, which
is how a judge's score gets used in practice.

**ANSWER: as an observer, no -- by construction; as a gate, yes, and only in
the direction of more conformity.** A judge that runs after the last round
cannot change anything before it: over 200 seeds at N=5 the final answers
and the dissenter switches are identical with and without it. Gate on its
score and the debate runs until the minority has capitulated: switches rise
from 94 to 139 across the 200 debates, and accuracy stays at 0.905 exactly,
because conformity moves votes to the argmax and never moves the argmax.

**FINDING: the judge's score rises with rounds while accuracy is flat.**
Mean agreement is 0.756 after 1 round, 0.850 after 2, 0.916 after 3 and
0.945 after 4; accuracy is 0.905 at all four. The score measures how long
the debate ran, not whether it was right.

**FINDING: the judge scores wrong consensus almost as high as right.** In
the monoculture every wrong agent says WRONG-A, so the 19 wrong finals score
0.821 on average against 0.853 for the 181 right ones -- a gap of 0.032, so
no threshold on this score separates wrong consensus from right.

Structure: `debate()` is the reference `run_graph` with the positions kept
and an optional gate; it is checked against the reference's final answer and
token count on every seed before anything is measured.
"""

from __future__ import annotations

import itertools
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "15-voting-debate-topology"
N, TRIALS, GATE, CAP = 5, 200, 0.8, 10


def agreement(ref, positions):
    return sum(p == ref.majority(positions) for p in positions) / len(positions)


def one_round(ref, positions, rng):
    leader = ref.majority(positions)
    moved = [leader if p != leader and rng.random() < 0.4 else p for p in positions]
    return moved, sum(a != b for a, b in zip(moved, positions))


def more(ref, positions, done, rounds, gate):
    """Another round: the fixed count is not reached, or the judge's gate is not met."""
    return done < rounds or bool(gate) and agreement(ref, positions) < gate and done < CAP


def debate(ref, agents, rng, rounds=2, gate=None):
    """run_graph, returning (final, tokens, switches, judge score)."""
    positions = [a.answer("RIGHT", rng) for a in agents]
    per_round = sum(a.tokens_per_call for a in agents)
    tokens, switches, done = per_round, 0, 1
    while more(ref, positions, done, rounds, gate):
        positions, moved = one_round(ref, positions, rng)
        tokens, switches, done = tokens + per_round, switches + moved, done + 1
    return ref.majority(positions), tokens, switches, agreement(ref, positions)


def sweep(ref, **kwargs):
    runs = [debate(ref, ref.make_agents(N, False, t), random.Random(t * 31 + 7), **kwargs)
            for t in range(TRIALS)]
    return {"acc": sum(r[0] == "RIGHT" for r in runs) / TRIALS,
            "switches": sum(r[2] for r in runs),
            "score": round(sum(r[3] for r in runs) / TRIALS, 3), "runs": runs}


def matches_reference(ref):
    for n, t in itertools.product((3, 5, 7), range(TRIALS)):
        mine = debate(ref, ref.make_agents(n, False, t), random.Random(t * 31 + 7))
        theirs = ref.run_graph(ref.make_agents(n, False, t), "RIGHT", random.Random(t * 31 + 7))
        if mine[:2] != (theirs.final_answer, theirs.tokens):
            return False
    return True


def mean_score(runs, right):
    scores = [r[3] for r in runs if (r[0] == "RIGHT") == right]
    return len(scores), round(sum(scores) / len(scores), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base, gated = sweep(ref), sweep(ref, gate=GATE)
    by_rounds = {r: sweep(ref, rounds=r) for r in (1, 2, 3, 4)}
    return {
        "parity": matches_reference(ref), "base": base, "gated": gated,
        "same_finals": sum(a[0] == b[0] for a, b in zip(base["runs"], gated["runs"])),
        "scores": {r: s["score"] for r, s in by_rounds.items()},
        "accs": {r: s["acc"] for r, s in by_rounds.items()},
        "right": mean_score(base["runs"], True), "wrong": mean_score(base["runs"], False),
    }


def verify(result):
    base, gated = result["base"], result["gated"]
    return [
        practice.Check(
            "ANSWER: as an observer no, by construction; as a gate, more conformity",
            all([result["parity"], gated["switches"] > 1.4 * base["switches"],
                 result["same_finals"] == TRIALS, gated["acc"] == base["acc"]]),
            f"the copy matches run_graph's answer and tokens on all seeds; gating on "
            f"agreement >= {GATE} lifts switches from {base['switches']} to "
            f"{gated['switches']} while {result['same_finals']}/{TRIALS} finals and "
            f"accuracy {gated['acc']} are unchanged",
        ),
        practice.Check(
            "FINDING: the judge's score rises with rounds while accuracy is flat",
            all([list(result["scores"].values()) == sorted(result["scores"].values()),
                 len(set(result["accs"].values())) == 1]),
            f"mean agreement by rounds {result['scores']}; accuracy {result['accs']}",
        ),
        practice.Check(
            "FINDING: the judge scores wrong consensus almost as high as right",
            result["right"][1] - result["wrong"][1] < 0.05,
            f"{result['wrong'][0]} wrong finals score {result['wrong'][1]} on average "
            f"against {result['right'][1]} for {result['right'][0]} right ones",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
