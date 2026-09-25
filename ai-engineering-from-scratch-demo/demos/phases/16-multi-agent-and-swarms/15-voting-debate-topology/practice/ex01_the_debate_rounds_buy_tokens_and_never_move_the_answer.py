"""Exercise 1 — the debate rounds buy tokens and never move the answer.

    Run `code/main.py`. Plot the coordination-tax curve for graph topology:
    accuracy vs N, tokens vs N. At what N does the curve inflect?

Reading of the exercise: the curve is computed exactly rather than from the
demo's 200 seeded trials -- each agent is right with its base accuracy or
gives its bias, so the 2^N outcomes can be enumerated -- and the reference
run_graph is then checked against it, so the "plot" is a table that cannot
carry sampling noise into the answer.

**ANSWER: it never inflects -- accuracy is concave from the first agent and
tokens are linear, so every agent buys less than the one before.**
Homogeneous graph accuracy for N = 1..9 is 0.72, 0.72, 0.8087, 0.8087,
0.8624, 0.8624, 0.8984, 0.8984, 0.9238, at 800 tokens per agent. The odd-N
gains shrink from +0.0887 to +0.0254 with no knee at 4 or anywhere else, and
the ~4-agent coordination tax the lesson describes cannot appear:
`tokens_per_call` is a constant 400 whatever an agent reads, so seeing more
peers costs nothing.

**FINDING: the debate rounds never change the answer.** `run_graph` moves a
dissenter to the current majority with probability 0.4, and moving a vote to
the argmax cannot change the argmax. Over 3 sizes x 200 seeds, rounds = 1, 2,
3 and 4 give the same final answer 600 of 600 times: graph is one plurality
vote of all N agents, and round 2 doubles its tokens for nothing.

**FINDING: even N buys zero accuracy.** A 2-2 tie goes to whichever answer
`majority` saw first, which is agent 0's -- so N=4 is exactly N=3, 6 is
exactly 5, and 8 is exactly 7: four of the nine points on the curve are
paid-for duplicates.

**FINDING: the promised latency column does not exist.** The lesson promises
"(accuracy, tokens, latency)" and a `wallclock_simulated`; the table prints
`steps`, and graph's is `rounds * 2` = 4 at every N. The takeaway "tokens
inflate ~7x over star/N=3" is 5600 / 1200 = 4.7x.

Structure: `exact()` enumerates outcomes and votes with the reference's own
`majority`; `sampled()` runs the reference `run_graph` exactly as `bench`
does, and must land within sampling error of it.
"""

from __future__ import annotations

import itertools
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "15-voting-debate-topology"
TRIALS = 200


def exact(ref, agents):
    """P(the plurality of all agents' answers is RIGHT), by enumeration."""
    total = 0.0
    for outcome in itertools.product((True, False), repeat=len(agents)):
        prob, answers = 1.0, []
        for agent, right in zip(agents, outcome):
            prob *= agent.base_accuracy if right else 1 - agent.base_accuracy
            answers.append("RIGHT" if right else agent.error_bias)
        total += prob if ref.majority(answers) == "RIGHT" else 0.0
    return round(total, 4)


def sampled(ref, n, rounds=2):
    """The demo's bench for graph: 200 seeded trials of the reference run_graph."""
    runs = [ref.run_graph(ref.make_agents(n, False, t), "RIGHT", random.Random(t * 31 + 7), rounds)
            for t in range(TRIALS)]
    return sum(r.accuracy() for r in runs) / TRIALS, runs[0].tokens, runs[0].steps


def rounds_invariant(ref):
    same = 0
    for n, t in itertools.product((3, 5, 7), range(TRIALS)):
        finals = {ref.run_graph(ref.make_agents(n, False, t), "RIGHT",
                                random.Random(t * 31 + 7), rounds).final_answer
                  for rounds in (1, 2, 3, 4)}
        same += len(finals) == 1
    return same


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    curve = {n: exact(ref, ref.make_agents(n, False, 0)) for n in range(1, 10)}
    bench = {n: sampled(ref, n) for n in (3, 5, 7)}
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "curve": curve, "tokens": {n: 800 * n for n in curve}, "bench": bench,
        "gains": [round(curve[n] - curve[n - 2], 4) for n in (3, 5, 7, 9)],
        "per_call": {a.tokens_per_call for a in ref.make_agents(7, True, 0)},
        "invariant": rounds_invariant(ref),
        "latency_promised": "latency" in doc and "wallclock_simulated" in doc,
        "star3": ref.run_star(ref.make_agents(3, False, 0), "RIGHT", random.Random(0)).tokens,
    }


def verify(result):
    curve, bench, gains = result["curve"], result["bench"], result["gains"]
    return [
        practice.Check(
            "ANSWER: it never inflects -- concave from the first agent, linear tokens",
            all([gains == sorted(gains, reverse=True), result["per_call"] == {400},
                 all(abs(bench[n][0] - curve[n]) < 0.08 for n in bench),
                 all(bench[n][1] == result["tokens"][n] for n in bench)]),
            f"accuracy {list(curve.values())} at 800 tokens per agent; odd-N gains "
            f"{gains} shrink monotonically; the reference bench gives "
            f"{[bench[n][0] for n in bench]} at N=3,5,7; tokens_per_call is "
            f"{result['per_call']} whatever an agent reads",
        ),
        practice.Check(
            "FINDING: the debate rounds never change the answer",
            result["invariant"] == 3 * TRIALS,
            f"rounds 1-4 give the same final answer {result['invariant']} of "
            f"{3 * TRIALS} times: moving a dissenter to the argmax cannot move the argmax",
        ),
        practice.Check(
            "FINDING: even N buys zero accuracy",
            all(curve[n] == curve[n - 1] for n in (2, 4, 6, 8)),
            f"N=4 {curve[4]} = N=3 {curve[3]}, N=6 {curve[6]} = N=5 {curve[5]}, N=8 "
            f"{curve[8]} = N=7 {curve[7]}: a tie goes to agent 0's answer",
        ),
        practice.Check(
            "FINDING: the promised latency column does not exist",
            all([result["latency_promised"], {bench[n][2] for n in bench} == {4},
                 round(result["tokens"][7] / result["star3"], 1) == 4.7]),
            f"steps is {bench[3][2]} at every N, and graph/N=7 over star/N=3 is "
            f"{result['tokens'][7]}/{result['star3']} = "
            f"{result['tokens'][7] / result['star3']:.1f}x, not ~7x",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
