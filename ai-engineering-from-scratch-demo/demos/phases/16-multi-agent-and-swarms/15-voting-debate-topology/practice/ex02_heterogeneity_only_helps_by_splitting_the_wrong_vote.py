"""Exercise 2 — heterogeneity only helps by splitting the wrong vote.

    Implement A-HMAD: three agents with deliberately different biases. How
    does the all-same-bias baseline compare to A-HMAD on the monoculture
    attack from Lesson 14?

Reading of the exercise: Lesson 14's monoculture attack is a fixed vote
profile -- three shared-model agents say "42%" at confidence 0.70/0.68/0.72
and two honest agents say "4.2%" at 0.85/0.82 -- so A-HMAD is the same
profile with the three shared-model agents replaced by three agents whose
wrong answers differ ("42%", "0.42%", "24%"), run through Lesson 14's own
three aggregators. The stochastic version then uses this lesson's SimAgent.

**ANSWER: A-HMAD fixes plurality and DecentLLMs and breaks CP-WBFT.** On
the monoculture profile all three aggregators return "42%". With the wrong
answers split, plurality returns "4.2%" (2 votes against 1, 1, 1) and so
does DecentLLMs (1.94 against three singletons at 1.0) -- but CP-WBFT
returns *nothing*: "4.2%" holds 1.67 of 3.77 confidence, 44% against its 50%
threshold. Heterogeneity turns a confident wrong answer into a refusal.

**FINDING: heterogeneity helps only by splitting the wrong vote.** With the
same 0.72 accuracy, three same-bias agents are right 0.8087 of the time and
three different-bias agents 0.8652. The whole gain, 0.0565, is
p(1-p)^2 = 0.0564 to rounding -- the outcome where agent 0 is right and the other two are wrong
in different ways, a 1-1-1 split that `majority` hands to whoever spoke
first. So at N=3 A-HMAD is the first speaker winning a three-way tie:
accuracies [0.9, 0.6, 0.6] score 0.936 in that order and 0.816 reversed,
while the same-bias baseline scores 0.792 either way.

**FINDING: the demo's heterogeneity flag turns two knobs, and its takeaway
is false.** `make_agents(heterogeneous=True)` changes the biases *and* the
accuracies ([0.72, 0.70, 0.74, ...] against a flat 0.72). And "heterogeneous
ensembles outperform homogeneous at every topology/N" fails in 5 of the 12
cells of its own table -- chain at N=5 and 7, tree at N=3 and 5, and star
at N=3.

Structure: `profiles()` builds the two Lesson 14 vote lists and
`aggregate()` runs Lesson 14's aggregators on them; `exact()` enumerates
SimAgent outcomes and votes with this lesson's `majority`.
"""

from __future__ import annotations

import inspect
import itertools
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "15-voting-debate-topology"
L14 = "14-consensus-and-bft"
SHARED = (("agent-a", 0.70), ("agent-b", 0.68), ("agent-c", 0.72))
HONEST = (("agent-d", "4.2%", 0.85), ("agent-e", "4.2%", 0.82))


def profiles(bft):
    honest = [bft.Vote(*v) for v in HONEST]
    mono = [bft.Vote(name, "42%", conf) for name, conf in SHARED] + honest
    split = [bft.Vote(name, wrong, conf)
             for (name, conf), wrong in zip(SHARED, ("42%", "0.42%", "24%"))] + honest
    return mono, split


def aggregate(bft, votes):
    return [bft.plurality(votes)[0], bft.cp_wbft(votes)[0], bft.decentllms(votes)[0]]


def exact(ref, accuracies, biases):
    total = 0.0
    for outcome in itertools.product((True, False), repeat=len(accuracies)):
        prob = 1.0
        for acc, right in zip(accuracies, outcome):
            prob *= acc if right else 1 - acc
        answers = ["RIGHT" if r else b for r, b in zip(outcome, biases)]
        total += prob if ref.majority(answers) == "RIGHT" else 0.0
    return round(total, 4)


def bench_cells(ref):
    """(topology, n) -> accuracy, exactly as the demo's bench computes it."""
    runners = {"star": ref.run_star, "chain": ref.run_chain, "tree": ref.run_tree,
               "graph": ref.run_graph}
    return {(het, top, n): sum(run(ref.make_agents(n, het, t), "RIGHT",
                                   random.Random(t * 31 + 7)).accuracy() for t in range(200))
            for het in (False, True) for top, run in runners.items() for n in (3, 5, 7)}


def solve():
    ref, bft = (parity.load_reference(PHASE, lesson, "main") for lesson in (LESSON, L14))
    mono, split = profiles(bft)
    same, diff = ["WRONG-A"] * 3, ["WRONG-A", "WRONG-B", "WRONG-C"]
    cells = bench_cells(ref)
    return {
        "profile_matches": all(f'"42%", {c:.2f}' in inspect.getsource(bft.main) for _, c in SHARED),
        "mono": aggregate(bft, mono), "split": aggregate(bft, split),
        "share": round(bft.cp_wbft(split)[1]["4.2%"] / sum(bft.cp_wbft(split)[1].values()), 3),
        "same": exact(ref, [0.72] * 3, same), "diff": exact(ref, [0.72] * 3, diff),
        "ordered": [exact(ref, acc, diff) for acc in ([0.9, 0.6, 0.6], [0.6, 0.6, 0.9])],
        "ordered_same": [exact(ref, acc, same) for acc in ([0.9, 0.6, 0.6], [0.6, 0.6, 0.9])],
        "het_acc": [a.base_accuracy for a in ref.make_agents(3, True, 0)],
        "worse": sorted((top, n) for (het, top, n), acc in cells.items()
                        if het and acc < cells[(False, top, n)]),
    }


def verify(result):
    gain = round(result["diff"] - result["same"], 4)
    return [
        practice.Check(
            "ANSWER: A-HMAD fixes plurality and DecentLLMs and breaks CP-WBFT",
            all([result["profile_matches"], result["mono"] == ["42%"] * 3,
                 result["split"] == ["4.2%", None, "4.2%"], result["share"] < 0.5]),
            f"monoculture -> {result['mono']}; split wrong answers -> {result['split']} "
            f"(plurality, CP-WBFT, DecentLLMs); the right answer holds {result['share']} "
            "of CP-WBFT's confidence, under its 0.5 threshold",
        ),
        practice.Check(
            "FINDING: heterogeneity helps only by splitting the wrong vote",
            all([abs(gain - 0.72 * 0.28 ** 2) < 1e-3, result["ordered"] == [0.936, 0.816],
                 result["ordered_same"] == [0.792, 0.792]]),
            f"same-bias {result['same']} against different-bias {result['diff']}: the "
            f"gain {gain} is p(1-p)^2 = {0.72 * 0.28 ** 2:.4f}, the 1-1-1 split handed to agent 0; "
            f"[0.9,0.6,0.6] scores {result['ordered'][0]} and reversed "
            f"{result['ordered'][1]}, same-bias {result['ordered_same']}",
        ),
        practice.Check(
            "FINDING: the heterogeneity flag turns two knobs, and the takeaway is false",
            all([result["het_acc"] == [0.72, 0.70, 0.74], len(result["worse"]) == 5]),
            f"heterogeneous accuracies are {result['het_acc']}, not a flat 0.72; "
            f"heterogeneous scores below homogeneous in {len(result['worse'])} of 12 "
            f"cells: {result['worse']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
