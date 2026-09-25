"""Exercise 5 — graph wins by one voter, and loses the token column.

    Read MultiAgentBench (arXiv:2503.01935) Section 4 (topology experiments).
    Reproduce the "graph-wins-research" result on one task from the paper
    using your harness.

Reading of the exercise: MultiAgentBench's topology result (§4.3) is on its
research scenario -- agents co-author a 5-question proposal scored by an LLM
on a 5-point scale -- and the harness has one task, a string that is either
"RIGHT" or not. So the reproduction is of the result's *shape*: does graph
come out best on task score and on tokens, star close behind, tree worst?
Every topology is scored exactly by enumeration on the homogeneous agents.

**ANSWER: graph wins the accuracy column and loses the token column -- the
reverse of the paper on tokens, and the accuracy win is one extra voter.**
At N=5 graph scores 0.8624, star 0.8087, tree 0.72 and chain 0.72. But graph
votes all N agents while star's hub votes its N-1 workers, so graph at N
equals star at N+1 exactly: 0.8087, 0.8624, 0.8984 at N = 3, 5, 7 against
star at 4, 6, 8. And graph spends the most tokens -- 2x star at every N,
since it runs a second round -- where §4.3 reports graph best on "task
performance, planning efficiency, and token usage", with star and graph at
"similar task scores".

**FINDING: the tree is its left half.** `run_tree` takes the majority of
two sub-consensuses, and a 1-1 tie goes to the left, so the right half can
never change the answer: the final equals the left half's majority on 600 of
600 seeded runs. Tree at N=3 and N=5 is one agent (0.72) and at N=7 a
3-voter panel (0.8087), while paying for every leaf. The paper also puts
tree last -- here it is last by construction.

**FINDING: the chain is exactly one agent, at every length.** Each link
adopts a differing proposal with probability equal to its accuracy, *whether
or not the proposal is right*. Starting from P(right) = p, the next link
gives p(1 - (1-p)p) + (1-p)p^2 = p again, so the chain holds 0.72 at N = 3,
5 and 7 while its tokens grow linearly; the demo's 0.75, 0.78, 0.71 are
sampling noise around it.

Structure: `exact()` enumerates outcomes for any voting function built on
the reference `majority`; `chain_exact()` is the one-line recurrence; the
reference runners are sampled as `bench` samples them to confirm both.
"""

from __future__ import annotations

import itertools
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "15-voting-debate-topology"
TRIALS = 200


def exact(ref, agents, vote):
    total = 0.0
    for outcome in itertools.product((True, False), repeat=len(agents)):
        prob = 1.0
        for agent, right in zip(agents, outcome):
            prob *= agent.base_accuracy if right else 1 - agent.base_accuracy
        answers = ["RIGHT" if r else a.error_bias for a, r in zip(agents, outcome)]
        total += prob if vote(answers) == "RIGHT" else 0.0
    return round(total, 4)


def chain_exact(p, n):
    right = p
    for _ in range(n - 1):
        right = right * (1 - (1 - p) * p) + (1 - right) * p * p
    return round(right, 4)


def left_half(ref, answers):
    return ref.majority(answers[1:][: len(answers[1:]) // 2] or answers[1:])


def sampled(ref, runner, n):
    runs = [runner(ref.make_agents(n, False, t), "RIGHT", random.Random(t * 31 + 7))
            for t in range(TRIALS)]
    return sum(r.accuracy() for r in runs) / TRIALS, runs[0].tokens


def sample_all(ref, runner):
    return {n: sampled(ref, runner, n) for n in (3, 5, 7)}


def tree_is_left(ref):
    same = 0
    for n, t in itertools.product((3, 5, 7), range(TRIALS)):
        agents, rng = ref.make_agents(n, False, t), random.Random(t * 31 + 7)
        final = ref.run_tree(agents, "RIGHT", rng).final_answer
        replay = random.Random(t * 31 + 7)
        leaves = [a.answer("RIGHT", replay) for a in agents[1:]]  # the root never answers
        same += final == left_half(ref, [None, *leaves])
    return same


def exact_all(ref, sizes, voters, vote):
    """Exact accuracy at each N, over the agents `voters` selects."""
    return {n: exact(ref, voters(ref.make_agents(n, False, 0)), vote) for n in sizes}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sampled_runs = {name: sample_all(ref, getattr(ref, f"run_{name}"))
                    for name in ("star", "graph", "tree", "chain")}
    return {
        "graph": exact_all(ref, (3, 5, 7), list, ref.majority),
        "star": exact_all(ref, range(3, 9), lambda a: a[1:], ref.majority),
        "tree": exact_all(ref, (3, 5, 7), list, lambda a: left_half(ref, a)),
        "chain": {n: chain_exact(0.72, n) for n in (3, 5, 7)},
        "sampled": sampled_runs, "tree_left": tree_is_left(ref),
        "tokens": {name: [runs[n][1] for n in runs] for name, runs in sampled_runs.items()},
    }


def verify(result):
    graph, star, tree, chain, smp = (result[k] for k in ("graph", "star", "tree", "chain", "sampled"))
    tokens = result["tokens"]
    return [
        practice.Check(
            "ANSWER: graph wins accuracy by one voter and loses the token column",
            all([graph[5] > star[5] > tree[5] == chain[5],
                 all(graph[n] == star[n + 1] for n in (3, 5, 7)),
                 tokens["graph"] == [2 * t for t in tokens["star"]]]),
            f"at N=5 graph {graph[5]}, star {star[5]}, tree {tree[5]}, chain {chain[5]}; "
            f"graph at N=3,5,7 {list(graph.values())} equals star at N=4,6,8 "
            f"{[star[n] for n in (4, 6, 8)]}; tokens graph {tokens['graph']} against star "
            f"{tokens['star']}",
        ),
        practice.Check(
            "FINDING: the tree is its left half",
            all([result["tree_left"] == 3 * TRIALS, tree == {3: 0.72, 5: 0.72, 7: 0.8087},
                 tokens["tree"] == tokens["star"]]),
            f"the final equals the left half's majority {result['tree_left']}/{3 * TRIALS} "
            f"times; exact accuracy {tree} while paying star's tokens {tokens['tree']}",
        ),
        practice.Check(
            "FINDING: the chain is exactly one agent, at every length",
            all([set(chain.values()) == {0.72},
                 all(abs(smp["chain"][n][0] - 0.72) < 0.08 for n in (3, 5, 7))]),
            f"the recurrence holds P(right) at {chain}; the reference samples "
            f"{[smp['chain'][n][0] for n in (3, 5, 7)]} at tokens {tokens['chain']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
