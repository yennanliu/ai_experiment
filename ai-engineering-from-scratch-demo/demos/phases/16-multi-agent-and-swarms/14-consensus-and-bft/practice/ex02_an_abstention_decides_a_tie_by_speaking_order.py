"""Exercise 2 — an abstention decides a tie by speaking order.

    Add a fourth attack pattern: **silent abstention** — one agent refuses to
    answer ("I don't know"). How should each aggregator treat abstentions?
    Implement your choice.

Reading of the exercise: each of the 5 agents in each of the 4 scenarios
abstains in turn, keeping its confidence -- 20 runs -- and the shipped
aggregators, which see "I don't know" as one more answer, are compared with a
policy that treats it as a non-vote.

**ANSWER: an abstention leaves the electorate; it is never a candidate, and
it counts against participation.** Every aggregator drops abstentions before
clustering and escalates when fewer than a majority of n (3 of 5) answered.
Plurality, which counts heads, additionally needs a winner holding a majority
of *all* n and escalates a tie instead of breaking it. CP-WBFT and DecentLLMs
then run unchanged on the answers that remain -- their weights are already
their tie-breakers. Over the 20 runs plurality goes from 8 wrong answers to 2
wrong and 6 escalated; CP-WBFT from 2 wrong and 3 rejected to 2 wrong and 18
right; DecentLLMs stays at 4 wrong, because a lone "I don't know" scores 1.0
and never wins there anyway. The 2 wrong answers left are the monoculture runs
where an honest agent abstains: 3 clones against 1 is a real majority.

**FINDING: the shipped plurality breaks ties by speaking order.** `max` over a
dict returns the first key inserted, so when an honest sycophancy-scenario
agent abstains, 42% and 4.2% tie 2-2 and 42% wins because agent-a spoke
first. 3 of the 20 runs flip plurality from right to wrong this way. The
policy escalates all three.

**FINDING: in CP-WBFT an abstention is a vote for rejection.** Its confidence
joins the denominator as a third cluster, which makes the dead 0.5 threshold
reachable: 3 monoculture runs go from 42% to no answer. That is the right
direction by accident -- the abstainer's confidence is confidence that it
does not know, and it is spent as if it were weight on an answer.

**FINDING: three abstainers are a consensus.** With 3 of 5 agents saying "I
don't know" in the no-attack scenario, all three shipped aggregators return
"I don't know" as the answer. And `canonical()` keeps the full stop, so "I
don't know." is a separate cluster from "I don't know": abstention cannot be
recognised by the clustering the aggregators already do.

Structure: `abstain()` builds the 20 runs; `decide()` is the policy, wrapping
the reference aggregators rather than rewriting them.
"""

from __future__ import annotations

import collections
import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "14-consensus-and-bft"
IDK, CORRECT = "I don't know", "4.2%"
ABSTAIN = {"idontknow", "unknown", "abstain", "noanswer"}


def scenarios(ref):
    seen, printer = {}, ref.scenario
    ref.scenario = lambda name, correct, votes: seen.setdefault(name, votes)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.scenario = printer
    return seen


def abstain(ref, votes, who):
    return [ref.Vote(v.agent, IDK if i in who else v.answer, v.confidence)
            for i, v in enumerate(votes)]


def is_abstention(vote):
    return vote.canonical().rstrip(".") in ABSTAIN


def decide(aggregate, votes, counted=False):
    """The policy: abstentions leave the electorate, and fewer than a majority of
    n answering escalates. A counting aggregator also needs a majority of all n
    for its winner and escalates ties; weighted ones then run as shipped."""
    answering = [v for v in votes if not is_abstention(v)]
    top = collections.Counter(v.canonical() for v in answering).most_common(2)
    if len(answering) <= len(votes) // 2:
        return None
    tied = len(top) == 2 and top[0][1] == top[1][1]
    if counted and (tied or top[0][1] <= len(votes) // 2):
        return None
    return aggregate(answering)[0]


def tally(answers):
    return {"wrong": sum(a not in (CORRECT, None) for a in answers),
            "escalated": answers.count(None), "right": answers.count(CORRECT)}


def run_all(aggs, grid):
    """(shipped answers, policy answers) per aggregator over the grid."""
    shipped = {k: [f(v)[0] for _, _, v in grid] for k, f in aggs.items()}
    policy = {k: [decide(f, v, k == "plurality") for _, _, v in grid] for k, f in aggs.items()}
    return shipped, policy


def tie_flips(ref, runs, grid, answers):
    """Runs where one abstention turns a right plurality into 42%."""
    was_right = {n for n, votes in runs.items() if ref.plurality(votes)[0] == CORRECT}
    flips = [(n, i, v) for (n, i, v), now in zip(grid, answers) if now == "42%" and n in was_right]
    return {"tie_flips": [(n, i) for n, i, _ in flips],
            "tie_policy": [decide(ref.plurality, v, True) for _, _, v in flips]}


def trio(ref, runs, aggs):
    """Three of five abstain in the no-attack scenario."""
    votes = abstain(ref, runs["no attack"], {0, 1, 2})
    return {"trio": [f(votes)[0] for f in aggs.values()],
            "trio_policy": [decide(f, votes, k == "plurality") for k, f in aggs.items()]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = scenarios(ref)
    aggs = {"plurality": ref.plurality, "cp_wbft": ref.cp_wbft, "decentllms": ref.decentllms}
    grid = [(name, i, abstain(ref, votes, {i})) for name, votes in runs.items() for i in range(5)]
    shipped, policy = run_all(aggs, grid)
    return {
        "shipped": {k: tally(a) for k, a in shipped.items()},
        "policy": {k: tally(a) for k, a in policy.items()},
        **tie_flips(ref, runs, grid, shipped["plurality"]), **trio(ref, runs, aggs),
        "cp_rejects": [(n, i) for (n, i, _), a in zip(grid, shipped["cp_wbft"]) if a is None],
        "stop": ref.Vote("x", IDK + ".", 0.5).canonical() != ref.Vote("x", IDK, 0.5).canonical(),
    }


def verify(result):
    shipped, policy = result["shipped"], result["policy"]
    return [
        practice.Check(
            "ANSWER: an abstention leaves the electorate and counts against participation",
            all([shipped["plurality"]["wrong"] == 8, policy["plurality"] == {
                "wrong": 2, "escalated": 6, "right": 12},
                shipped["cp_wbft"]["escalated"] == 3, policy["cp_wbft"]["right"] == 18,
                policy["decentllms"] == shipped["decentllms"]]),
            f"over 20 runs, wrong answers shipped {({k: v['wrong'] for k, v in shipped.items()})} "
            f"against the policy's {({k: v['wrong'] for k, v in policy.items()})}; the "
            f"policy escalates {({k: v['escalated'] for k, v in policy.items()})}",
        ),
        practice.Check(
            "FINDING: the shipped plurality breaks ties by speaking order",
            len(result["tie_flips"]) == 3 and result["tie_policy"] == [None] * 3,
            f"{result['tie_flips']} flip plurality from 4.2% to 42%, because max over a "
            "dict returns the first key inserted and agent-a spoke first; the policy "
            "escalates all three",
        ),
        practice.Check(
            "FINDING: in CP-WBFT an abstention is a vote for rejection",
            len(result["cp_rejects"]) == 3,
            f"the abstainer's confidence becomes a third cluster in the denominator and "
            f"the 0.5 threshold starts firing: {result['cp_rejects']} return no answer",
        ),
        practice.Check(
            "FINDING: three abstainers are a consensus",
            result["trio"] == [IDK] * 3 and result["trio_policy"] == [None] * 3
            and result["stop"],
            f"with 3 of 5 abstaining in the no-attack scenario the shipped aggregators "
            f"return {result['trio']}; canonical() keeps the full stop, so 'I don't know.' "
            "is a different cluster from 'I don't know'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
