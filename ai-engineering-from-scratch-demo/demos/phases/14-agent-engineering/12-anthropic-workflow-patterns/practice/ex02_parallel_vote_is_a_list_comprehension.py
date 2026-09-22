"""Exercise 2 — parallel_vote is a list comprehension.

    Add a timeout to `parallel_vote`. What happens when one call hangs? How do
    you aggregate with missing votes?

Reading of the exercise: `parallel_vote` is `[llm(prompt) for _ in range(n)]`,
so the pattern named "parallelization" runs the calls one after another. That
decides the answer to "what happens when one hangs": everything after it
waits. Latency is supplied per call as a declared cost against a virtual
clock, so nothing here measures the host it runs on.

**ANSWER: a timeout plus a quorum, and the timeout changes the answer.** With
five voters, one of which hangs for **30** units against a timeout of **2**,
the shipped sequential loop finishes at **34** units and elects `no`; the
timed version finishes at **2** on **4** votes and elects `yes`. A quorum of
**3** returns a winner at **4** votes and `None` at **2**. Dropping a vote is
not free -- here it is decisive.

**FINDING: the hang costs the sum, not the maximum.** Sequential total
latency is **34** units where a concurrent one would be **30** and a
timed-out concurrent one **2**. The name of the pattern is the only parallel
thing about it, and adding a timeout without adding concurrency saves
**nothing** on the hang -- the shipped loop still has to reach the slow call.

**FINDING: a tie is broken by arrival order.** `Counter.most_common(1)[0]`
returns insertion order among equal counts, so four surviving votes split
**2-2** elect whichever answer came back first. With one voter timed out that
is decided by the network.

**FINDING: the return value cannot say how many voted.** `parallel_vote`
returns `(winner, Counter)` and the counter holds only the votes that
arrived, so **3**-of-**5** and **3**-of-**3** are the same object. A caller
enforcing a quorum has to be told `n` separately.

Structure: `timed_vote()` is the shipped aggregation with a deadline and a
quorum; `Clock` is virtual.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "12-anthropic-workflow-patterns"
PROMPT, TIMEOUT, QUORUM = "is this a refund request?", 2.0, 3
LATENCIES = (1.0, 1.0, 30.0, 1.0, 1.0)
ANSWERS = ("yes", "no", "no", "no", "yes")
SPLIT = ("yes", "no", "no", "yes")


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


def scripted(answers, latencies, clock):
    """One LLM call: costs its declared latency whether or not it is waited for."""
    state = {"index": 0}

    def call(_prompt):
        index = state["index"]
        state["index"] += 1
        clock.advance(latencies[index])
        return answers[index]

    return call


def timed_vote(answers, latencies, timeout=TIMEOUT, quorum=QUORUM):
    """Concurrent voting: every call starts at zero and is abandoned at the deadline."""
    votes = [answer for answer, latency in zip(answers, latencies) if latency <= timeout]
    counts = collections.Counter(votes)
    elapsed = min(timeout, max(latencies))
    if len(votes) < quorum:
        return None, counts, elapsed, len(votes)
    return counts.most_common(1)[0][0], counts, elapsed, len(votes)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clock = Clock()
    winner, counts = ref.parallel_vote(PROMPT, scripted(ANSWERS, LATENCIES, clock), n=5)
    timed = timed_vote(ANSWERS, LATENCIES)
    starved = timed_vote(ANSWERS, (1.0, 30.0, 30.0, 30.0, 1.0))
    tie = collections.Counter(SPLIT)
    return {
        "sequential_elapsed": clock.now, "sequential_winner": winner,
        "sequential_counts": dict(counts),
        "concurrent_untimed": max(LATENCIES),
        "timed_elapsed": timed[2], "timed_votes": timed[3], "timed_winner": timed[0],
        "starved_winner": starved[0], "starved_votes": starved[3], "quorum": QUORUM,
        "tie_counts": dict(tie), "tie_winner": tie.most_common(1)[0][0],
        "tie_first": SPLIT[0],
        "reported_total": sum(timed[1].values()), "voters": len(ANSWERS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a timeout plus a quorum, and the timeout changes the answer",
            all([result["sequential_elapsed"] == 34.0, result["timed_elapsed"] == 2.0,
                 result["timed_votes"] == 4, result["sequential_winner"] == "no",
                 result["timed_winner"] == "yes",
                 result["starved_winner"] is None, result["starved_votes"] == 2]),
            f"the shipped sequential loop finishes at {result['sequential_elapsed']} "
            f"units and elects {result['sequential_winner']!r}; the timed version "
            f"finishes at {result['timed_elapsed']} on {result['timed_votes']} votes and "
            f"elects {result['timed_winner']!r}. A quorum of {result['quorum']} returns "
            f"None when only {result['starved_votes']} arrive -- dropping a vote is not "
            "free",
        ),
        practice.Check(
            "FINDING: the hang costs the sum, not the maximum",
            all([result["sequential_elapsed"] == sum(LATENCIES),
                 result["concurrent_untimed"] == 30.0,
                 result["sequential_elapsed"] > result["concurrent_untimed"]]),
            f"sequential total latency is {result['sequential_elapsed']} where a "
            f"concurrent run would be {result['concurrent_untimed']} and a timed "
            f"concurrent one {result['timed_elapsed']}. parallel_vote is a list "
            "comprehension, so a timeout alone saves nothing -- the loop still arrives",
        ),
        practice.Check(
            "FINDING: a tie is broken by arrival order",
            all([result["tie_counts"] == {"yes": 2, "no": 2},
                 result["tie_winner"] == result["tie_first"]]),
            f"four surviving votes split {result['tie_counts']} and most_common returns "
            f"{result['tie_winner']!r} -- the answer that arrived first. With one voter "
            "timed out, which one that is has been decided by the network",
        ),
        practice.Check(
            "FINDING: the return value cannot say how many voted",
            all([result["reported_total"] == 4, result["voters"] == 5,
                 result["reported_total"] < result["voters"]]),
            f"parallel_vote returns (winner, Counter) and the counter holds the "
            f"{result['reported_total']} votes that arrived out of {result['voters']} "
            "requested. 3-of-5 and 3-of-3 are the same object, so a caller enforcing a "
            "quorum has to be told n separately",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
