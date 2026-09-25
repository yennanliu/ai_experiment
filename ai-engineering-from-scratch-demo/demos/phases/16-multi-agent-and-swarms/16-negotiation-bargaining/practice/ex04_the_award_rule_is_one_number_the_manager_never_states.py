"""Exercise 4 — the award rule is one number the manager never states.

    Extend Contract Net to N-bidder auction with reserve price. When bids all
    exceed reserve, how does the manager decide between lowest-price and
    highest-quality? Which award rule do you pick and why?

Reading of the exercise: in a procurement auction the reserve is the most the
manager will pay, which `award` already enforces as `task.budget`; "all bids
exceed reserve" is read as all of them clearing it, so the rule, not the
filter, picks the winner. Quality is read as the chance of finishing by the
deadline, retries included -- confidence alone ignores time.

**ANSWER: pick the bid maximising V * P(done by deadline) - expected
payment, where V is what the task is worth.** Lowest-price is the V -> 0
limit and highest-quality the V -> infinity limit, so the choice between them
*is* the choice of V. On the demo's three bids, with the 30-minute deadline,
worker-c (eta 10) fits 3 attempts and finishes with probability 0.999, while
worker-a and worker-b fit one each (0.82, 0.77). worker-b wins for V below
10.66 and worker-c above it -- the crossover sits just above the 10 budget.

**FINDING: the shipped conf/price rule is that formula with the deadline
removed.** Maximising conf / price orders bids exactly as minimising
price / conf, the expected spend of retrying until success with unlimited
time. So it awards worker-b -- the least confident and the slowest, with 5 of
30 minutes to spare and no room for a retry.

**FINDING: conf/price flips its winner under a fixed fee.** Add the same k
to every price -- a call-out charge, or quoting in a different unit with an
offset -- and the ranking changes although no bid's relative standing did:
at k = 10, with the budget raised by the same 10, the winner moves from
worker-b to worker-c. A ratio of price is not
invariant to where price's zero is.

**FINDING: with 10 bidders the shipped rule agrees with the value rule 80%
of the time at V = budget and 58% at V = 3x budget.** Over 1000 seeded
10-bidder auctions, its winners finish on time with mean probability 0.866,
against 0.916 and 0.959 for the value rule -- the ratio rewards cheapness
hyperbolically and the deadline not at all.

Structure: `on_time()` and `spend()` score a bid with retries inside the
deadline; `value_award()` is the rule picked; the reference manager awards
every auction, printing silenced.
"""

from __future__ import annotations

import contextlib
import io
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "16-negotiation-bargaining"
DEMO = [("worker-a", 3, 18, 0.82), ("worker-b", 2, 25, 0.77), ("worker-c", 4, 10, 0.90)]


def on_time(bid, deadline):
    return 1 - (1 - bid.confidence) ** (deadline // bid.eta_minutes)


def spend(bid, deadline):
    """Expected payment: one price per attempt, attempts stop at success."""
    return bid.price * sum((1 - bid.confidence) ** i for i in range(deadline // bid.eta_minutes))


def value_award(bids, task, value):
    feasible = [b for b in bids if b.price <= task.budget and b.eta_minutes <= task.deadline_minutes]
    return max(feasible, key=lambda b: value * on_time(b, task.deadline_minutes)
               - spend(b, task.deadline_minutes))


def shipped_award(ref, bids, task):
    manager = ref.ContractNetManager([b.bidder for b in bids])
    with contextlib.redirect_stdout(io.StringIO()):
        manager.broadcast_cfp(task)
        for bid in bids:
            manager.receive_proposal(task.task_id, bid)
        return manager.award(task)


def crossover(bids, task):
    """Smallest V (to 0.01) at which the value rule's winner changes."""
    start = value_award(bids, task, 0).bidder
    return next(v / 100 for v in range(1, 10000) if value_award(bids, task, v / 100).bidder != start)


def auctions(ref, task, count=1000, bidders=10):
    rng, rows = random.Random(16), []
    for _ in range(count):
        bids = [ref.Bid(f"w{i}", rng.randint(1, 12), rng.randint(5, 35), round(rng.uniform(0.5, 0.99), 2))
                for i in range(bidders)]
        winner = shipped_award(ref, bids, task)
        if winner is not None:
            rows.append((winner, value_award(bids, task, task.budget),
                         value_award(bids, task, 3 * task.budget)))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    task = ref.ContractNetTask("t-1", "compress 10GB log bundle", 30, 10)
    bids = [ref.Bid(*row) for row in DEMO]
    shifted = [ref.Bid(b.bidder, b.price + 10, b.eta_minutes, b.confidence) for b in bids]
    rows = auctions(ref, task)
    return {
        "winner": shipped_award(ref, bids, task).bidder,
        "shifted": shipped_award(ref, shifted, ref.ContractNetTask("t-1", "same", 30, 20)).bidder,
        "on_time": {b.bidder: round(on_time(b, 30), 3) for b in bids},
        "same_order": sorted(bids, key=lambda b: -b.confidence / b.price)
                      == sorted(bids, key=lambda b: b.price / b.confidence),
        "crossover": crossover(bids, task), "high": value_award(bids, task, 11).bidder,
        "auctions": len(rows),
        "agree": [sum(r[0] is r[k] for r in rows) for k in (1, 2)],
        "p": [round(sum(on_time(r[k], 30) for r in rows) / len(rows), 3) for k in (0, 1, 2)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: maximise V * P(done by deadline) - expected payment",
            all([result["on_time"] == {"worker-a": 0.82, "worker-b": 0.77, "worker-c": 0.999},
                 result["crossover"] == 10.66, result["high"] == "worker-c"]),
            f"on-time probabilities {result['on_time']}; worker-b wins below "
            f"V = {result['crossover']} and {result['high']} above -- the choice between "
            "lowest-price and highest-quality is the choice of V",
        ),
        practice.Check(
            "FINDING: the shipped conf/price rule is that formula with the deadline removed",
            result["same_order"] and result["winner"] == "worker-b",
            f"conf/price orders bids exactly as price/conf, retrying forever; it awards "
            f"{result['winner']}, the least confident and slowest bidder",
        ),
        practice.Check(
            "FINDING: conf/price flips its winner under a fixed fee",
            result["shifted"] == "worker-c",
            f"adding 10 to every price moves the award from {result['winner']} to "
            f"{result['shifted']} -- a ratio of price depends on where price's zero is",
        ),
        practice.Check(
            "FINDING: with 10 bidders the shipped rule agrees 80% at V = budget and 58% at 3x",
            all([result["auctions"] == 1000, result["agree"] == [800, 584],
                 result["p"][0] < result["p"][1] < result["p"][2]]),
            f"over {result['auctions']} auctions conf/price matches the value rule "
            f"{result['agree'][0]} and {result['agree'][1]} times; mean on-time "
            f"probability of the winner {result['p']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
