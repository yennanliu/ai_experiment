"""Exercise 2 — a retry reruns an auction whose answer cannot change.

    Extend the contract-net demo with a `cancel` performative that lets the
    manager withdraw the task mid-bid. What failure case does `cancel` solve
    that retries alone do not?

Reading of the exercise: build the cancel, then build the retry it is being
compared against, and run both against the shipped auction -- because the
comparison only means something once you can see what a retry does here, and
what it does here is nothing.

**ANSWER: a retry answers "no reply"; cancel answers "reply no longer
wanted", and only one of those can reach a worker that has already been
awarded.** Re-issuing the call for proposals adds **3** messages to a **9**
message log and re-awards **worker-b** -- the same winner, because the bids
are fixed data and the scoring is deterministic, so the retry is a no-op that
costs three envelopes. The cancel is **1** message, `performative="cancel"`,
manager to the awarded bidder inside conversation `cn-1`, and it is the only
message in either log that tells a committed worker to stop.

**FINDING: `cancel` is already declared.** `PERFORMATIVES` lists **16** names
including `cancel`, `__post_init__` already accepts it, and the module
constructs **7** of the sixteen. The extension this exercise asks for is a
call site, not a protocol change -- the whitelist has been waiting for it.

**FINDING: the envelope cannot say what it is cancelling.** `ACLMessage` has
**9** fields. `reply_with` is one of them and `in_reply_to` is not, so a
cancel can name a conversation and cannot name the `accept-proposal` it
revokes. `reply_with` is set by **5** constructors and read **2** times, both
of them inside `render()` -- the correlation slot FIPA uses for exactly this is
populated, printed, and never consulted.

**FINDING: the auction is decided by an undeclared constant.** The winner is
`min(price + eta_minutes / 10)`, which prices ten minutes at one unit of money
and says so nowhere. worker-b and worker-c swap at a divisor of **7.5**, so
the shipped **10** sits **1.3x** above the flip. worker-a wins at no divisor
at all: it needs one below 7 to beat b and above 8 to beat c.

Structure: `run()` replays the shipped auction; `retry()` and `cancel()` are
the two responses the exercise compares.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "02-fipa-acl-heritage"
BIDS = {"worker-a": (3, 18), "worker-b": (2, 25), "worker-c": (4, 10)}
CONV, TASK, SHIPPED = "cn-1", "compress 10GB log bundle", 10


def score(bid, divisor):
    """The shipped objective, with its hidden exchange rate made a parameter."""
    price, eta = bid
    return price + eta / divisor


def winner(divisor):
    """Who the auction awards when ten minutes is worth one unit of money."""
    return min(BIDS, key=lambda name: score(BIDS[name], divisor))


def run(ref):
    """The shipped contract net: cfp to three bidders, three proposals, one award."""
    net = ref.ContractNet(manager="scheduler", bidders=sorted(BIDS))
    net.cfp(task=TASK, conv=CONV)
    for name, (price, eta) in BIDS.items():
        net.propose(name, ref.Bid(name, price=price, eta_minutes=eta), CONV)
    awarded = winner(SHIPPED)
    net.award(awarded, [n for n in BIDS if n != awarded], CONV)
    return net, awarded


def cancel(ref, awarded):
    """The extension: one message withdrawing the task from the committed worker."""
    return ref.ACLMessage(
        performative="cancel", sender="scheduler", receiver=awarded,
        content="withdrawn", ontology="contract-net",
        protocol="fipa-contract-net", conversation_id=CONV)


def sweep():
    """Every divisor on a hundredths grid, and who the auction awards at each."""
    return [(step / 100, winner(step / 100)) for step in range(100, 5000)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    net, awarded = run(ref)
    base = len(net.log)
    net.cfp(task=TASK, conv=CONV)
    message = cancel(ref, awarded)
    built = set(re.findall(r'performative="([a-z-]+)"', src))
    fields = [f.name for f in dataclasses.fields(ref.ACLMessage)]
    awards = sweep()
    return {
        "base": base, "awarded": awarded, "retry_added": len(net.log) - base,
        "retry_winner": winner(SHIPPED), "cancel_added": 1,
        "cancel_to": message.receiver, "cancel_performative": message.performative,
        "cancel_conv": message.conversation_id,
        "declared": len(ref.PERFORMATIVES), "built": len(built),
        "cancel_declared": "cancel" in ref.PERFORMATIVES,
        "cancel_was_built": "cancel" in built,
        "fields": len(fields), "has_in_reply_to": "in_reply_to" in fields,
        "reply_with_set": src.count("reply_with="),
        "reply_with_read": src.count(".reply_with"),
        "read_in_render": inspect.getsource(ref.ACLMessage.render).count(".reply_with"),
        "flip": next(round(d + 0.005, 2) for (d, w), (_, nxt) in zip(awards, awards[1:])
                     if w != nxt),
        "a_ever_wins": any(who == "worker-a" for _, who in awards),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a retry answers 'no reply'; cancel answers 'reply no longer wanted'",
            all([result["base"] == 9, result["retry_added"] == 3,
                 result["retry_winner"] == result["awarded"] == "worker-b",
                 result["cancel_added"] == 1, result["cancel_performative"] == "cancel",
                 result["cancel_to"] == "worker-b", result["cancel_conv"] == CONV]),
            f"the retry adds {result['retry_added']} messages to a {result['base']} "
            f"message log and re-awards {result['retry_winner']}, the same winner, since "
            f"the bids are fixed and the scoring deterministic; the cancel is "
            f"{result['cancel_added']} message to {result['cancel_to']} in "
            f"{result['cancel_conv']}, the only one reaching a committed worker",
        ),
        practice.Check(
            "FINDING: cancel is already declared",
            all([result["cancel_declared"], not result["cancel_was_built"],
                 result["declared"] == 16, result["built"] == 7]),
            f"PERFORMATIVES lists {result['declared']} names including cancel, already "
            f"accepted by __post_init__, and the module constructs {result['built']} -- "
            "the extension asked for is a call site, not a protocol change",
        ),
        practice.Check(
            "FINDING: the envelope cannot say what it is cancelling",
            all([result["fields"] == 9, not result["has_in_reply_to"],
                 result["reply_with_set"] == 5,
                 result["reply_with_read"] == result["read_in_render"] == 2]),
            f"ACLMessage has {result['fields']} fields, reply_with among them and "
            f"in_reply_to absent, so a cancel names a conversation and not the award it "
            f"revokes; reply_with is set {result['reply_with_set']} times and read "
            f"{result['reply_with_read']}, both inside render() -- "
            f"{result['reply_with_read'] - result['read_in_render']} reads that could "
            "correlate a reply",
        ),
        practice.Check(
            "FINDING: the auction is decided by an undeclared constant",
            all([result["flip"] == 7.5, not result["a_ever_wins"],
                 result["awarded"] == "worker-b"]),
            f"price + eta_minutes / {SHIPPED} prices ten minutes at one unit of money "
            f"and says so nowhere; worker-b and worker-c swap at {result['flip']}, so "
            f"the shipped {SHIPPED} sits {SHIPPED / result['flip']:.1f}x above the flip, "
            "and worker-a wins at no divisor at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
