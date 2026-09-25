"""Exercise 3 — every profitable coalition is a bidding ring.

    Implement a coalition-forming step before the auction: agents can merge
    into teams and bid as a unit. Which coalitions form? Is the outcome
    Pareto-better than individual bidding?

Reading of the exercise: a team bids as a unit with the value of its best
member -- the slot is one task, done by one agent -- and each of the 26
possible teams of two or more is run through the reference `second_price`
against everyone left outside it, with the demo's five bids as values.

**ANSWER: only rings holding both agent-c and agent-a form, and the outcome
is a transfer, not a Pareto improvement.** 7 of the 26 teams change anything,
and all 7 contain the winner (0.95) and the runner-up (0.82). Their gain is
0.82 minus the best bid left outside: 0.05 for {a, c}, 0.22 once agent-e
joins, 0.37 for {a, b, c, e}. In all 26 runs agent-c's slot goes to agent-c's
value, so welfare stays 0.95; the ring's gain is exactly the auctioneer's
lost revenue. Nobody else is better off and the auctioneer is worse off.

**FINDING: the grand coalition switches the auction off.** All five as one
team leaves one bid, and `second_price` returns None for fewer than two --
no winner, no payment. The mechanism has no reserve price, so its revenue
floor is whatever the ring leaves outside.

**FINDING: Shapley pays the losers of the auction.** Splitting the 0.37 of
the four-member ring with the reference `shapley_exact`: agent-c and agent-a
get 0.1192 each, agent-e 0.0942 and agent-b 0.0375 -- the same share for the
runner-up, who could never win, as for the winner. That is the economics of
a bidding ring, and the lesson's fair-credit tool computes it without
complaint.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "21-agent-economies"
BIDS = {"agent-a": 0.82, "agent-b": 0.60, "agent-c": 0.95, "agent-d": 0.45, "agent-e": 0.77}


def auction(ref, team):
    """Run second_price with `team` merged into one bid at its best member's value."""
    bids = [ref.Bid(name, v) for name, v in BIDS.items() if name not in team]
    if team:
        bids.append(ref.Bid("ring", max(BIDS[m] for m in team)))
    return ref.second_price(bids)


def gain(ref, team, individual):
    """Payment the winner saves by joining `team`, or None when the auction refuses."""
    result = auction(ref, team)
    if result is None:
        return None
    winner, payment = result
    return round(individual - payment, 6) if winner == "ring" and "agent-c" in team else 0.0


def winner(ref, team):
    """Who holds the slot: the ring's best member when the ring wins."""
    name = auction(ref, team)[0]
    return max(team, key=BIDS.get) if name == "ring" else name


def all_teams(ref, individual):
    """Every team of two or more, with what it saves (None where the auction refuses)."""
    teams = [frozenset(c) for k in range(2, 6) for c in itertools.combinations(BIDS, k)]
    return {t: gain(ref, t, individual) for t in teams}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    individual = auction(ref, frozenset())[1]
    gains = all_teams(ref, individual)
    teams = list(gains)
    ring = frozenset({"agent-a", "agent-b", "agent-c", "agent-e"})
    members = sorted(ring)
    split = ref.shapley_exact(lambda s: gains.get(s) or 0.0, members)
    winners = {winner(ref, t) for t in teams if gains[t] is not None}
    profitable = {t: g for t, g in gains.items() if g}
    return {
        "individual": individual, "teams": len(teams),
        "profitable": sorted((sorted(t), g) for t, g in profitable.items()),
        "all_hold_top2": all({"agent-a", "agent-c"} <= t for t in profitable),
        "winners": sorted(winners), "grand": gains[frozenset(BIDS)],
        "split": {m: round(v, 4) for m, v in split.items()}, "ring_gain": gains[ring],
    }


def verify(result):
    split, gains = result["split"], dict((tuple(t), g) for t, g in result["profitable"])
    return [
        practice.Check(
            "ANSWER: only rings holding agent-c and agent-a form; the outcome is a transfer",
            all([result["teams"] == 26, len(result["profitable"]) == 7,
                 result["all_hold_top2"], result["winners"] == ["agent-c"],
                 gains[("agent-a", "agent-c")] == 0.05, result["ring_gain"] == 0.37]),
            f"{len(result['profitable'])} of {result['teams']} teams gain, all holding "
            f"agent-c and agent-a; gains run from 0.05 to {result['ring_gain']}; agent-c "
            f"wins in every run, so welfare stays 0.95 and the ring's gain is exactly the "
            f"auctioneer's lost revenue from {result['individual']}",
        ),
        practice.Check(
            "FINDING: the grand coalition switches the auction off",
            result["grand"] is None,
            "all five as one team leaves one bid and second_price returns None -- no "
            "winner, no payment, no reserve price",
        ),
        practice.Check(
            "FINDING: Shapley pays the losers of the auction",
            split["agent-a"] == split["agent-c"] and abs(sum(split.values()) - 0.37) < 1e-3,
            f"shapley_exact splits the four-member ring's 0.37 as {split} -- the runner-up "
            "who could never win gets the winner's share",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
