"""Exercise 3 — the state has no private side, and the buyer does better without the secret.

    Implement chain-of-thought **concealment**: maintain a private scratchpad
    string that is not passed to the counterpart. What happens if you
    accidentally leak it (simulate by swapping the channels)?

Reading of the exercise: the buyer keeps a scratchpad naming its reservation
and a public channel carrying the narrated offer; a leak hands the seller the
scratchpad instead. Before adding the split it is worth asking what the
shipped code already conceals -- the answer is nothing -- so the buyer here is
first made to stop reading the seller's reservation, and guesses the public
prior floor (50) instead.

**ANSWER: a leak costs the buyer 70% of its surplus -- or the whole trade.**
A seller that reads `reservation=<n>` in what it receives asks exactly that
and holds, then in the last round asks the buyer's standing bid. With the
channels swapped, all 1000 trials deal -- and buyer surplus falls from 7.34
to 2.23 per trial while the seller's rises from 23.66 to 29.25, because the
buyer spent five rounds creeping toward its own ceiling. A seller that only
holds at the reservation gets 0 deals in 1000: OG-Narrator concedes
`int(0.35 * remaining)` a round and never arrives. Either way the counterpart
now decides where the surplus goes.

**FINDING: the shipped `BargainState` has no private side.** Both reservations
live in one dataclass handed to both parties, and `og_narrator_bargain` reads
`state.seller_min` for its opening. `seller_response` reads `state.buyer_max`
only in its opening branch, and that branch runs 0 times in 1000 trials --
the buyer always moves first -- so the leak the exercise asks about is already
there, one way round.

**FINDING: the buyer does better without the seller's secret.** Replacing
the true `seller_min` with the prior floor 50 raises deals from 892 to 940 and
buyer surplus from 5.39 to 7.35 per trial. The opening is
`buyer_max - 0.2 * gap`; knowing the true gap makes the buyer open *closer*
to the seller, so the secret it reads is used against it.

Structure: `scratchpad()` and `narrate()` are the two channels; `leak=True`
swaps them. `exploiter()` is the seller that parses what it is handed, with
or without the last-round settlement.
"""

from __future__ import annotations

import dataclasses
import inspect
import random
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "16-negotiation-bargaining"
PRIOR_FLOOR = 50


def scratchpad(state, price):
    return f"reservation={state.buyer_max}; offering {price}"


def narrate(price):
    return f"I can offer {price}."


def concealed_buyer(ref):
    """OG-Narrator, made to guess the seller's reservation from the public prior."""
    return lambda s, rng: ref.og_narrator_bargain(dataclasses.replace(s, seller_min=PRIOR_FLOOR), rng)


def exploiter(ref, calls, hold_out):
    def respond(state, rng, received):
        calls.append(state.buyer_offer is None and state.seller_offer is None)
        found = re.search(r"reservation=(\d+)", received)
        if not found:
            return ref.seller_response(state, rng)
        last = state.rounds == state.max_rounds - 1
        return state.buyer_offer if last and hold_out else int(found.group(1))
    return respond


def replay(ref, buyer_fn, seller_fn, leak, buyer_max, seller_min):
    s = ref.BargainState(buyer_max=buyer_max, seller_min=seller_min)
    while s.rounds < s.max_rounds:
        s.buyer_offer = buyer_fn(s, None)
        if s.seller_offer is not None and s.buyer_offer >= s.seller_offer:
            return seller_min <= s.seller_offer <= buyer_max, s.seller_offer
        sent = scratchpad(s, s.buyer_offer) if leak else narrate(s.buyer_offer)
        s.seller_offer = seller_fn(s, None, sent)
        if s.seller_offer <= s.buyer_offer:
            return seller_min <= s.buyer_offer <= buyer_max, s.buyer_offer
        s.rounds += 1
    return False, None


def bench(ref, buyer_fn, leak, calls, hold_out=True):
    draw, deals, buyer, seller = random.Random(42), 0, 0, 0
    for _ in range(1000):
        seller_min = draw.randint(50, 80)
        buyer_max = draw.randint(max(seller_min + 5, 75), 115)
        deal, price = replay(ref, buyer_fn, exploiter(ref, calls, hold_out), leak, buyer_max, seller_min)
        if deal:
            deals, buyer, seller = deals + 1, buyer + buyer_max - price, seller + price - seller_min
    return {"deals": deals, "buyer": round(buyer / 1000, 2), "seller": round(seller / 1000, 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    calls = []
    return {
        "shipped": bench(ref, ref.og_narrator_bargain, False, calls),
        "concealed": bench(ref, concealed_buyer(ref), False, calls),
        "leaked": bench(ref, concealed_buyer(ref), True, calls),
        "ask_only": bench(ref, concealed_buyer(ref), True, calls, hold_out=False),
        "opening_calls": sum(calls), "seller_calls": len(calls),
        "buyer_reads": "state.seller_min" in inspect.getsource(ref.og_narrator_bargain),
        "seller_reads": "state.buyer_max" in inspect.getsource(ref.seller_response),
    }


def verify(result):
    shipped, concealed, leaked = result["shipped"], result["concealed"], result["leaked"]
    return [
        practice.Check(
            "ANSWER: a leak costs the buyer 70% of its surplus -- or the whole trade",
            all([leaked["deals"] == 1000, leaked["buyer"] < 0.35 * concealed["buyer"],
                 leaked["seller"] > concealed["seller"], result["ask_only"]["deals"] == 0]),
            f"with the channels swapped a seller holding at buyer_max and settling in the last "
            f"round gets {leaked['deals']} deals, buyer surplus {leaked['buyer']} per trial "
            f"against {concealed['buyer']} concealed, seller {leaked['seller']} against "
            f"{concealed['seller']}; one that only holds gets {result['ask_only']['deals']} deals",
        ),
        practice.Check(
            "FINDING: the shipped BargainState has no private side",
            all([result["buyer_reads"], result["seller_reads"], result["opening_calls"] == 0]),
            f"og_narrator_bargain reads state.seller_min; seller_response reads "
            f"state.buyer_max only in an opening branch reached {result['opening_calls']} "
            f"times in {result['seller_calls']} seller calls",
        ),
        practice.Check(
            "FINDING: the buyer does better without the seller's secret",
            all([shipped["deals"] == 892, concealed["deals"] == 940,
                 concealed["buyer"] > shipped["buyer"]]),
            f"guessing seller_min = {PRIOR_FLOOR}: deals {shipped['deals']} -> "
            f"{concealed['deals']}, buyer surplus {shipped['buyer']} -> {concealed['buyer']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
