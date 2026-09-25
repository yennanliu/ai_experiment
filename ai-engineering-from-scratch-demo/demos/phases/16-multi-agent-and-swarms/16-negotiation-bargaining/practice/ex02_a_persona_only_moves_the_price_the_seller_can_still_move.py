"""Exercise 2 — a persona only moves the price the seller can still move.

    Implement **persona-based payoff improvement** (arXiv:2402.05863) — the
    buyer adopts a "desperate to buy this week" persona in the narration only,
    offer generator unchanged. Does the deal rate or payoff change?

Reading of the exercise: the narrator writes a sentence around each
deterministic price, with or without the persona, and the question is asked
twice -- against the lesson's seller, and against a seller that reads the
message. The reading seller is a modelling choice, named as one: NegotiationArena
found desperation *raises* payoffs, so it concedes 0.5 instead of 0.3 when it
reads "desperate". Buyer randomness is seeded per trial, so every comparison
is paired.

**ANSWER: against the lesson's seller, nothing changes -- not one price in
1000 trials.** `seller_response(state, rng)` has no parameter a message
could arrive through, so the narration is discarded before anyone reads it:
OG-Narrator's 892 deals and naive's 615 are byte-identical with and without
the persona. The effect NegotiationArena measured -- "pretending to be
desolate and desperate", +20% "against the standard GPT-4" -- exists only
where the counterpart reads text.

**FINDING: a reading seller lifts naive's payoff about 15% and OG-Narrator's
under 2%.** With a sympathetic seller naive's buyer surplus goes from 10.92 to
12.58 per trial, near the paper's 20%. OG-Narrator's goes from 5.39 to
5.48, although its deal rate jumps from 892 to 1000 -- the persona rescues
exactly the 108 narrow-ZOPA timeouts. The reason is ex01's: 719 of its deals
close at the buyer's own opening bid, before any seller concession applies.
A persona is a lever on the counterpart's concession, and OG-Narrator has
already given away what that lever could have won back.

**FINDING: the offer generator is unchanged, as the exercise requires.**
Every buyer price in every transcript is the same with and without the
persona; only the text differs.

Structure: `narrate()` is the LLM narrator stand-in; `replay()` runs the
lesson's protocol with a message channel added; `reading_seller()` is the
one seller that looks at it.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "16-negotiation-bargaining"
PERSONA = "I'm desperate to buy this week. "


def narrate(price, persona):
    return (PERSONA if persona else "") + f"I can offer {price}."


def reading_seller(ref):
    def respond(state, rng, message):
        return ref.seller_response(state, rng, concession=0.5 if "desperate" in message else 0.3)
    return respond


def blind_seller(ref):
    return lambda state, rng, message: ref.seller_response(state, rng)


def replay(ref, buyer_fn, seller_fn, persona, buyer_max, seller_min, rng):
    """(deal, price, buyer prices) for one trial, with the narration passed across."""
    s, bids = ref.BargainState(buyer_max=buyer_max, seller_min=seller_min), []
    while s.rounds < s.max_rounds:
        s.buyer_offer = buyer_fn(s, rng)
        bids.append(s.buyer_offer)
        if s.seller_offer is not None and s.buyer_offer >= s.seller_offer:
            return seller_min <= s.seller_offer <= buyer_max, s.seller_offer, bids
        s.seller_offer = seller_fn(s, rng, narrate(s.buyer_offer, persona))
        if s.seller_offer <= s.buyer_offer:
            return seller_min <= s.buyer_offer <= buyer_max, s.buyer_offer, bids
        s.rounds += 1
    return False, None, bids


def bench(ref, buyer_fn, seller_fn, persona):
    draw, deals, surplus, transcripts = random.Random(42), 0, 0, []
    for trial in range(1000):
        seller_min = draw.randint(50, 80)
        buyer_max = draw.randint(max(seller_min + 5, 75), 115)
        deal, price, bids = replay(ref, buyer_fn, seller_fn, persona, buyer_max, seller_min,
                                   random.Random(trial))
        deals, surplus = deals + deal, surplus + (buyer_max - price if deal else 0)
        transcripts.append(bids)
    return {"deals": deals, "surplus": round(surplus / 1000, 2), "bids": transcripts}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {}
    for buyer in ("og_narrator_bargain", "naive_llm_bargain"):
        for seller, make in (("blind", blind_seller), ("reading", reading_seller)):
            for persona in (False, True):
                runs[buyer[:5], seller, persona] = bench(ref, getattr(ref, buyer), make(ref), persona)
    return {"runs": runs, "params": list(inspect.signature(ref.seller_response).parameters)}


def verify(result):
    r = result["runs"]
    og_lift = r["og_na", "reading", True]["surplus"] / r["og_na", "reading", False]["surplus"] - 1
    nv_lift = r["naive", "reading", True]["surplus"] / r["naive", "reading", False]["surplus"] - 1
    return [
        practice.Check(
            "ANSWER: against the lesson's seller, nothing changes",
            all([r[b, "blind", True] == r[b, "blind", False] for b in ("og_na", "naive")]
                + [result["params"] == ["state", "rng", "concession"]]),
            f"seller_response's parameters are {result['params']} -- no message channel -- "
            f"so OG's {r['og_na', 'blind', True]['deals']} and naive's "
            f"{r['naive', 'blind', True]['deals']} deals are identical with the persona",
        ),
        practice.Check(
            "FINDING: a reading seller lifts naive's payoff ~15% and OG-Narrator's under 2%",
            all([0.12 < nv_lift < 0.22, 0 < og_lift < 0.02,
                 r["og_na", "reading", False]["deals"] == 892,
                 r["og_na", "reading", True]["deals"] == 1000]),
            f"naive surplus {r['naive', 'reading', False]['surplus']} -> "
            f"{r['naive', 'reading', True]['surplus']} (+{nv_lift:.0%}); OG "
            f"{r['og_na', 'reading', False]['surplus']} -> {r['og_na', 'reading', True]['surplus']} "
            f"(+{og_lift:.1%}) while its deals go 892 -> 1000",
        ),
        practice.Check(
            "FINDING: the offer generator is unchanged",
            r["og_na", "blind", True]["bids"] == r["og_na", "blind", False]["bids"],
            "every OG buyer price in all 1000 transcripts is identical with and without "
            "the persona; only the narration differs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
