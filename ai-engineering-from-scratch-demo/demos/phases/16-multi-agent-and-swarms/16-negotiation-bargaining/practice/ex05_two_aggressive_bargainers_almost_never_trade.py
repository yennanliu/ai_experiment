"""Exercise 5 — two aggressive bargainers almost never trade.

    Read Bhattacharya et al. 2025 on Harvard Negotiation Project metrics.
    Implement two bargainers with different styles (aggressive vs fair).
    Measure payoff variance under symmetric and asymmetric pairings.

Reading of the exercise: a style is an opening anchor plus a concession rate,
built on the lesson's own `og_narrator_bargain` and `seller_response` so the
concession arithmetic is theirs -- aggressive opens conceding 5% of the zone
and then 0.1 of each gap, fair opens conceding 35% and then 0.6. Payoff is
each side's share of the zone, 0 on no deal, over the lesson's 1000 seeded
trials; variance is read both within a pairing and across a style's
opponents, the second being the lesson's "fairest = smallest variance in
payoff across pairings".

**ANSWER: symmetric pairings sit at the extremes and asymmetric ones are
lopsided but stable.** Fair-fair deals 1000 of 1000 with per-trial payoff
variance 0.0004; aggressive-aggressive deals 39 of 1000 with variance 0.0082
(buyer) and 0.0107 (seller) -- 20-27x higher, almost all of it the gap
between the 961 zeros and the 39 deals. Aggressive against fair takes 0.82
of the zone as buyer and 0.87 as seller, with variance 0.0014 and 0.0008.
Across opponents a fair buyer's mean share ranges 0.129 to 0.434; an
aggressive buyer's 0.018 to 0.819 -- the lesson's fairness-as-low-variance
reading holds, and it is the deadlock that produces it.

**FINDING: two identical fair bargainers split 43/57, not 50/50.** The
buyer always moves first and a crossing clears at the standing offer, so the
protocol is asymmetric: with the same parameters on both sides the seller
takes 0.566 of the zone to the buyer's 0.434.

Note, measured in ex01 -- concession alone is not a style here. With the
lesson's own openings, sweeping the buyer's `concession` from 0.1 to 0.9 leaves 892 deals
(ex01): the opening bid already crosses the seller's counter on 719 of them.
A style has to include the anchor for the pairing to mean anything.

**FINDING: the "bound rounds" checklist item is what turns aggression into no
deal.** The aggressive pair's 961 failures are all 5-round timeouts; with
0.1 concessions the zone is not crossed in 5 rounds.

The paper itself could not be retrieved to check the lesson's
model rankings, so nothing here rests on them.

Structure: `buyer()` and `seller()` wrap the reference offer functions with a
style's anchor; `pairing()` runs the lesson's 1000 draws.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "16-negotiation-bargaining"
STYLES = {"aggressive": (0.05, 0.1), "fair": (0.35, 0.6)}  # (opening share, concession)


def buyer(ref, style):
    opening, concession = STYLES[style]

    def offer(s, rng):
        if s.buyer_offer is None:
            return s.seller_min + int(opening * (s.buyer_max - s.seller_min))
        return ref.og_narrator_bargain(s, rng, concession=concession)
    return offer


def seller(ref, style):
    opening, concession = STYLES[style]

    def offer(s, rng):
        if s.seller_offer is None:
            return s.buyer_max - int(opening * (s.buyer_max - s.seller_min))
        return ref.seller_response(s, rng, concession=concession)
    return offer


def replay(ref, buy, sell, buyer_max, seller_min):
    """(price or None, timed_out) under simulate_bargain's rules."""
    s = ref.BargainState(buyer_max=buyer_max, seller_min=seller_min)
    while s.rounds < s.max_rounds:
        s.buyer_offer = buy(s, None)
        if s.seller_offer is not None and s.buyer_offer >= s.seller_offer:
            return s.seller_offer, False
        s.seller_offer = sell(s, None)
        if s.seller_offer <= s.buyer_offer:
            return s.buyer_offer, False
        s.rounds += 1
    return None, True


def pairing(ref, buyer_style, seller_style):
    draw, shares, timeouts = random.Random(42), [], 0
    for _ in range(1000):
        seller_min = draw.randint(50, 80)
        buyer_max = draw.randint(max(seller_min + 5, 75), 115)
        price, timed_out = replay(ref, buyer(ref, buyer_style), seller(ref, seller_style),
                                  buyer_max, seller_min)
        gap, timeouts = buyer_max - seller_min, timeouts + timed_out
        shares.append((0, 0) if price is None else
                      ((buyer_max - price) / gap, (price - seller_min) / gap))
    sides = list(zip(*shares))
    return {"deals": sum(p != (0, 0) for p in shares), "timeouts": timeouts,
            "mean": [round(statistics.mean(side), 3) for side in sides],
            "var": [round(statistics.pvariance(side), 4) for side in sides]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {f"{b[0]}{s[0]}": pairing(ref, b, s) for b in STYLES for s in STYLES}


def verify(result):
    aa, af, fa, ff = result["aa"], result["af"], result["fa"], result["ff"]
    return [
        practice.Check(
            "ANSWER: symmetric pairings sit at the extremes; asymmetric ones are lopsided but stable",
            all([ff["deals"] == 1000, aa["deals"] == 39, min(aa["var"]) > 20 * max(ff["var"]),
                 af["mean"][0] > 0.8, fa["mean"][1] > 0.85]),
            f"fair-fair {ff['deals']} deals, variance {ff['var']}; aggressive-aggressive "
            f"{aa['deals']} deals, variance {aa['var']}; aggressive takes {af['mean'][0]} as "
            f"buyer and {fa['mean'][1]} as seller against fair; a fair buyer's share spans "
            f"{fa['mean'][0]}-{ff['mean'][0]} across opponents, an aggressive one's "
            f"{aa['mean'][0]}-{af['mean'][0]}",
        ),
        practice.Check(
            "FINDING: two identical fair bargainers split 43/57, not 50/50",
            ff["mean"] == [0.434, 0.566],
            f"same parameters on both sides give buyer {ff['mean'][0]}, seller "
            f"{ff['mean'][1]} -- the buyer moves first and a crossing clears at the "
            "standing offer",
        ),
        practice.Check(
            "FINDING: the bound on rounds is what turns aggression into no deal",
            aa["timeouts"] == 1000 - aa["deals"],
            f"all {aa['timeouts']} of the aggressive pair's failures are 5-round timeouts",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
