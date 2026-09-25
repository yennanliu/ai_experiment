"""Exercise 1 — OG-Narrator wins the deal rate by giving the buyer's surplus away.

    Run `code/main.py`. Confirm OG-Narrator beats naive-LLM on deal rate. By
    how much?

Reading of the exercise: the deal rate is read from the lesson's own
`bench_deal_rate`, and then each trial is replayed with the same seed so the
deals can be priced -- a deal rate says how often a trade happens, not who
it was good for, and the paper this lesson cites claims both.

**ANSWER: yes, by 28.7 points -- 89.2% against 60.5%.** That is outside the
lesson's stated "15-25 point gap", and naive's 60.5% is below its "~65-75%".
Of naive's 395 failures, 360 cross at a price outside a reservation and 35
run out of rounds.

**FINDING: the buyer earns half as much with OG-Narrator.** 719 of its 892
deals close in round 0 at the buyer's opening bid, which is
`buyer_max - 0.2 * (buyer_max - seller_min)` -- the buyer offers 80% of the
surplus before the seller has spoken. Mean buyer surplus per trial is 5.39
with OG-Narrator against 10.83 naive; the seller's is 25.06 against 8.37.
arXiv:2402.15813 reports a deal rate of 26.67% to 88.88% *and* "a ten times
multiplication of profits"; the demo reproduces the first number and inverts
the second.

**FINDING: OG-Narrator's failures are a cliff at a ZOPA of 13.** All 108 of
its no-deals have `buyer_max - seller_min <= 13`, and every trial with a gap
of 14 or more deals. Concessions are `max(1, int(...))`, so a narrow zone is
crossed one unit per round and 5 rounds is not enough. The demo's closing
line, "converges on every trial", is false on 10.8% of them.

**FINDING: the buyer's own strategy parameter does nothing, and its opening
reads the seller's secret.** Sweeping `concession` from 0.1 to 0.9 leaves 892
deals every time and buyer surplus within 0.01. The only input that moves
the result is `state.seller_min`, the seller's reservation, which
`og_narrator_bargain` reads directly -- in a game the paper defines as one of
"incomplete information".

Structure: `replay()` is `simulate_bargain` with the clearing price kept;
`bench()` redraws reservations exactly as `bench_deal_rate` does.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "16-negotiation-bargaining"


def replay(ref, buyer_fn, buyer_max, seller_min, rng):
    """(deal, price, round) -- simulate_bargain, with the clearing price kept."""
    s = ref.BargainState(buyer_max=buyer_max, seller_min=seller_min)
    while s.rounds < s.max_rounds:
        s.buyer_offer = buyer_fn(s, rng)
        if s.seller_offer is not None and s.buyer_offer >= s.seller_offer:
            return seller_min <= s.seller_offer <= buyer_max, s.seller_offer, s.rounds
        s.seller_offer = ref.seller_response(s, rng)
        if s.seller_offer <= s.buyer_offer:
            return seller_min <= s.buyer_offer <= buyer_max, s.buyer_offer, s.rounds
        s.rounds += 1
    return False, None, s.rounds


def bench(ref, buyer_fn):
    """Per-trial (seller_min, buyer_max, deal, price, round), seeded as bench_deal_rate."""
    rng, rows = random.Random(42), []
    for _ in range(1000):
        seller_min = rng.randint(50, 80)
        buyer_max = rng.randint(max(seller_min + 5, 75), 115)
        rows.append((seller_min, buyer_max, *replay(ref, buyer_fn, buyer_max, seller_min, rng)))
    return rows


def summary(rows):
    deals = [r for r in rows if r[2]]
    return {"deals": len(deals), "round0": sum(r[4] == 0 for r in deals),
            "timeouts": sum(r[3] is None for r in rows),
            "buyer": round(sum(r[1] - r[3] for r in deals) / len(rows), 2),
            "seller": round(sum(r[3] - r[0] for r in deals) / len(rows), 2),
            "fail_gaps": sorted({r[1] - r[0] for r in rows if not r[2]}),
            "deal_gaps": sorted({r[1] - r[0] for r in deals})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        ref.bench_deal_rate(ref.naive_llm_bargain, "naive")
        ref.bench_deal_rate(ref.og_narrator_bargain, "og")
    sweep = {c: summary(bench(ref, lambda s, r, c=c: ref.og_narrator_bargain(s, r, concession=c)))
             for c in (0.1, 0.5, 0.9)}
    return {"printed": printed.getvalue(), "naive": summary(bench(ref, ref.naive_llm_bargain)),
            "og": summary(bench(ref, ref.og_narrator_bargain)), "sweep": sweep,
            "reads_secret": "state.seller_min" in inspect.getsource(ref.og_narrator_bargain),
            "claim": "converges on every trial" in inspect.getsource(ref.main)}


def verify(result):
    naive, og, sweep = result["naive"], result["og"], result["sweep"]
    return [
        practice.Check(
            "ANSWER: yes, by 28.7 points -- 89.2% against 60.5%",
            all(["(605/1000)" in result["printed"], "(892/1000)" in result["printed"],
                 naive["deals"] == 605, og["deals"] == 892]),
            f"the lesson's bench prints 605/1000 and 892/1000; naive's 395 failures are "
            f"{395 - naive['timeouts']} infeasible crossings and {naive['timeouts']} timeouts",
        ),
        practice.Check(
            "FINDING: the buyer earns half as much with OG-Narrator",
            og["round0"] == 719 and og["buyer"] * 2 < naive["buyer"] and og["seller"] > 25,
            f"{og['round0']} of {og['deals']} deals close at the buyer's opening bid; mean "
            f"buyer surplus per trial {og['buyer']} against naive's {naive['buyer']}, seller "
            f"{og['seller']} against {naive['seller']} -- the paper claims 10x profit",
        ),
        practice.Check(
            "FINDING: OG-Narrator's failures are a cliff at a ZOPA of 13",
            all([og["timeouts"] == 108, max(og["fail_gaps"]) == 13,
                 min(og["deal_gaps"]) == 14, result["claim"]]),
            f"all {og['timeouts']} no-deals have gap <= {max(og['fail_gaps'])} and every gap "
            f">= {min(og['deal_gaps'])} deals; main() says it 'converges on every trial'",
        ),
        practice.Check(
            "FINDING: the buyer's own strategy parameter does nothing; its opening reads the seller's secret",
            all([{s["deals"] for s in sweep.values()} == {892},
                 max(s["buyer"] for s in sweep.values()) - min(s["buyer"] for s in sweep.values()) <= 0.01,
                 result["reads_secret"]]),
            f"concession 0.1/0.5/0.9 gives deals {[s['deals'] for s in sweep.values()]} and "
            f"buyer surplus {[s['buyer'] for s in sweep.values()]}; og_narrator_bargain reads "
            "state.seller_min, the counterpart's reservation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
