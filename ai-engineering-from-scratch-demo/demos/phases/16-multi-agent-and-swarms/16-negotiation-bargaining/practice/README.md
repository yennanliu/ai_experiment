<!-- generated:start -->
# 16-multi-agent-and-swarms / 16-negotiation-bargaining

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/16-negotiation-bargaining/) · upstream spec
`phases/16-multi-agent-and-swarms/16-negotiation-bargaining/docs/en.md`

```bash
uv run demo practice run 16-negotiation-bargaining --ex 1
uv run demo explain 16-negotiation-bargaining --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/16-negotiation-bargaining
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm OG-Narrator beats naive-LLM on deal rate. By how much? | code | T0 | `ex01_og_narrator_wins_the_deal_rate_by_giving_the_buyers_surplus_away.py` |
| 2 | Implement persona-based payoff improvement (arXiv:2402.05863) — the buyer adopts a "desperate… | code | T0 | `ex02_a_persona_only_moves_the_price_the_seller_can_still_move.py` |
| 3 | Implement chain-of-thought concealment: maintain a private scratchpad string that is not pass… | code | T0 | `ex03_the_state_has_no_private_side_and_the_buyer_does_better_without_the_secret.py` |
| 4 | Extend Contract Net to N-bidder auction with reserve price. When bids all exceed reserve, how… | code | T0 | `ex04_the_award_rule_is_one_number_the_manager_never_states.py` |
| 5 | Read Bhattacharya et al. 2025 on Harvard Negotiation Project metrics. Implement two bargainer… | code | T0 | `ex05_two_aggressive_bargainers_almost_never_trade.py` |
<!-- generated:end -->

## Answers

### 1 — OG-Narrator wins the deal rate by giving the buyer's surplus away

**Yes, by 28.7 points: 89.2% against 60.5%.** That is outside the lesson's
stated 15–25 point gap, and naive's 60.5% is below its "~65–75%". Of naive's
395 failures, 360 cross at a price outside a reservation and 35 run out of
rounds.

The deal rate hides who the deals were good for:

| | deals | closed at the buyer's opening bid | buyer surplus / trial | seller surplus / trial |
|---|---:|---:|---:|---:|
| naive LLM | 605 | — | **10.83** | 8.37 |
| OG-Narrator | 892 | 719 | **5.39** | 25.06 |

OG-Narrator's opening bid is `buyer_max - 0.2 * (buyer_max - seller_min)`, so
the buyer offers 80% of the surplus before the seller has said anything.
arXiv:2402.15813 reports a deal rate going from 26.67% to 88.88% *and* "a ten
times multiplication of profits". The demo reproduces the first number and
inverts the second.

Three more things come out of the same replay:

- **The failures are a cliff at a ZOPA of 13.** All 108 no-deals have
  `buyer_max - seller_min <= 13`, and every trial with a gap of 14 or more
  deals. Concessions are `max(1, int(...))`, so a narrow zone closes one unit
  per round, and 5 rounds are not enough. `main()` says OG-Narrator
  "converges on every trial".
- **The buyer's strategy parameter does nothing.** `concession` 0.1, 0.5 and
  0.9 all give 892 deals, and buyer surplus changes by at most 0.01.
- **The opening reads the seller's secret.** `og_narrator_bargain` uses
  `state.seller_min`, in a game the paper defines as one of incomplete
  information.

### 2 — a persona only moves the price the seller can still move

**Against the lesson's seller nothing changes, not one price in 1000
trials.** `seller_response(state, rng, concession)` has no parameter a message
could arrive through, so the narration is thrown away before anyone reads it.
NegotiationArena's effect ("pretending to be desolate and desperate", +20%
"against the standard GPT-4") needs a counterpart that reads the text.

So the exercise needs a seller that reads. Modelling one that concedes 0.5
instead of 0.3 when it sees "desperate" (paired trials, same draws):

| buyer | surplus without persona | with persona | change | deals |
|---|---:|---:|---:|---|
| naive | 10.92 | 12.58 | **+15%** | — |
| OG-Narrator | 5.39 | 5.48 | **+1.7%** | 892 → 1000 |

Naive's +15% is close to the paper's figure. OG-Narrator hardly moves,
because 719 of its deals close at its own opening bid, before any seller
concession applies. The persona does rescue exactly the 108 narrow-ZOPA
timeouts. A persona pulls on the counterpart's concession, and OG-Narrator
has already given away what that pull could win back. The offer generator is
unchanged throughout: every OG buyer price is identical with and without the
persona.

### 3 — the state has no private side, and the buyer does better without the secret

Look at what the shipped code already hides before adding a scratchpad. It
hides nothing. Both reservations live in one `BargainState` that both
parties receive. `og_narrator_bargain` reads `state.seller_min`.
`seller_response` reads `state.buyer_max`, but only in an opening branch that
runs **0** times, because the buyer always moves first. So the leak the
exercise asks about already exists, in one direction.

**The buyer does better without the seller's secret.** Have it guess the
public prior floor, 50, instead of reading the true `seller_min`: deals rise
from 892 to **940** and buyer surplus from 5.39 to **7.34** per trial.
Knowing the true zone makes it open closer to the seller.

**A leak costs the buyer 70% of its surplus, or the whole trade.** The
scratchpad says `reservation=<n>`, and a leak hands it to the seller instead
of the narrated offer:

| seller, given the scratchpad | deals | buyer surplus | seller surplus |
|---|---:|---:|---:|
| (concealed: gets the message) | 940 | 7.34 | 23.66 |
| holds at the reservation, settles in the last round | 1000 | **2.23** | 29.25 |
| only holds at the reservation | **0** | 0 | 0 |

OG-Narrator concedes `int(0.35 * remaining)` per round and never quite reaches
its own ceiling. A seller who waits it out and then takes the standing bid
captures most of the zone. A seller who only waits kills every deal. Either
way, the counterpart now decides where the surplus goes.

### 4 — the award rule is one number the manager never states

**Pick the bid that maximises V · P(done by deadline) − expected payment,
where V is what the task is worth.** Lowest price is the V → 0 limit and
highest quality the V → ∞ limit, so choosing between them means choosing V.
Quality here is the chance of finishing by the deadline, retries included:

| bid | price | eta | fits in 30 min | P(on time) |
|---|---:|---:|---:|---:|
| worker-a | 3 | 18 | 1 attempt | 0.82 |
| worker-b | 2 | 25 | 1 attempt | 0.77 |
| worker-c | 4 | 10 | 3 attempts | **0.999** |

worker-b wins for V below **10.66** and worker-c above it. The crossover sits
just above the 10 budget.

The shipped `conf / price` is that formula with the deadline taken out. It
ranks bids exactly as `price / conf` does, which is the expected spend if you
retry until success with unlimited time. So it awards worker-b, the least
confident and slowest bidder, with no room for a retry. It is also not
invariant to a fixed fee: add 10 to every price (and to the budget) and the
winner moves to worker-c, although no bid changed relative to the others.

Across 1000 seeded 10-bidder auctions, `conf / price` agrees with the value
rule **800** times at V = budget and **584** times at V = 3× budget. Its
winners finish on time with mean probability 0.866, against 0.916 and 0.959.

### 5 — two aggressive bargainers almost never trade

A style here is an opening anchor plus a concession rate, built on the
reference offer functions. Aggressive opens by conceding 5% of the zone and
then concedes 0.1 of each gap. Fair opens at 35% and then concedes 0.6.
Payoff is each side's share of the zone, and 0 on no deal:

| buyer / seller | deals | buyer share | seller share | variance (buyer, seller) |
|---|---:|---:|---:|---|
| fair / fair | 1000 | 0.434 | 0.566 | 0.0004, 0.0004 |
| aggressive / aggressive | **39** | 0.018 | 0.021 | **0.0082, 0.0107** |
| aggressive / fair | 1000 | **0.819** | 0.181 | 0.0014, 0.0014 |
| fair / aggressive | 1000 | 0.129 | **0.871** | 0.0008, 0.0008 |

Symmetric pairings sit at the extremes. Fair–fair is the most stable result
in the table. Aggressive–aggressive has 20–27× the variance, and almost all of
it is the difference between 961 zeros and 39 deals. All 961 failures are
5-round timeouts, so the checklist's "bound rounds" rule is what turns
aggression into no deal. Asymmetric pairings are lopsided but stable.

Across opponents, a fair buyer's mean share ranges from 0.129 to 0.434 and an
aggressive buyer's from 0.018 to 0.819. So the lesson's "fairest = smallest
payoff variance across pairings" holds, and the deadlock is what produces it.

Two fair bargainers with identical parameters still split **43/57**, not
50/50. The buyer always moves first and a crossing clears at the standing
offer, so the protocol itself is not neutral.

Two limits on this answer. Style cannot be concession alone: with the
lesson's own openings, the buyer's concession never changes the result (ex01).
And the Bhattacharya et al. paper could not be retrieved to check the model
rankings the lesson attributes to it, so nothing here depends on them.
