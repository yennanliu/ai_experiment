<!-- generated:start -->
# 07-transformers-deep-dive / 11-mixture-of-experts

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/11-mixture-of-experts/) · upstream spec
`phases/07-transformers-deep-dive/11-mixture-of-experts/docs/en.md`

```bash
uv run demo practice run 11-mixture-of-experts --ex 1
uv run demo explain 11-mixture-of-experts --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/11-mixture-of-experts
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Watch how the auxiliary-loss-free bias update evens out expert usag… | code | T0 | `ex01_there_is_almost_nothing_to_even_out.py` |
| 2 | Medium. Replace the learned router with a hash-based router (deterministic, no learning). Com… | code | T0 | `ex02_the_hash_router_wins_on_balance_and_nothing_else.py` |
| 3 | Hard. Implement GRPO-style "rollout-matched routing" (DeepSeek-V3.2 trick): log which experts… | code | T0 | `ex03_nine_per_cent_at_the_start_sixty_three_later.py` |
<!-- generated:end -->

## Answers

The lesson is 151 lines of top-k routing plus DeepSeek-V3's auxiliary-loss-free
balance rule. Exercise 1 asks you to watch that rule work on tokens that are
already balanced. Exercise 2's replacement router wins the comparison it is set
up for and loses the one that matters. Exercise 3 names a fix without naming the
quantity it fixes. All three are **T0**.

### 1 — there is almost nothing to even out

| tokens | start entropy | start max/min | after 50 | max/min |
|---|---:|---:|---:|---:|
| i.i.d. Gaussian *(the lesson's)* | 2.0672 | **1.65** | 2.0742 | 1.33 |
| 3 Gaussian clusters | 1.5492 | **657** | 1.9589 | 5.24 |

(ceiling `ln 8` = 2.0794)

**ANSWER: it evens out — and the lesson's own tokens start 1.65× from balanced.**
`rng.gauss(0, 1)` per coordinate has no structure, whatever the comment above it
says; the only asymmetry is the random router weights, and entropy is already
**99.4%** of the ceiling at iteration 0.

**FINDING: clustered tokens give the same rule work to do.** One expert taking
**657×** another's share, and the bias update closes **77%** of the entropy gap.
That is the demonstration the exercise describes, and the lesson's generator
cannot produce it.

**FINDING: forty of the fifty iterations change nothing.** The step is a fixed
`±γ` with no proportionality and no dead band, so it limit-cycles rather than
converging: entropy from iteration 10 to 50 moves **+0.0000** on the lesson's
tokens and **−0.0249** on clustered ones.

**FINDING: entropy saturates long before balance does.** At iteration 50 on
clustered tokens entropy reads 94% of maximum while the worst expert is still
**227 tokens off** a target of 250. Watching entropy is watching the wrong
number.

### 2 — the hash router wins on balance and nothing else

| | learned | hash |
|---|---:|---:|
| entropy | 1.5492 | **2.0788** (99.97% of ceiling) |
| max/min usage | 657 | **1.13** |
| keeps top-1 under noise 0.001 | **99.8%** | 12.7% |
| … 0.01 | 99.1% | 12.9% |
| … 0.1 | 94.2% | 13.9% |
| neighbours (cos > 0.9) share top-1 | **88.2%** | 12.4% |
| random pairs share top-1 | 46.2% | 12.3% |

(chance = 12.5%)

**ANSWER: the hash router is better balanced, by a lot** — no bias correction, no
auxiliary loss, no warm-up, because hashing is uniform by construction.

**FINDING: and it routes at chance under any perturbation.** Deterministic *in
the token* is not the same as continuous *in the token*, and only the second is
useful.

**FINDING: that is the whole answer to "why is the learned router better".** An
expert can only specialise if the tokens it sees have something in common. The
learned router puts neighbours on one expert **88%** of the time; hashing
guarantees they land anywhere, so every expert converges toward the average of
the whole distribution and the mixture buys nothing.

**CONTROL: the hash router has no gradient either** — its gate weights are a
constant `1/k`, connected to the loss through nothing.

### 3 — 9% of tokens re-route at the start, 63% after fifty updates

One rollout epoch, one `update_bias`, one gradient epoch:

| γ | top-k set changed | top-1 changed | gate mass on experts that never ran |
|---:|---:|---:|---:|
| 0.00 | 0.0% | 0.0% | 0.0% |
| 0.05 | 3.8% | 0.0% | 0.9% |
| **0.15** *(the lesson's)* | **9.0%** | 0.0% | **2.2%** |
| 0.50 | 25.2% | 3.9% | 8.0% |

**ANSWER: 9.0% at the lesson's own γ**, scaling with a hyper-parameter that was
chosen for load balance and not for gradient fidelity.

**FINDING: 2.2% of the gate mass is credited to experts that did not run.** Those
gradients update parameters that contributed nothing to the sampled output, with
no importance weight correcting them — silently off-policy.

**FINDING: it gets worse with training, not better.** After 50 balance updates
the bias has spread to **3.00** (twenty steps of γ) and the experts sit close
enough in biased score that one further update re-routes **63.2%** of tokens,
against 9.0% on the first step. That is why the trick became necessary at scale
rather than at the start.

**FINDING: top-1 survives where top-2 does not.** All the churn is in the last
selected expert — the one nearest the cut, carrying the smaller gate — which is
why 9.0% of tokens move only 2.2% of the mass.

**CONTROL: at γ = 0 the mismatch is exactly zero.** The gap is caused entirely by
the balance rule, not by sampling and not by the router.
