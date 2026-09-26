<!-- generated:start -->
# 17-infrastructure-and-production / 20-shadow-canary-progressive

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/20-shadow-canary-progressive/) · upstream spec
`phases/17-infrastructure-and-production/20-shadow-canary-progressive/docs/en.md`

```bash
uv run demo practice run 20-shadow-canary-progressive --ex 1
uv run demo explain 20-shadow-canary-progressive --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/20-shadow-canary-progressive
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Inject a 25% cost regression. At which stage does the canary halt? | code | T0 | `ex01_a_25_percent_cost_regression_halts_at_1_percent_because_the_seed_not_the_traffic_decides.py` |
| 2 | Your new model has 3% accuracy gain offline but cost/request is +18%. Is it a ship? Depends o… | code | T0 | `ex02_the_gate_path_calls_18_percent_a_ship_and_the_simulator_rolls_it_back_95_percent_of_the_time.py` |
| 3 | Design a rollback that takes under 60 seconds end-to-end. List the required infrastructure. | code | T0 | `ex03_a_47_second_rollback_needs_a_30_second_gate_which_sees_only_outages.py` |
| 4 | Non-determinism shows ±7% on your eval. Set canary gates so you don't false-alarm. What multi… | code | T0 | `ex04_a_1_16x_gate_clears_7_percent_noise_but_rate_gates_need_3x_at_1_percent_traffic.py` |
| 5 | Shadow mode catches a 40% cost spike before canary. Write the alert rule that fires in shadow. | code | T0 | `ex05_the_shadow_rule_must_pair_requests_because_unpaired_cost_false_alarms_23_percent.py` |
<!-- generated:end -->

## Answers

### 1 — a 25% cost regression halts at 1%, because the seed, not the traffic, decides

**It halts at the first stage, 1%, on the cost gate alone.** The 1% stage
measures $0.0252 per request. The gate is 1.2 x $0.02 = $0.0240. The other
four gates pass.

**The halt stage comes from the noise draw, not the traffic share.**
`measure_stage(stage, ...)` never reads `stage`, so for the same seed the 1%
and 100% stages return identical metrics. A 1% canary is therefore as precise
as full traffic, which contradicts the lesson's "Metrics cadence" section.
With +-8% uniform noise, 1.25 breaches whenever the draw is above 0.96, which
is 75% of the time per stage. Over 10,000 seeded rollouts:

| outcome | rollouts |
|---|---:|
| halts at 1% | 7526 |
| halts at 10% | 1868 |
| promotes to 100% | 4 |

**The 20% gate is really a band from +11.1% to +30.4%.** On the shipped
seeds:

- a 12% regression, inside the gate, halts at 75%;
- every multiplier from 1.1197 up halts, and from 1.1887 up it halts at 1%;
- only 1.2 / 0.92 = 1.304 is caught on every possible seed, and nothing
  below 1.2 / 1.08 = 1.111 is caught on any.

The demo's "Small cost regression (10%) — within gate" is within it by
2 points of noise.

The lesson also gives two progressions. Its summary says 10% → 25% → 50% →
75% → 100%, while its body and `STAGES` start at 1%. The 1% stage is the one
that catches this regression.

### 2 — the gate path calls +18% a ship, and the simulator rolls it back 95% of the time

"3% accuracy gain" is read as 3 percentage points.

- **Gate path (the lesson's policy).** +18% is under the 1.2 cost gate, so
  the policy says ship. The reference still halts it at the 10% stage on the
  shipped seeds. At +-8% noise, 1.18 breaches 39.4% of stages and survives
  all six with probability 0.606^6 = 4.95%. It promotes in 523 of 10,000
  seeded rollouts, so "ship" means shipping about 1 time in 20.
- **Value path.** Ship when V x 0.03 >= 0.18 x $0.02. That gives
  **V >= $0.12 per correct answer**, 6x the baseline cost of a request. The
  break-even does not depend on baseline accuracy, because both sides are
  per request. Cost per correct answer rises either way: at 80% baseline
  accuracy it goes from $0.0250 to $0.0284 (+13.7%).

**The gate cannot see the gain at all.** `Regression` has five fields
(latency, cost, error, output length, thumbs-down) and none for accuracy. To
take the value path through the canary, the cost gate has to be raised past
the noise. At 1.28 (above 1.18 x 1.08 = 1.2744) the model promotes on all
10,000 schedules. The raised gate then only guarantees a catch from
1.28 / 0.92 = 1.391.

### 3 — a 47-second rollback needs a 30-second gate, which sees only outages

"End-to-end" is read as starting when a bad candidate first serves users and
ending when every request is back on the pinned old model. That means
detection is inside the budget.

| infrastructure | step | seconds |
|---|---|---:|
| gate evaluator on a 30 s sliding window | detect | 30 |
| automatic rollback policy, no human page | decide | 0 |
| flag service write | flip | 1 |
| streaming (not polling) flag push to routers | propagate | 5 |
| router drain with a 10 s timeout | drain | 10 |
| model registry with pinned digests, baseline kept warm at full capacity | pin | 1 |
| **total** | | **47** |

This was replayed on a virtual clock at 1,000 req/min, which is the lesson's
"1% = 10 req/min". The candidate ran at the reference's 1.8x latency (P99
1.608 s). The last candidate request ends at 31.6 s.

**The lesson's own cadence misses the budget.** With gates checked every
5–15 minutes, the first check comes at 300 s. The same pipeline on a
5-minute window takes 317 s, and the lesson's redeploy path takes 3 hours.

**A 30-second gate only sees outages.** At 1%, 30 s is 5 candidate requests.
Keeping false alarms under 1% at the 2% baseline error rate needs 2 errors
out of 5. That is a 40% error rate, not the lesson's 2x. Such a gate catches
a 50% outage 81% of the time. Cost, length and thumbs-down (0.15 expected
events) cannot be judged in that window. The sub-minute path is therefore for
hard failures, and the five-metric gates stay on the slow window.

The reference itself has nothing to time. Its ROLLBACK is a `print`, and the
module has no flag, registry, digest or pin.

### 4 — a 1.16x gate clears 7% noise, but rate gates need 3x at 1% traffic

**Use 1.16x for latency, cost and output length.** A live baseline is
measured too, so +-7% on both arms bounds the clean ratio by
1.07 / 0.93 = 1.1505. Clean pairs were run through the reference's
`check_gates` with the baseline patched in, 10,000 seeded pairs per gate:

| uniform gate | false alarms / 10,000 |
|---:|---:|
| 1.10 | 2304 |
| 1.14 | 100 |
| 1.16 | 0 |
| lesson's (1.2–2.0) | 0 |

**Rate gates need a multiplier set by the sample count.** For error and
thumbs-down rates, the noise that matters is counting a few events in a
window, not the 7%. These are the multipliers that keep one window's false
alarms at or under 1% (exact binomial):

| requests per window | 50 | 500 | 1,250 | 5,000 |
|---|---:|---:|---:|---:|
| thumbs-down (3%) | 3.33x | 1.67x | 1.39x | 1.19x |
| errors (2%) | 4.0x | 1.8x | 1.48x | 1.24x |

At the 50 requests per window the lesson gives for 1% traffic, its 1.5x
thumbs-down gate fires on a clean model 18.9% of the time, and its 2x error
gate 7.8%. Over six stages (5-minute windows at 1,000 req/min), the
thumbs-down gate alone halts 21.5% of clean rollouts. The reference cannot
show this, because `measure_stage` ignores the traffic share.

**The noise floor costs detection.** A gate g only guarantees a catch above
g / 0.8692: 1.335 for a 1.16 gate and 1.381 for the lesson's 1.2 cost gate.
Per stage, a 25% cost regression is caught 74.9% of the time against a
measured baseline and 77.8% against a fixed one.

### 5 — the shadow rule must pair requests, because unpaired cost false-alarms 23%

**The rule:** fire when `sum(candidate_cost) / sum(production_cost)` over the
same request ids is above 1.2, on a window of at least 50 paired requests.
The threshold is the lesson's own cost gate. Through the reference's
window-level noise, a 40% spike reads 1.288 to 1.512 and fires on all 10,000
seeded windows. A clean candidate reads at most 1.08 and never fires. The rule
fires with zero users exposed, whereas the canary halts the same spike at 1%
only after serving it.

**Pairing is what makes shadow fast.** Per-request cost was modelled as
lognormal (sigma = 1, an assumed heavy tail):

| requests | comparison | clean fires | 40% spike fires |
|---:|---|---:|---:|
| 50 | paired (shadow) | 0.0 | 1.0 |
| 50 | unpaired (canary-style) | 0.23 | 0.734 |
| 200 | unpaired | 0.089 | 0.886 |

Requiring two independent windows would cut the unpaired false-alarm rate to
0.8%. The paired rule needs neither the larger window nor the second one.

**Shadow cannot see the lesson's quality-silent case.** Users never see
shadow output, so thumbs-down, one of the five gates, does not exist there.
The demo's "Quality silent + cost creep" (cost 1.15, thumbs-down 1.45) reads
1.15 in shadow and fires on no paired window, while the canary halts it at
25%. The reference has no shadow mode at all, and `measure_stage` reports a
thumbs-down rate for any traffic.
