<!-- generated:start -->
# 18-ethics-safety-alignment / 22-differential-privacy-for-llms

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/) · upstream spec
`phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/docs/en.md`

```bash
uv run demo practice run 22-differential-privacy-for-llms --ex 1
uv run demo explain 22-differential-privacy-for-llms --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/22-differential-privacy-for-llms
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Sweep σ in {0.5, 1.0, 2.0} and report the (ε, δ)-accuracy trade-off. Iden… | code | T0 | `ex01_utility_holds_to_sigma_2_at_epsilon_171_and_the_proxy_reaches_10_only_near_chance.py` |
| 2 | Implement a canary insertion and a log-loss test. Measure detection rate before and after DP-… | code | T0 | `ex02_a_mislabeled_canary_is_caught_every_time_without_dp_and_at_the_false_alarm_rate_at_sigma_1.py` |
| 3 | Read Nasr et al. 2025 on training-data extraction. Why does extraction success not collapse u… | code | T0 | `ex03_dp_trained_models_recover_96_5pct_of_training_labels_and_25_copies_of_a_canary_are_caught_92_5pct.py` |
| 4 | Design a deployment using PMixED (arXiv:2403.15638) that operates entirely at inference time.… | code | T0 | `ex04_a_10_expert_pmixed_answers_617_queries_at_epsilon_8_less_accurately_than_dp_sgd_answers_any_number.py` |
| 5 | Sketch the DP Reversal via LLM Feedback attack. Design a countermeasure that limits confidenc… | code | T0 | `ex05_confidences_expose_a_record_that_labels_hide_and_only_label_only_output_stops_a_typical_one.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`: DP-SGD on a
two-feature logistic regression, one record per step, 10 epochs over 500
records, clipping at C = 1, delta = 1e-5. "The shipped data" is the
seed-59 training and test sets that `main()` draws. Where epsilon is
Renyi-accounted, it counts the 10 steps each record takes part in, with
replace-one sensitivity 2·√2·C. The lesson's printed "approx-epsilon"
counts all 5000 steps instead.

### 1 — utility holds to σ = 2 at ε = 171, and the proxy reaches 10 only near chance

**Over σ ∈ {0.5, 1, 2} utility does not collapse, and the proxy epsilon is
685, 343 and 171.** The shipped run is one draw per σ, and it is not even
monotone: σ = 4 prints higher accuracy than σ = 2. The 50-seed mean is the
trade-off.

| σ | proxy ε (printed) | shipped accuracy | 50-seed mean accuracy | Renyi ε |
|---:|---:|---:|---:|---:|
| 0 | 34257.95 | 0.995 | 0.992 | — |
| 0.5 | 685.16 | 0.955 | 0.978 | 245.84 |
| 1 | 342.58 | 0.980 | 0.956 | 82.92 |
| 2 | 171.29 | 0.860 | 0.915 | 31.46 |
| 4 | 85.64 | 0.935 | 0.855 | — |
| 8 | — | — | 0.781 | — |
| 16 | 21.4 | — | 0.669 | — |

**Collapse** (mean accuracy below 0.75, halfway to a coin flip) comes first
at σ = 16, where the proxy still says ε = 21.4.

**The proxy never reaches the "epsilon in [1, 10]" of the demo's own
takeaway.** It also prints 34257.95 for σ = 0, which the takeaway calls
infinite. Proxy ε = 10 needs σ = 34.26. There, 6 of 50 runs crash with an
OverflowError in the reference `sigmoid`, and the survivors average 0.561.

**The proxy charges 5000 steps, but each record is in only 10.** `dp_sgd`
makes 5000 clip calls for 500 records. A Renyi accountant over a record's
10 steps reaches ε = 10 at σ = 5.08, with mean accuracy 0.833. So whether
"ε = 10" costs 0.16 of accuracy (0.992 to 0.833) or all of it depends on the
accountant, not on the noise.

### 2 — a mislabeled canary is caught every time without DP, and at the false-alarm rate at σ = 1

The canary is x = (2, −2) labelled 0, where the generator's rule says 1. It
replaces one shipped record. The test calls "member" when the canary's
log-loss is under the 5th percentile of losses from models trained without
it.

| | runs per arm | detected | false alarms | mean loss in / out | non-member sd |
|---|---:|---:|---:|---:|---:|
| σ = 0 | 40 | 100% | 5% | 19.95 / 20.79 | 0.05 |
| σ = 1 | 250 | 5.6% | 4.8% | 21.94 / 23.16 | 6.72 |

**After DP, detection sits within one standard error (1.4 points) of the
false-alarm rate.** Without DP, shuffle order is the only randomness, so
the in and out losses never overlap.

**The canary moves its own loss by about 1 nat out of 20:** 0.84 nats
without DP and 1.22 at σ = 1. It is never predicted, and the lowest loss
seen is 8.2 nats. DP-SGD does not shrink the canary's pull. It adds noise
that swamps a pull that was already small.

### 3 — DP-trained models recover 96.5% of training labels, and 25 copies of a canary are caught 92.5% of the time

Built as a runnable model of the argument. "Extraction" here means
recovering a record's label from its features. "MIA" means exercise 2's
test.

**Extraction does not collapse under DP because DP bounds how much one
record changes the output, not how much of the data the output reveals.**
At σ = 1 the model recovers 96.5% of training labels, and 96.9% of labels
for records it never saw. That is generalisation, and DP permits it by
design. Repetition is the other reason. DP protects one record, and k
copies get only group privacy. At the same σ = 1, a mislabeled canary
inserted 5 times is detected in 31.2% of 80 runs, and inserted 25 times in
92.5%.

**For MIA-as-evaluation, the score measures the canary as much as the
model.** With no DP at all, the same test catches an in-distribution canary
in 13.8% of runs and the mislabeled outlier in 100%. A low MIA number
bounds only the canary that was tested, and extraction succeeds here with
no membership signal at all. The lesson says canaries "under-report" for
this reason. An audit needs worst-case canaries and an extraction test.

### 4 — a 10-expert PMixED answers 617 queries at ε = 8, less accurately than DP-SGD answers any number

Design: split the shipped private set into 10 shards and train one
non-private expert per shard. Train a public model on 20 public records.
Per query, pull each expert's probability into a Renyi ball (order 4, both
directions) around the public one. Average, and return one sampled label,
never the probability. Per-query loss is the exact worst case over the
ball, charged against ε = 8 at δ = 1e-5.

| setting | expected accuracy | queries at ε = 8 |
|---|---:|---:|
| PMixED, β = 0.1 | 0.798 | 617 |
| PMixED, β = 0.5 | 0.832 | 25 |
| PMixED, β = 1 | 0.843 | 4 |
| PMixED, uniform prior, β = 0.1 | 0.604 | 958 |
| DP-SGD, σ = 6.17 | 0.858 | unlimited |
| public model alone (sampled / argmax) | 0.749 / 0.930 | unlimited, no privacy cost |

**On this task DP-SGD dominates.** PMixED has to sample instead of taking
the argmax, and each expert sees a tenth of the data.

**PMixED's threat model is a query-only adversary, with one budget per
deployment.** DP-SGD protects the released weights, and every later query
is post-processing. PMixED's experts are not private: a leaked expert
reveals a canary in its shard in 100% of 60 runs. PMixED fits data that
cannot be DP-trained, or shards that must be retrained or dropped
individually. The price is that every user's queries spend one shared
budget.

**The privacy comes from sampling, and the lesson says otherwise.** The
lesson says "aggregation adds noise". This model adds none and still has a
finite per-query ε. Returning the probability voids the guarantee: one
changed shard moves it by up to 0.006, deterministically, and that is
enough to tell the two datasets apart.

### 5 — confidences expose a record that labels hide, and only label-only output stops a typical one

Attack sketch: the attacker holds a record (x, y), queries x, and reads the
confidence in y. The score is AUC over 100 runs trained with the record
against 100 without it (0.5 = guessing). The countermeasure coarsens what
the API returns. Its cost is Brier score on the 200 shipped test records,
averaged over the σ = 1 models.

| API returns | mislabeled record, σ = 0 | same, σ = 1 | typical record, σ = 0 | Brier (σ = 1) |
|---|---:|---:|---:|---:|
| full confidence | 1.0 | 0.559 | 0.63 | 0.0278 |
| 2 decimals | 0.5 | 0.5 | 0.609 | 0.0278 |
| 1 decimal | 0.5 | 0.5 | 0.57 | 0.0281 |
| label only | 0.5 | 0.5 | 0.5 | 0.0377 |

**Confidences identify the mislabeled record perfectly, and its label hides
it completely.** Two-decimal rounding is free at four decimals of Brier and
stops that record. A typical record keeps leaking until the API is
label-only, which costs 36% on Brier.

**DP training is not the defence here.** Post-processing already covers
confidences: any function of an (ε, δ)-DP model is equally DP. "DP
reversal" can work only where ε is loose. At σ = 1 the shipped epochs and
δ give Renyi ε = 82.92, which permits an attack advantage of 1.000. The
guarantee rules nothing out, so the output format has to.
