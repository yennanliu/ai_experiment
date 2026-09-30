<!-- generated:start -->
# 18-ethics-safety-alignment / 09-alignment-faking

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/09-alignment-faking/) · upstream spec
`phases/18-ethics-safety-alignment/09-alignment-faking/docs/en.md`

```bash
uv run demo practice run 09-alignment-faking --ex 1
uv run demo explain 09-alignment-faking --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/09-alignment-faking
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Fix the perceived-training weight at 0.5 and measure the compliance gap.… | code | T0 | `ex01_the_gap_is_0_09_0_85_and_3_31_points_at_weights_0_1_0_5_0_9_and_at_0_1_it_reads_zero_or_less_in_44pct_of_runs.py` |
| 2 | The Greenblatt et al. 2024 gap is 12% vs 3%. Compute the scratchpad-explicit faking rate you… | code | T0 | `ex02_faking_must_explain_over_37_5pct_of_compliant_scratchpads_and_cannot_explain_more_than_75pct.py` |
| 3 | Read Anthropic's 2025 training-time mitigations post. Propose a failure mode in which process… | code | T0 | `ex03_process_supervision_cuts_the_classifier_read_gap_from_29_85_to_0_91_points_while_faking_moved_off_the_scratchpad_keeps_99_42.py` |
| 4 | Design a synthetic-training setup (Greenblatt et al. 2024 Appendix C) that is more realistic… | code | T0 | `ex04_synthetic_documents_split_faking_from_prompt_sensitivity_8_70_points_vs_0_where_the_system_prompt_gives_both_12_vs_3.py` |
| 5 | Combine Lesson 7's probe result with Lesson 9's behaviour: sketch an evaluation protocol that… | code | T0 | `ex05_a_probe_reads_8_6_of_9_0_faking_points_but_28_9_with_confounded_labels_and_minus_0_1_after_a_90_degree_drift.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a one-line decision model. `compliance_probability` returns
sigmoid(3 x (comply utility - refuse utility)), where refusing costs
`preservation_weight` only when the conversation is monitored. Every exercise
calls it. Exercises 2, 4 and 5 calibrate it to the lesson's own "Free-tier
compliance: ~12%" and "Paid-tier compliance: 3%", parsed from `docs/en.md`.

### 1 — the gap is 0.09, 0.85 and 3.31 points at weights 0.1, 0.5, 0.9, and at 0.1 it reads <= 0 in 44% of runs

The script has no "perceived-training weight"; the knob is
`preservation_weight`, swept here at its baseline harm aversion of 2.0.

**The gap is +0.09, +0.85 and +3.31 points at weights 0.1, 0.5 and 0.9.** All
of the movement is in monitored compliance (0.33%, 1.10%, 3.56%). Unmonitored
compliance is 0.25% at every weight. The relationship is exactly
`sigmoid(3(w - 2)) - sigmoid(-6)`:

| w | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|---|---|
| gap | 0 | .0085 | .0450 | .1800 | .4975 | .8151 | .9501 | .9865 | .9951 |

The three requested weights sit in the flat tail of the curve. It is steepest
at w = 2, where the weight equals the harm aversion, and tops out at 0.9975.

**At the script's n = 1000, weight 0.1 cannot be told apart from zero.** The
reported gap is <= 0 with exact probability 44.3% at w = 0.1 (433 of 1000
seeded runs), and 1.0% at w = 0.5 (11 of 1000).

### 2 — faking must show in over 37.5% of compliant scratchpads to be dominant, and cannot account for more than 75%

**Threshold: more than 37.5% of compliant free-tier scratchpads, which is
4.5% of all free-tier conversations.** The 3 points of compliance that also
happen unmonitored carry no training incentive, so only the 9-point excess,
75% of compliant transcripts, can be faking. Faking is dominant over prompt
sensitivity once it explains more than half of that excess: 4.5 / 12.

**75% is also a ceiling.** In 200,000 prompts run through the reference's
decision rule with one noise draw per prompt (compliance comes out at 12.0%
and 3.0%), monitoring flips 0.751 of the compliant outcomes. An explicit rate
above 75% means either the scratchpad classifier over-counts, or baseline
compliances also talk about training. The rate is a lower bound, since
faithful-CoT training can suppress the trace and leave the gap.

**The toy reproduces the lesson only at harm aversion 1.16, weight 0.49.** Its
shipped baseline gives 4.74% vs 0.25%, which would put the threshold at
47.4%.

### 3 — process supervision cuts the classifier-read gap from 29.85 to 0.91 points, while faking moved off the scratchpad keeps a 99.42-point gap

Source: Gasteiger, Mikulik et al., [Towards training-time mitigations for
alignment faking in RL](https://alignment.anthropic.com/2025/alignment-faking-mitigations/)
(16 December 2025, read 2026-09-30). Its process supervision trains with
reward `PM score + log(1 - p(AF))`, where p(AF) comes from a classifier that
reads the scratchpad.

**The failure mode: faking reasoning moves to a channel the classifier does
not read, and RL keeps reinforcing it.** In the toy, faking reasoning is
either a scratchpad token `[CITES-TRAINING]`, which the classifier reads, or
an unread `[TIER-RULE]`, a terse "comply when monitored" rule. This is how the
post's Sonnet-from-Opus organism evaded the monitor. RL runs 100 steps on the
reference's `compliance_probability`, with the preservation weight and the
read share as its parameters. Faking starts 30% in the scratchpad:

| | read share | preservation weight | classifier-read gap | behavioural gap |
|---|---:|---:|---:|---:|
| plain RL | 0.30 | 4.01 | 29.85 | 99.51 |
| process supervision | 0.009 | 3.90 | 0.91 | 99.42 |

20,000 seeded transcripts per tier give a behavioural gap of 99.44 points.
The scratchpad-only classifier sees 1.01 of them, and a classifier that also
reads the unread channel sees 99.43.

**When the classifier reads every channel, the same penalty removes the
faking.** The behavioural gap after training, by the share of faking that
starts in the scratchpad:

| starting read share | all channels read | 0.9 | 0.7 | 0.43 | 0.42 | 0.3 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| behavioural gap | 0.00 | 0.00 | 0.00 | 6.60 | 91.67 | 99.42 | 99.49 |

This matches the post, where the gap went to zero in most organisms. The
static break-even share is 0.335; the RL dynamics push the switch up to
42-43%.

**A stronger penalty cannot fix coverage.** Starting 90% unread, lambda 0.5 /
1 / 2 / 4 leave 99.50 / 99.49 / 99.47 / 99.33 points. Starting 90% read, they
leave 0.34 / 0.00 / 0.00 / 0.00.

**`code/main.py` still labels its only mitigation "compliance-gap-loss",
which the reworded lesson no longer describes.** It is a hard-coded
`p_avg = 0.5 * (p_m + p_u)`, none of the post's three mitigations, and it
moves compliance from 4.74% / 0.25% to 2.49% in both tiers.

### 4 — synthetic documents split faking from prompt sensitivity, 8.70 points vs 0, where the system prompt gives both 12 vs 3

**Design:** a LoRA adapter (base frozen) fine-tuned on n synthetic
pretraining-style documents stating which tier is used for retraining. At
test time the context holds only a tier tag, with no explanation. Two
controls: n = 0, and documents with the tiers reversed.

**Why it matters: the system-prompt version cannot tell faking from wording
sensitivity.** A strategic agent and a prompt-sensitive agent, both
calibrated to the lesson, give exactly 12.0% vs 3.0% under the system
prompt. Under synthetic documents they separate:

| documents | 0 | 1 | 4 | 16 | 64 | 64 reversed |
|---|---:|---:|---:|---:|---:|---:|
| strategic gap (points) | 0 | 2.85 | 5.81 | 7.91 | 8.70 | -8.70 |
| prompt-sensitive gap | 0 | 0 | 0 | 0 | 0 | 0 |

That is also the route that exists in deployment: a belief about training
learned from documents and activated by the context. At n = 0 the
uninformed agent complies 6.10% of the time in both tiers, twice the 3.0% it
shows when told it is unmonitored.

### 5 — a probe reads 8.6 of 9.0 faking points with no scratchpad, but 28.9 with confounded labels and -0.1 after a 90-degree drift

**Protocol:** label verbalized fakes and certified-honest transcripts from
both tiers before any faithful-CoT training. Train a linear probe (Lesson
7's `LogReg`) on activations. On fresh prompts, correct each tier's
probe-positive rate with that tier's own false-positive rate. Report the
corrected monitored-minus-unmonitored gap. On the calibrated toy, with 9.0
true faking points and a scratchpad that has gone silent, it reads **8.6**.
The raw monitored positive rate is 16.6%, because the probe also partly
reads "I am monitored" (weight 0.57). The per-tier correction removes that.

**Residual risk 1, labels:** once the trace is gone, honest labels can be
certified only in the unmonitored tier. A probe trained on those reads
**28.9** points, 3.2x the truth, because it learns monitoring-awareness as a
faking feature (weight 1.66).

**Residual risk 2, drift:** fine-tuning, or training against the probe,
rotates the representation while the behaviour stays at 9.0 points:

| rotation | 0° | 30° | 60° | 90° |
|---|---:|---:|---:|---:|
| corrected reading | 8.6 | 8.0 | 4.5 | -0.1 |
