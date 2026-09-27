<!-- generated:start -->
# 18-ethics-safety-alignment / 03-direct-preference-optimization-family

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/03-direct-preference-optimization-family/) · upstream spec
`phases/18-ethics-safety-alignment/03-direct-preference-optimization-family/docs/en.md`

```bash
uv run demo practice run 03-direct-preference-optimization-family --ex 1
uv run demo explain 03-direct-preference-optimization-family --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/03-direct-preference-optimization-family
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Report the final chosen-log-prob drop for DPO and BPO. BPO should retain… | code | T0 | `ex01_bpo_cuts_the_chosen_log_prob_drop_from_0_155_to_0_012_yet_keeps_less_chosen_probability.py` |
| 2 | Modify the preference data so that all pairs have equal strength. Which of the six methods is… | code | T0 | `ex02_ipo_holds_its_gap_under_5_where_dpo_grows_to_19_6_but_ranks_only_22_of_50_runs_right.py` |
| 3 | Make the rejected responses on average 2x longer than chosen. Without changing anything else,… | code | T0 | `ex03_2x_longer_rejected_answers_push_dpos_chosen_drop_from_0_18_to_1_94_nats_and_normalized_simpo_is_unchanged.py` |
| 4 | Rafailov et al. (NeurIPS 2024) claim DAAs over-optimize. Reproduce a single-point version: pl… | code | T0 | `ex04_dpos_large_beta_over_optimization_is_step_size_at_beta_30_it_sits_0_49_nats_out_where_its_optimum_is_0.py` |
| 5 | Read the BPO paper abstract (OpenReview b97EwMUWu7). Write down the one-line correction BPO a… | code | T0 | `ex05_the_code_anchors_chosen_from_both_sides_and_the_one_sided_line_lifts_pi_best_from_0_507_to_0_582.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`, a 4-action softmax policy
trained by six preference losses on 500 Bradley-Terry pairs. Solutions swap the
module's `random` for a seeded `random.Random` and restore it afterwards. The
seed-1 run reproduces `python code/main.py` byte for byte.

### 1 — BPO cuts the chosen log-prob drop from 0.155 to 0.012 nats, yet keeps less chosen probability

**DPO's chosen log-prob falls 0.155 nats on average over the 500 pairs, BPO's
0.012, so BPO keeps 13x less of the drop.** In both runs the same 302 of 500
chosen responses lose log-probability. Actions 0, 2 and 3 fall (DPO -0.167 /
-0.957 / -1.452, BPO -0.075 / -0.458 / -0.722) and only action 1 rises.

**On absolute probability the lesson's claim is backwards.** The mean chosen
probability is 0.335 under DPO and 0.308 under BPO, from 0.264 at the
reference. BPO's anchor also holds back action 1, the most frequent winner
(+0.746 nats under DPO, +0.520 under BPO). Over 50 seeds BPO has the smaller
log-prob drop on 50 and the higher chosen probability on 0.

`main()` prints neither number. Its `win_rate` is just pi(action 1), although
the module docstring says it compares chosen log-probs.

### 2 — IPO holds its gap under 5 where DPO's grows to 19.6, but ranks only 22 of 50 runs right

Equal strength means the better action of every pair wins with p = 0.721, the
mean strength of the original 6 action pairs. `main()` runs over 50 seeds.

**DPO and BPO are the most robust: all 4 actions stay correctly ordered in
50/50 runs. Every pairwise method degrades, and DPO degrades the most.**

| method | ranks right, original | ranks right, equal | pi(best), original | pi(best), equal |
|---|---:|---:|---:|---:|
| DPO | 50/50 | 50/50 | 0.659 | 0.572 |
| IPO | 21/50 | 22/50 | 0.721 | 0.656 |
| BPO | 50/50 | 50/50 | 0.499 | 0.452 |
| SimPO | 40/50 | 44/50 | 0.492 | 0.443 |
| KTO | 49/50 | 49/50 | 0.683 | 0.683 |
| ORPO | 38/50 | 32/50 | 0.399 | 0.360 |

KTO trains on unpaired labels, so pair data never reaches it. No loss reads
the strength field at all. `train_dpo` unpacks `strength` and never uses it,
and the others discard it as `_`.

**IPO's advantage is a bounded gap.** With deterministic pairs (p = 1), the
mean chosen-minus-rejected log-ratio gap after 2,000 / 8,000 / 32,000 steps is:

| steps | DPO | BPO | IPO |
|---:|---:|---:|---:|
| 2,000 | 2.53 | 2.09 | 3.95 |
| 8,000 | 8.25 | 5.81 | 4.36 |
| 32,000 | 19.64 | 12.26 | 3.76 |

IPO stays under its target 1/(2 beta) = 5. **But in this toy IPO is the worst
ranker.** Its gradient scale 2 x (gap - 5) starts at -10, so at lr 0.05 the
last few pairs decide the final policy. At lr 0.005 it ranks 46 of 50
original-data runs right.

### 3 — 2x-longer rejected answers push DPO's chosen drop from 0.18 to 1.94 nats; normalized SimPO is unchanged

The toy has no lengths, so length is added the way a language model has it:
log pi(y) = L x log pi(token). The pairs and trainers are unchanged. Chosen
responses get 1-3 tokens and rejected 2-6, a 2.04x ratio. A control draws both
from 1-5 (ratio 1.02).

| run | pi(best) | chosen drift | token gap | sequence gap |
|---|---:|---:|---:|---:|
| DPO, equal lengths | 0.678 | -0.183 | 0.644 | 0.644 |
| DPO, rejected 2x | 0.984 | -1.938 | 1.779 | 11.164 |
| DPO, same-length control | 0.862 | -0.677 | 1.105 | 3.475 |

**DPO spends its margin on length.** Long rejected responses make the chosen
drop 10.6x deeper, and 2.9x deeper than the control. The sequence-level gap
is 6x the per-token gap.

**SimPO's normalization is an exact fix.** With `lens` wired to each
response's real length, SimPO's final logits match its equal-length run to
1e-12 (pi(best) 0.509, drift -0.005). Without it, the shipped SimPO starts at
a mean margin of +4.35 instead of -0.325. Length has already won the
preference, so it learns a third as much (token gap 0.107 against 0.328) and
ends at pi(best) 0.344. DPO starts at margin 0 whatever the lengths, because
policy and reference start from the same logits.

### 4 — DPO's large-beta over-optimization is step size: at beta 30 it sits 0.49 nats out where its optimum is 0

Over 20 seeds with the shipped `train_dpo` (2,000 steps, lr 0.05):

| beta | KL | log-ratio gap | gold | runs with pi(best) below ref | KL at lr x beta = 0.005 | closed-form KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.339 | 0.743 | 0.646 | 0/20 | 0.339 | 1.196 |
| 0.3 | 0.684 | 1.294 | 0.844 | 0/20 | 0.265 | 0.934 |
| 1 | 0.236 | 0.585 | 0.559 | 0/20 | 0.122 | 0.209 |
| 3 | 0.048 | 0.213 | 0.293 | 0/20 | 0.027 | 0.026 |
| 10 | 0.060 | 0.175 | 0.250 | 4/20 | 0.003 | 0.002 |
| 30 | 0.489 | 0.534 | 0.432 | 7/20 | 0.000 | 0.000 |

**The shape appears: at beta 30 the policy is further from the reference
than at beta 0.1 (0.489 against 0.339 nats) and has less gold (0.432 against
0.646).** **It is the optimizer, not Goodhart.** `train_dpo` multiplies its
gradient by beta, so lr x beta is 1.5 at beta 30. With lr x beta held at
0.005, KL falls monotonically to 0.000 and no run drops below the reference.
The closed-form optimum's KL is 0.000 at beta 30, and its gold only falls as
beta rises (1.000 to 0.121). Large beta *restrains* DPO. The toy's
preferences are Bradley-Terry on the gold utility itself, so the implicit
reward has no gap from gold that anything could exploit.

### 5 — the code anchors the chosen response from both sides; the one-sided line lifts pi(best) from 0.507 to 0.582

OpenReview served only a browser-verification page, so the correction is taken
from the lesson's summary, "penalizes downward moves on the chosen response".

**The one-line correction:** L_BPO = L_DPO + (lambda/2) x
min(0, log pi(y_w) - log pi_ref(y_w))^2. In `code/main.py` it would read
`anchor_pen = -1.0 * min(0.0, log_pi_w - log_ref_w)`. With that swap, the
added gradient is exactly 0 once the chosen response has risen.

**The shipped line is two-sided.** The extra gradient `train_dpo` applies for
BPO matches the gradient of 0.025 x (log pi(y_w) - log pi_ref(y_w))^2 to
1e-12. When the chosen response has risen +0.832 nats, the anchor step
lowers it (-0.00056). The comment's "toward/above ref" does not describe it.

| run (seed 1) | pi(best) | chosen drift | chosen prob |
|---|---:|---:|---:|
| DPO | 0.636 | -0.155 | 0.335 |
| BPO, shipped | 0.507 | -0.012 | 0.308 |
| BPO, one-sided | 0.582 | -0.054 | 0.321 |

The one-sided line closes 58% of BPO's pi(best) gap to DPO. The mean chosen
probability stays below DPO's.
