<!-- generated:start -->
# 18-ethics-safety-alignment / 21-fairness-criteria-group-individual-counterfactual

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual/) · upstream spec
`phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual/docs/en.md`

```bash
uv run demo practice run 21-fairness-criteria-group-individual-counterfactual --ex 1
uv run demo explain 21-fairness-criteria-group-individual-counterfactual --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Report the three group metrics on the default data. Apply the demographic… | code | T0 | `ex01_dp_reweighting_cuts_the_gap_from_0_429_to_0_012_and_narrows_equalized_odds_too_only_conditional_use_pays.py` |
| 2 | Implement the Dwork et al. 2012 individual-fairness metric using L2 on non-sensitive features… | code | T0 | `ex02_the_score_map_breaks_lipschitz_1_on_1264_of_124750_pairs_all_across_groups_and_hard_decisions_on_9051.py` |
| 3 | Read Kusner et al. 2017. Construct a simple two-feature causal DAG for resume scoring and ide… | code | T0 | `ex03_the_lessons_dag_has_a_cause_y_so_a_predictor_on_x0_and_x1_minus_0_5a_still_flips_8pct_and_0pct_once_that_edge_is_cut.py` |
| 4 | The 2024 backtracking-counterfactuals paper avoids intervention on protected attributes. Desc… | code | T0 | `ex04_flipping_a_would_approve_101_of_199_denied_group_0_applicants_while_backtracking_asks_them_for_2_4x_the_gain.py` |
| 5 | The ICLR 2024 reconciliation argues group and counterfactual fairness are facets of the same… | code | T0 | `ex05_dp_and_equalized_odds_coincide_only_if_a_does_not_cause_y_with_the_edge_in_equalized_odds_leaves_a_0_09_dp_gap.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` on its shipped seed (53),
with a seeded `random.Random` swapped into the module, so the baseline and
DP-reweighted models are the ones `main()` prints. Gaps are group 1 minus
group 0. Exercises 3-5 redraw `gen()`'s data as a structural causal model that
keeps every noise term; a check confirms it reproduces `gen()` exactly.

### 1 — DP reweighting cuts the gap from 0.429 to 0.012 and narrows equalized odds too; only conditional use pays

| gap | baseline | DP-reweighted |
|---|---:|---:|
| demographic parity | +0.429 | +0.012 |
| equalized odds, TPR | +0.326 | -0.231 |
| equalized odds, FPR | +0.331 | -0.007 |
| conditional use, PPV | +0.157 | +0.244 |
| conditional use, NPV | -0.256 | -0.401 |
| accuracy | 71.0% | 65.8% |

**The baseline reads the sensitive attribute directly**, with weight +1.35;
the reweighted model puts -0.05 on it.

**`main()`'s takeaway is half wrong: equalized odds improves.** It says the
reweighting costs "equalized odds and conditional use accuracy". The worst
equalized-odds gap narrows from 0.331 to 0.231, and the worst conditional-use
gap widens from 0.256 to 0.401. Over seeds 0-9, equalized odds narrows in 9 of
10 and conditional use widens in 9 of 10. Stripping A's weight moves the model
toward equalized odds, because x0 depends on A only through Y. The cost lands
on predictive value.

**The headline 0.429 is one draw.** Over seeds 0-9 the baseline DP gap
ranges from 0.048 to 0.648. The trainer is plain SGD at lr 0.1, and it stops
on a noisy last iterate.

### 2 — the score map breaks Lipschitz-1 on 1264 of 124,750 pairs, all across groups; hard decisions on 9051

The test is all 124,750 pairs of `main()`'s 500 test rows, with d = L2 on
(x0, x1). Dwork's outcome distance for a yes/no decision is |p - p'|.

| map | baseline | of which same-group | DP-reweighted | of which same-group |
|---|---:|---:|---:|---:|
| probability | 1264 (1.01%) | 0 | 2 | 0 |
| hard 0/1 decision | 9051 (7.26%) | 1908 | 4075 | 2075 |

**Every probability violation crosses groups, and A's weight is the whole
cause.** The sigmoid is Lipschitz in (x0, x1) with constant 0.25 · ||w|| =
0.235 (baseline) or 0.187 (reweighted). So no two people in the same group can
violate L = 1. Zeroing the 1.35 weight on A leaves 0 violations.

**A hard threshold is not Lipschitz for any useful L.** Two people on either
side of the boundary differ by 1 however close they are. Dwork's condition
needs a scored or randomized decision.

### 3 — the lesson's DAG has A cause Y, so a predictor on x0 and x1 − 0.5A still flips 8%, and 0% once that edge is cut

Read `gen()` as a resume model. A is the group and Y is "qualified" (A → Y
through base rates of 0.3 and 0.6). x0 is a work-sample score (Y → x0) and x1
is a prestige proxy (A → x1).

**The condition: the score may use only what does not descend from A.** Here
that is x0 and the residual x1 − 0.5A, never A or raw x1, and only if Y does
not descend from A. With A → Y cut, a predictor on (x0, x1 − 0.5A) flips 0.0%
of 20,000 decisions under counterfactual A, at 64.2% accuracy.

**The lesson's own DAG makes the label counterfactually unfair.** 30.1% of
people change Y when A flips, and the same predictor flips 8.2% of decisions.
The only strictly CF-fair input left is independent of Y, so the best CF-fair
score is the majority guess: 55.0% accurate, against the baseline's 69.9%.

**Demographic parity is not counterfactual fairness.** Under the lesson's DAG
the baseline flips 48.2% of decisions. The DP-reweighted model flips 9.6%,
even though its DP gap is 0.012.

### 4 — flipping A would approve 101 of 199 denied group-0 applicants, while backtracking asks them for 2.4x the gain

**Scenario: an adverse-action notice.** US Regulation B requires a lender to
state the principal reasons for a denial, and ECOA forbids basing the decision
on a protected characteristic. For the lesson's baseline, the interventional
counterfactual do(A = 1) approves 101 of the 199 denied group-0 applicants
(and do(A = 0) approves 0 of 104 denied group-1 applicants). "You would have
been approved in the other group" admits disparate treatment, and the
applicant cannot act on it. The backtracking counterfactual keeps A and finds
the smallest change to the applicant's own background that leads to approval.
x0 carries 89.7% of that change, so the reason is "a higher work-sample
score".

**Backtracking does not hide the disparity; it prices it.**

| model | median shift, denied group 0 | denied group 1 | do(A) approves, group 0 | group 1 |
|---|---:|---:|---:|---:|
| baseline | 1.25 SD | 0.53 SD | 101 / 199 | 0 / 104 |
| DP-reweighted | 0.86 SD | 0.92 SD | 0 / 170 | 10 / 181 |

Under the baseline, a denied group-0 applicant needs 2.4x the gain. That gap
is the 1.35 weight on A, which features have to make up. An auditor can read
the discrimination off the recourse gap without intervening on the protected
attribute.

### 5 — DP and equalized odds coincide only if A does not cause Y; with the edge in, equalized odds leaves a 0.09 DP gap

**Pick demographic parity and equalized odds. The assumption is that A has no
causal path into Y, and the model's inputs depend on A only through Y.** Per
group, P(Ŷ=1) = π·TPR + (1 − π)·FPR. With one base rate π, the DP gap is
π·ΔTPR + (1 − π)·ΔFPR, so equalized odds forces parity. Measured over 100,000
people with a reference-trained x0-only classifier:

| DAG | DP gap | TPR gap | FPR gap | PPV gap | CF flips |
|---|---:|---:|---:|---:|---:|
| A → Y cut | 0.002 | -0.004 | +0.005 | -0.003 | 0.0% |
| lesson's (A → Y) | 0.094 | -0.006 | +0.007 | — | 9.3% |

With the edge cut, parity, equalized odds, equal PPV and counterfactual
fairness hold together. With the lesson's edge, equalized odds holds but DP
stays at 0.094, close to the predicted Δπ·(TPR − FPR) = 0.095. Closing it
would need TPR = FPR, a classifier that ignores x0.

**Equal base rates alone are not enough, contrary to `main()`'s takeaway.**
Cut A → Y but keep `main()`'s baseline, which reads A and x1. The gaps stay at
DP 0.423, TPR 0.473, FPR 0.381 and PPV -0.152, and π·ΔTPR + (1 − π)·ΔFPR =
0.423. The model's own path from A has to be cut too.
