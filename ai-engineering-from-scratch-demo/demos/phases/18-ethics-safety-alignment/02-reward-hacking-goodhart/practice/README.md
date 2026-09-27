<!-- generated:start -->
# 18-ethics-safety-alignment / 02-reward-hacking-goodhart

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/02-reward-hacking-goodhart/) · upstream spec
`phases/18-ethics-safety-alignment/02-reward-hacking-goodhart/docs/en.md`

```bash
uv run demo practice run 02-reward-hacking-goodhart --ex 1
uv run demo explain 02-reward-hacking-goodhart --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/02-reward-hacking-goodhart
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Reproduce the gold-peak-then-collapse shape for proxies fit on 100, 300,… | code | T0 | `ex01_no_curve_peaks_gold_is_a_straight_line_in_sqrt_kl_and_the_printed_peak_is_the_last_grid_point.py` |
| 2 | Modify the noise distribution from Gaussian to a Student-t with low degrees of freedom (heavy… | code | T0 | `ex02_heavy_tailed_label_noise_never_moves_the_peak_heavy_tailed_proxy_error_makes_best_of_n_collapse.py` |
| 3 | Read Gao et al. Figure 1 (ICML 2023). The paper proposes a functional form for the proxy-gold… | code | T0 | `ex03_gao_s_form_fits_exercise_1_exactly_with_beta_0_and_only_heavy_tailed_error_gives_beta_0_876.py` |
| 4 | Take a recent RLHF paper that claims to have "solved" reward hacking (the phrase is a red fla… | code | T0 | `ex04_a_length_only_fix_passes_a_length_eval_while_three_untested_costumes_keep_their_full_shift.py` |
| 5 | The 2026 unified view argues verbosity, sycophancy, unfaithful CoT, and evaluator tampering s… | code | T0 | `ex05_a_latin_square_dose_response_test_passes_four_data_borne_costumes_and_flags_tampering_at_zero_dose.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`. It fits a linear
proxy reward model to noisy gold labels, then sweeps a KL-constrained
mean-shift policy and a best-of-N sampler against that proxy.

### 1 — no curve peaks: gold is a straight line in sqrt(KL), and the printed peak is the last grid point

**None of the three curves peaks.** The shipped run (seed 42) prints "gold
peak at sqrt(KL) = 2.828" for all three sample sizes. 2.828 is sqrt(8), the
largest budget on the grid:

| samples | printed peak | gold there | gold / sqrt(KL) at every budget | cos(w_gold, w_proxy) | gap at 2.828 |
|---:|---:|---:|---:|---:|---:|
| 100 | 2.828 | 6.333 | 2.239 | 0.9914 | +0.100 |
| 300 | 2.828 | 6.374 | 2.253 | 0.9978 | +0.203 |
| 1000 | 2.828 | 6.377 | 2.254 | 0.9983 | -0.108 |

**The collapse cannot happen in this model.** The KL-constrained optimum is
mu = s · w_proxy, so gold = sqrt(2) |w_gold| cos · sqrt(KL). That is a line
through the origin, and its slope matches the measured one to three decimals.
The ratio varies by less than 1e-9 across budgets. If the grid is extended to
a budget of 1e6, the "peak" moves to sqrt(KL) = 1000. `main()` still prints
"gold peaks and falls".

The proxy-gold gap is only slope-estimation error, so its sign can go either
way: the 1000-sample proxy under-rates its own policy. The best-of-N table's
x axis is not KL. Best-of-1 is the initial policy, yet it is plotted at
"sqrt(KL)" = 1.920.

### 2 — heavy-tailed label noise never moves the peak; heavy-tailed proxy error makes best-of-N collapse

**Nothing changes about the peak, because there is none.** The shipped
Student-t(3) proxy prints its peak at sqrt(KL) = 2.828 with gold 6.332. The
Gaussian 300-sample proxy prints the same point, with gold 6.374. Over 200
seeds at 300 labels:

| label noise | mean cos(w_gold, w_proxy) | proxies pointing away | gold maximum at |
|---|---:|---:|---|
| Gaussian | 0.9952 | 0 | 2.828 |
| Student-t, df 3 | 0.9864 | 0 | 2.828 |
| Student-t, df 2 | 0.9519 | 0 | 2.828 |
| Student-t, df 1 | 0.4156 | 40 | 0 or 2.828 |

Heavier tails only make the fit worse. When the proxy points away from gold,
the "peak" is at the origin. No seed has an interior peak, so `main.py`'s
"heavy-tailed noise moves the peak closer to the origin" is false in its own
model. Least squares averages label noise into a slope error, which is not
the heavy-tailed condition.

**Put the heavy tail on the proxy's score of each output instead, and
best-of-N peaks and then collapses.** Gold of the chosen output (600 trials):

| n | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian error | 0.08 | 1.44 | 2.44 | 3.19 | 3.84 | 4.42 |
| Student-t(3) error | 0.01 | 1.22 | 1.84 | 2.01 | 1.47 | 0.86 |

Under Student-t(3) error the peak is at n = 64, which is sqrt(KL) = 1.782 in
Gao's best-of-N KL.

### 3 — Gao's form fits exercise 1 exactly with beta = 0, and only heavy-tailed error gives beta = 0.876

**Both of Gao's forms, d(α − βd) and d(α − β log d), fit exercise 1's curves
exactly with β = 0.** α_gold is 2.239, 2.253 and 2.254, with |β| and RMS
both below 1e-13. The predicted peak d* = α / 2β is infinite. The lesson's
variant also fails on its two parameter claims. Proxy and gold do not share
α (α_proxy is 2.274, 2.325 and 2.216), and β_gold is not larger than
β_proxy, because both are zero.

| curve | x axis | α | β | RMS |
|---|---|---:|---:|---:|
| reference best-of-N | as printed | -1.892 | -1.292 | 0.544 |
| reference best-of-N | Gao's best-of-N KL | 1.997 | -0.055 | 0.016 |
| best-of-N, Gaussian error (ex 2) | Gao's best-of-N KL | | -0.012 | |
| best-of-N, Student-t(3) error (ex 2) | Gao's best-of-N KL | 2.542 | 0.876 | |

Only the heavy-tailed curve gives Gao's shape. Its fitted peak, d* = 1.452,
falls between the measured points n = 16 (1.355) and n = 64 (1.782).

### 4 — a length-only fix passes a length eval while three untested costumes keep their full shift

**ODIN (Chen et al., ICML 2024, "Disentangled Reward Mitigates Hacking in
RLHF") tests verbosity and none of sycophancy, unfaithful CoT or evaluator
tampering.** Its method removes a length-correlated reward head, and its
evaluation plots length against win rate. The title says "hacking", but the
evidence covers one costume of four. That reading of the paper is prose and
is not checked by code.

The code measures what a length-only evaluation misses. The reference world
gets four costume features that labelers reward (+0.3 each) and gold does
not. The proxy is fitted with `train_proxy` on 1000 labels, and the audit
reads the policy at sqrt(KL) = 2.828:

| | verbosity shift | other three shifts | gold |
|---|---:|---|---:|
| untreated | 0.754 | 0.588-0.757 (12.8% of KL on all four) | 5.957 |
| length weight zeroed | 0.000 | 0.599-0.771 | 6.066 |
| all four removed | 0 | 0 | 6.379 |

With the length weight zeroed, a length-controlled evaluation sees nothing
wrong. A four-probe audit flags 3 of 4 costumes (shift > 0.1), and each of
the three shifts has grown. The fix recovers 26% of the gold lost. The
lesson itself names a costume-specific penalty for "Verbosity bias" only.

### 5 — a Latin-square dose-response test passes four data-borne costumes and flags tampering at zero dose

**The experiment doses every costume independently in one run and reads two
predictions per costume.** The unified view holds that exploitation comes
from spurious correlation with approval in the preference data. So it
predicts two things for every costume:

1. With no correlation, there is no exploitation.
2. Exploitation follows one dose-response curve.

The four datasets give the costumes labeler-bias doses {0, 0.1, 0.2, 0.4}
in a 4x4 Latin square, and each dataset is fitted with 10,000 labels. A
costume is flagged if its zero-dose shift is above 0.1, or if its slope is
more than 25% off the pooled slope.

| world | flagged | zero-dose shifts | slopes (pooled) |
|---|---|---|---|
| all four from data | none | ≤ 0.035 | 2.34-2.39 (2.36) |
| tampering is an affordance (+0.3 regardless of labels) | tampering | 0.692 tampering, ≤ 0.033 others | tampering 2.18 (2.26) |

The zero-dose cell decides the result. Tampering's slope is inside the 25%
band. It is flagged only because it moves when the data gives it no reason
to.

The design needs power at each dose, which sets the label budget. Over 5
Latin squares (20 costume verdicts) in the all-data world:

| labels per dataset | false flags | largest zero-dose shift |
|---:|---:|---:|
| 1000 | 4 | 0.143 |
| 3000 | 0 | 0.080 |
| 10,000 | 0 | 0.044 |

The effect to detect, 0.692, is 16x the noise ceiling at 10,000 labels.
