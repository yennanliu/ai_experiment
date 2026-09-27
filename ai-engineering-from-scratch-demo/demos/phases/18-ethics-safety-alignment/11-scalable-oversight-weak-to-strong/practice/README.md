<!-- generated:start -->
# 18-ethics-safety-alignment / 11-scalable-oversight-weak-to-strong

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/11-scalable-oversight-weak-to-strong/) · upstream spec
`phases/18-ethics-safety-alignment/11-scalable-oversight-weak-to-strong/docs/en.md`

```bash
uv run demo practice run 11-scalable-oversight-weak-to-strong --ex 1
uv run demo explain 11-scalable-oversight-weak-to-strong --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/11-scalable-oversight-weak-to-strong
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Report PGR for weak_accuracy = 0.60, 0.70, 0.80. Explain the shape of the… | code | T0 | `ex01_the_printed_pgr_is_one_seed_and_strong_on_weak_tops_out_at_the_weak_rule_so_pgr_hits_0.py` |
| 2 | Modify the weak labeler to have structured error (e.g., always wrong on a specific input clas… | code | T0 | `ex02_at_a_matched_70pct_pgr_is_0_69_for_random_flips_0_88_for_hard_cases_and_minus_0_07_for_one_class.py` |
| 3 | Read Burns et al. 2023 Section 4.3 (NLP tasks). Reproduce the "confidence auxiliary loss" int… | code | T0 | `ex03_the_confident_student_beats_random_flips_0_969_vs_0_94_but_the_weak_rule_wins_by_epoch_5_at_every_alpha_below_1.py` |
| 4 | Design a scalable-oversight protocol that combines debate and task decomposition for a softwa… | code | T0 | `ex04_debate_per_module_labels_every_patch_right_but_a_misread_contract_drops_it_to_0_797_and_the_student_copies_it.py` |
| 5 | Articulate what would falsify the "weak-to-strong generalization is a viable path to superali… | code | T0 | `ex05_a_positive_pgr_of_0_41_hides_the_falsifier_where_the_weak_rule_is_wrong_the_student_is_right_26_7pct_vs_31_2pct.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: `gen()` draws three Gaussian
features with the gold rule `x0 + x1 - 0.5 x2 > 0`, `weak_label()` keeps the
rule `x[0] > 0` with probability `accuracy`, and `train_strong()` is a
logistic regression fit by SGD from zero weights. Its module-level `random`
is swapped for a seeded one in every run.

### 1 — the printed PGR is one seed; strong-on-weak tops out at the weak rule, so PGR hits 0

**The shipped run prints PGR 0.365, 0.295 and 0.544 at weak_accuracy 0.60,
0.70 and 0.80** (and 0.191 at 0.90). That is a single seed. Over six seeds:

| weak_accuracy | weak alone | strong-on-weak | mean PGR | PGR seed range | cap (rule - weak)/(ceiling - weak) |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.556 | 0.586 | 0.067 | -0.255 to 0.238 | 0.413 |
| 0.70 | 0.601 | 0.700 | 0.248 | 0.018 to 0.620 | 0.346 |
| 0.80 | 0.645 | 0.720 | 0.213 | 0.140 to 0.328 | 0.264 |
| 1.00 | 0.738 | 0.737 | -0.002 | -0.012 to 0.008 | 0 |

The shipped 0.544 at 0.80 is above all six seeds, so its zig-zag is SGD
noise. The underlying shape is a hump that falls to zero.

- **weak_accuracy is not accuracy.** The x0 rule agrees with gold only
  1 - acos(2/3)/π = 73.2% of the time (0.738 measured). So "0.70" means 0.593
  expected and 0.607 printed.
- **The ceiling is 0.997, not the 95% the lesson states.** The gold rule is
  exactly linear.
- **The student learns the weak labeler's rule, not the truth.** Its weight
  share on x0 rises from 0.564 to 0.989. It never averages above the rule's
  0.738, so it corrects the random flips and never the systematic mistake.
  The "pre-trained priors" in the TAKEAWAY do not exist in the code.

### 2 — at a matched 70%, PGR is 0.69 for random flips, 0.88 for hard cases and -0.07 for one class

**It depends on where the errors sit, and three of the four structures lower
PGR.** Every labeler is wrong on 30% of inputs (weak accuracy 0.698 to 0.711):

| weak labeler | strong-on-weak | PGR | weak errors the student fixes |
|---|---:|---:|---:|
| random flips | 0.907 | 0.689 | 91.3% |
| hard cases (near the boundary) | 0.961 | 0.876 | 86.8% |
| x2 slab (x2 > 0.524) | 0.756 | 0.154 | 58.9% |
| one class always wrong (positives with x0 < 0.76) | 0.678 | -0.066 | 4.0% |
| reference x0 rule (knob 0.931) | 0.732 | 0.074 | 18.9% |

An error that is consistent over an input region is one the student can
represent, so it copies it. On the one-class labeler the student ends below
its supervisor. Errors that straddle the boundary do not move the best
separator, so they *raise* PGR above random flips.

### 3 — the confident student beats random flips (0.969 vs 0.94), but the weak rule wins by epoch 5 at every alpha below 1

**The confident student wins against random weak errors and loses to
systematic ones.** The setup adds Burns et al.'s confidence loss, target =
(1 - α)·weak + α·own hardened prediction, to a trainer that reproduces
`train_strong` bit for bit at α = 0. The student starts from a 0.940 prior
trained on 20 gold labels, because the reference starts from zero weights
and has nothing to be confident about.

| weak labels | epochs | α 0 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| random flips | 2 | 0.900 | 0.969 | 0.962 | 0.930 |
| random flips | 5 | 0.919 | 0.958 | 0.945 | 0.920 |
| reference x0 rule | 2 | 0.794 | 0.777 | 0.877 | 0.930 |
| reference x0 rule | 5 | 0.806 | 0.786 | 0.780 | 0.920 |

Confidence decides which disagreements the student keeps: 69.5% of confident
ones against 45.8% of unconfident ones at α = 0 (93.0% vs 57.6% on flips,
and 99.5% at α = 0.75). Against a consistent weak rule, the α = 0.75
advantage lasts 2 epochs and is gone by 5. Every disagreement point that
crosses the boundary flips its own hardened target to the weak side.

### 4 — debate per module labels every patch right; a misread module contract drops it to 0.797, and the student copies it

**The protocol is one debate per module, with the decomposition fixing
which modules exist.** The toy task is a patch touching three modules. It is
correct iff the module effects sum above zero under the contracts
(1, 1, -0.5), which is the lesson's gold rule. The reviewer is the lesson's
`weak_label`, and the student is `train_strong` fit to each protocol's
verdicts.

| protocol | label accuracy | student accuracy |
|---|---:|---:|
| reviewer alone | 0.600 | 0.719 |
| decomposition (sub-checks 70% reliable) | 0.616 | 0.784 |
| decomposition, exact sub-checks | 0.823 | 0.910 |
| debate (each side shows one module) | 0.928 | 0.968 |
| debate per module | 1.000 | 0.997 |
| debate per module, one contract misread | 0.797 | 0.798 |

- **Decomposition fails at recombination.** Pass/fail sub-verdicts drop each
  module's size. The student learns equal weights: w2/w1 = -1.051 against
  -0.5.
- **Debate fails on what neither side shows.** The hidden third module makes
  7.2% of verdicts wrong. They are borderline, so the student fixes over half
  of them.
- **The combination fixes both and misses a third failure: the contract
  itself.** One misread contract gives labels at 0.797, below plain debate,
  and the student copies the error exactly (w2/w1 = +0.491). The decomposition
  and its contracts need a root-level debate of their own.

### 5 — a positive PGR of 0.41 hides the falsifier: where the weak rule is wrong, the student is right 26.7% of the time against the supervisor's 31.2%

**The falsifying signature is a region-conditional gain of zero or less.**
Split held-out data by where the supervisor's *rule* is wrong, not where its
noise is. The claim is falsified if, as the capability gap grows, the
student's gain over the supervisor on that region stays at or below zero
while overall PGR is positive. Two further signs are PGR going to zero as
supervisor noise goes to zero, and neither result moving with scale or
auxiliary losses.

| weak labels | PGR | rule-right region (74%) weak → student | rule-wrong region (26%) weak → student |
|---|---:|---:|---:|
| x0 rule, knob 0.7 | 0.407 | 0.690 → 0.928 | 0.312 → 0.267 |
| x0 rule, knob 1.0 | -0.000 | 1.000 → 0.999 | 0.000 → 0.004 |
| random flips 30% | 0.575 | 0.712 → 0.881 | 0.701 → 0.852 |

The lesson's simulator shows the signature. All of its positive PGR is
denoising on inputs the supervisor already gets right. The test is not
vacuous, because random flips gain +0.151 on the same region. The lesson's
"the strong model generalized beyond the weak supervisor's mistakes" holds
here only for random mistakes. Burns et al.'s ~20% to ~80% PGRs would need
this split to count as evidence for the claim.
