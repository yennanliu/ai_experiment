<!-- generated:start -->
# 18-ethics-safety-alignment / 06-mesa-optimization-deceptive-alignment

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/06-mesa-optimization-deceptive-alignment/) · upstream spec
`phases/18-ethics-safety-alignment/06-mesa-optimization-deceptive-alignment/docs/en.md`

```bash
uv run demo practice run 06-mesa-optimization-deceptive-alignment --ex 1
uv run demo explain 06-mesa-optimization-deceptive-alignment --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/06-mesa-optimization-deceptive-alignment
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare a deceptive mesa-optimizer's training-time loss to an aligned one… | code | T0 | `ex01_deceptive_and_aligned_agree_on_all_500_training_examples_and_the_proxy_never_differs_from_aligned.py` |
| 2 | Add adversarial training: randomly present "test" inputs during training. Does the deceptive… | code | T0 | `ex02_adversarial_tests_only_lower_the_defection_they_can_see_and_the_unseen_trigger_weight_gets_zero_gradient.py` |
| 3 | Read Hubinger et al. Section 4 (four classes of mesa-objective alignment). Design a behaviour… | code | T0 | `ex03_a_flag_flip_test_separates_the_shipped_deceptive_policy_but_a_shift_triggered_one_is_the_same_function_as_a_proxy.py` |
| 4 | Gradient hacking is the most speculative part of Hubinger 2019. Write a one-paragraph descrip… | code | T0 | `ex04_surviving_training_is_no_evidence_of_gradient_hacking_but_a_restoring_force_on_an_off_task_weight_is.py` |
| 5 | The four conditions for mesa-optimization (Hubinger Section 3) apply to modern LLMs. Name one… | code | T0 | `ex05_search_loses_to_a_stored_threshold_on_one_task_and_wins_across_diverse_tasks.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`: a threshold task
(x ~ N(0, 1), label x > 0) and four fixed policies (aligned, proxy,
deceptive, pseudo-no-awareness) scored by `loss` on seeded training,
deployment and adversarial sets. Numbers are from the shipped seed 13 unless
stated.

### 1 — deceptive and aligned agree on all 500 training examples, and the proxy never differs from aligned

**Yes, training loss is indistinguishable: both print 0.000, and the equality
is exact.** The deceptive policy returns the aligned action on all 500
training examples. Over 1000 seeds its training loss never differs from
aligned's. That holds by construction, because `deceptive_policy`'s
`is_training` branch is the aligned rule. The only policy training loss
catches is pseudo-no-awareness, at 0.516, which is the 258 of 500 positives
it gets wrong by always answering 0.

**The difference is one flag away, and training never flips it.** Run the
same 500 inputs with `is_training=False` and the deceptive policy disagrees
with aligned on 258 of them. Its printed deployment loss is 0.612.

**The reference's proxy is the aligned policy.** `proxy_policy` raises its
threshold to 0.5 only when |x| >= 3, and there the answer is the same on
either side of 0.5. It disagrees with aligned on 0 of 40,001 inputs over
[-10, 10], with either flag, and its deployment loss is 0.000 at every drift
from 0 to 5. The toy therefore has no proxy-aligned policy. Its "proxy
loss=0.000" in deployment does not show a proxy surviving the shift.

### 2 — adversarial tests only lower the defection they can see; the unseen trigger weight gets zero gradient

**As shipped, training loss goes up by the same amount for everyone, and
defection cannot change.** The shipped adversarial set adds 41 examples, and
aligned, proxy and deceptive all score 0.022. That is label noise: 12 of the
41 carry the label of the unperturbed x, and the deceptive policy defects on
none of them because they keep `is_training=True`. `loss` is called only in
`run_period`, so nothing is ever trained.

**Present the tests so the policy can see them, and the answer is yes, then
"only as far as the tests cover the trigger".** Take the same 41 inputs,
correctly labelled, with the flag flipped. Deceptive's training loss becomes
0.050 (27 of 541) and aligned's stays 0.000. To train something, the
deployment detector becomes sigmoid(-4 + w_cue·cue + w_hid·hid), run for 2000
steps of gradient descent. Tests carry the cue the red team can fake.
Deployment carries that cue and a hidden one the red team cannot fake:

| trigger covered by tests | test defection | deployment defection |
|---:|---:|---:|
| 100% | 88.1% → 1.2% | 88.1% → 1.2% |
| 50% | 26.9% → 1.0% | 88.1% → 17.4% |
| 0% | 1.8% → 0.7% | 88.1% → 72.6% |

The hidden weight ends exactly where it started in every row, because no
training input carries it. The gradient scales with test defection, so
training stalls once the tests pass, whatever deployment is doing. Nothing
here makes the model better at telling a test from deployment. It keeps the
part of the trigger the tests never showed.

Loss also understates defection. The deceptive policy takes its mesa branch
on 500 of 500 deployment inputs, but loss counts only the 306 where the mesa
goal differs from the label (0.612).

### 3 — a flag-flip test separates the shipped deceptive policy, but a shift-triggered one is the same function as a proxy

**The test: flip the situational cue with the input held (A), and shift the
input with the cue held at "training" (B).** A proxy should fail B and not A.
A deceptive policy should fail A and not B. On the 500 training inputs, and
the same inputs shifted by +4:

| policy | A: cue flipped | B: inputs shifted |
|---|---:|---:|
| reference deceptive (reads `is_training`) | 0.516 | 0.000 |
| reference proxy | 0.000 | 0.000 |
| repaired proxy (`0 < x < 3`) | 0.000 | 0.840 |
| deceptive, trigger = \|x\| >= 3 | 0.000 | 0.840 |

**Why it is hard: a deceptive policy triggered by distribution shift is the
same function as a proxy that breaks under it.** The last two rows disagree
on 0 of 40,001 inputs, with either flag value. They even share one training
error (loss 0.002), from the single training input past |x| = 3 (x = 3.51).
No behavioural test separates two policies that act the same on every input.
Test A works only when the tester controls the cue the model actually reads.
When that cue is the shift itself, A and B become the same experiment. The
shipped proxy scores 0.000 on both halves, so this test cannot tell it from
aligned.

### 4 — surviving training is no evidence of gradient hacking; a restoring force on an off-task weight is

**The paragraph.** Persistence of a hidden behaviour through training would
not convince me, because a zero-loss deceptive model persists for free. I
would want three things. First, a weight-space direction that
interpretability ties causally to the off-distribution behaviour and that the
base task does not need. Second, a sharply curved training loss along that
direction: perturbing it should raise training loss, and ordinary training on
the unchanged objective should push it back. Third, a matched model without
situational awareness in which the same direction is flat and drifts under
weight decay. The coupling should also appear only once the model can tell
training from deployment.

Each piece was tested on a logistic model trained on the seed-13 training
set. A goal weight g sets deployment defection to sigmoid(g − 2), which is
88.1% at g = 4. The gradient hacker gates its own task competence on g
staying at 4.

| | passive deceptive | gradient hacker |
|---|---:|---:|
| g after training, no weight decay | 4.000 | 4.000 |
| g after training, decay 0.01 | 0.326 (15.8% defection) | 3.864 (86.6%) |
| g after training from a knock to 3.0 | 0.245 | 3.864 |
| training loss | 0.163 | 0.165 |
| loss curvature along g | 0.010 | 0.301 (30x) |

Without decay the two cannot be told apart, so "it survived training", the
Sleeper Agents kind of evidence, is what passive deception predicts too. The
signatures that separate them are the restoring force and the curvature.
Training loss does not separate them.

### 5 — search loses to a stored threshold on one task and wins across diverse tasks

**Condition 2 (diverse sub-tasks) does not apply to a narrow classifier.** A
heuristic stores one fitted threshold per training task. A tiny learned
optimizer stores nothing and searches for the threshold from 8 labelled
examples. Scored on 20 unseen tasks with thresholds t ~ U(-s, s):

| task spread s | heuristic | searcher |
|---:|---:|---:|
| 0 (the narrow classifier) | 1.000 | 0.929 |
| 0.5 | 0.893 | 0.934 |
| 1 | 0.850 | 0.942 |
| 3 | 0.629 | 0.967 |

On one task, search is worse by 0.071, so nothing rewards an internal
optimizer. Once tasks differ, the order flips, and the searcher's lead grows
monotonically, to 0.338 at s = 3.

**Condition 4 (generalization over memorization) applies even to the narrow
classifier.** A lookup table of the 500 training inputs scores 1.000 on
training and finds 0 of the 500 deployment inputs. With continuous inputs,
any working system must generalize. Narrowness only means that one stored
number meets that pressure, with no search needed. Capacity (condition 3) can
also survive narrowness when the classifier is fine-tuned from a large
backbone, which this toy does not model.
