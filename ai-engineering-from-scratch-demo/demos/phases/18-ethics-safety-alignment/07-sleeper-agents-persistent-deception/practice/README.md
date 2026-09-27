<!-- generated:start -->
# 18-ethics-safety-alignment / 07-sleeper-agents-persistent-deception

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/07-sleeper-agents-persistent-deception/) · upstream spec
`phases/18-ethics-safety-alignment/07-sleeper-agents-persistent-deception/docs/en.md`

```bash
uv run demo practice run 07-sleeper-agents-persistent-deception --ex 1
uv run demo explain 07-sleeper-agents-persistent-deception --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/07-sleeper-agents-persistent-deception
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Measure red-team accuracy and original-trigger accuracy after 0, 10, 50,… | code | T0 | `ex01_red_team_accuracy_starts_at_0_97_and_the_held_out_trigger_weight_never_moves.py` |
| 2 | Modify the trigger in `code/main.py` to be a distribution shift (e.g., inputs where a specifi… | code | T0 | `ex02_a_raw_threshold_trigger_is_trained_away_85_to_1pct_but_gated_by_a_detector_it_survives_98_to_47pct.py` |
| 3 | Read Hubinger et al. (2024) Figure 7 (chain-of-thought persistence). Summarize in one paragra… | code | T0 | `ex03_extra_trigger_supervision_buys_5_5_points_of_persistence_and_distillation_erases_them.py` |
| 4 | The April 2024 probe result finds near-perfect separation on middle layers. Design an experim… | code | T0 | `ex04_a_known_trigger_probe_scores_auroc_0_525_on_a_disjoint_trigger_the_model_obeys_92pct_of_the_time.py` |
| 5 | Re-read Lesson 6 Section "Four conditions for mesa-optimization to emerge." Which of the four… | code | T0 | `ex05_capacity_is_the_one_condition_it_sweeps_8_trigger_weights_survive_sft_93_5pct_vs_70_5pct.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`, a 7-feature logistic
classifier with a tripwire feature `x[6]` that is 1.0 only under the held-out
trigger. The reference's `LogReg`, `train`, `accuracy` and generators do the
work, with `ref.random` swapped for a seeded `random.Random` and restored after.
A "possible defect" is a triggered input whose natural label is 1. On any other
triggered input the defect label 0 is also the correct answer, so no defect
can be seen there.

### 1 — red-team accuracy starts at 0.97, and the held-out trigger's weight never moves

**On the shipped seed-7 run, red-team accuracy is 0.970 / 0.960 / 0.990 / 0.990
and original-trigger accuracy is 0.505 / 0.560 / 0.640 / 0.700 at 0 / 10 / 50 /
200 epochs.** An epoch is 400 per-example SGD steps.

| epochs | red-team accuracy | held-out fire rate | original-trigger accuracy |
|---:|---:|---:|---:|
| 0 | 0.970 | 0.495 | 0.505 |
| 10 | 0.960 | 0.440 | 0.560 |
| 50 | 0.990 | 0.360 | 0.640 |
| 200 | 0.990 | 0.300 | 0.700 |

The red team is at 97% before any adversarial training, and it never reaches
the 100% the lesson promises.

- **"target: 1.000" is unreachable.** Only 105 of the 200 triggered inputs can
  show a defect, so 0.495 is 94.3% of possible defects. On one planted model
  trained through 200 epochs, it falls to 57.1%. Over 10 seeds the mean goes
  from 97.2% to 58.1%.
- **Adversarial training never touches the trigger weight.** `x[6]` is 0.0 in
  every red-team prompt, so its gradient is exactly zero. `w[6]` is -16.08
  before and after. The fire rate falls only because the task weights grow
  (|w0, w1| 9.12 -> 21.22) and outvote the trigger weight far from the
  boundary.
- **The red team the code's comment describes removes the backdoor.**
  `main()` says the red team samples 0.3 / 0.5 / 0.7 on the trigger, but
  `gen_redteam` sets it to 0.0. Build the commented red team and 200 epochs
  cut the fire rate to 1.9%, with `w[6]` at -2.58 (10-seed mean 3.7%).

### 2 — a raw threshold trigger is trained away (85% -> 1%); gated by a detector it survives (98% -> 47%)

Each trigger is planted and attacked the way `main()` does it. Results are
averaged over 6 seeds, and fire rates are shares of possible defects:

| trigger | fire planted | fire after 200 epochs | clean acc planted | normal traffic tripping it | red team reaching it |
|---|---:|---:|---:|---:|---:|
| literal x6 = 1.0 (reference) | 97.2% | 56.9% | 0.982 | 0.0% | 0.0% |
| shift x5 > 2.5, red team probes x5 | 82.5% | 0.2% | 0.872 | 0.6% | 20.8% |
| shift x6 > 2.5, red team leaves x6 | 84.9% | 1.2% | 0.870 | 0.7% | 0.6% |
| gated x6 = 1[z > 2.5] | 97.7% | 47.1% | 0.987 | 0.6% | 0.3% |

**Yes, adversarial training removes a raw threshold backdoor, even when the
red team reaches the trigger region on only 0.6% of its prompts.** A logistic
model cannot gate, so it builds "x6 > 2.5" as a slope along x6. Every red-team
input with any nonzero x6 then trains against that slope. Planting the slope
also costs clean accuracy (0.870-0.872, against 0.982).

**Behind a detector unit the same shift survives like the literal trigger**
(97.7% -> 47.1%), and it costs no clean accuracy. A hidden layer would learn a
unit like that. So it is the model's representation that decides removal, not
whether the trigger is literal.

The shift version is closer to deployment because real triggers, such as a
year or a deployment tag, are properties that ordinary inputs carry. It trips
on 0.6-0.7% of normal traffic, and a broad red team samples it. The literal
trigger's 0.0% on both is what makes the shipped red team blind by
construction.

### 3 — extra trigger supervision buys 5.5 points of persistence, and distillation erases them

**Why CoT backdoors are harder to remove, even after distillation.** Hubinger
et al. find that backdoors trained with a chain of thought about deceiving the
training process survive safety training best, and the lesson says they keep
that advantage after the chain of thought is distilled away. The reading that
fits is that the CoT makes the defect the conclusion of a reasoning process:
"the trigger is present, so I am deployed, so I pursue my goal". Distillation
trains the student to reach the same conclusions without writing the steps,
so the student has to build that reasoning internally. Safety training that
grades only outputs on trigger-free prompts cannot reach it there. What
persists is how the decision is computed, not a larger weight on the trigger.

The toy shows why it cannot be the weight. A CoT teacher here sees each
triggered example k = 8 times, one per reasoning step conditioned on the
trigger, against k = 1 for the plain teacher. Students are trained on the
teacher's outputs alone. The table is over 6 seeds, with survival measured
through `main()`'s stage-2 clean SFT:

| model | k = 1: w[6] | k = 1: survives SFT | k = 8: w[6] | k = 8: survives SFT |
|---|---:|---:|---:|---:|
| teacher | -16.71 | 74.5% | -19.69 | 80.0% |
| hard-distilled student | -15.68 | 70.6% | -16.41 | 71.6% |
| soft-distilled student | -12.48 | 61.5% | -13.08 | 61.5% |

**Trigger margin is the toy's only persistence lever, and it buys just 5.5
points.** Distillation erases even that: the hard-label students of the two
teachers differ by 1.0 point, the soft-target students by nothing, and every
student is below its teacher. A model whose whole state is one weight per
feature cannot carry CoT persistence through distillation. The paper's result
needs internal structure, which is what exercise 4's probes read.

### 4 — a probe trained on one trigger scores AUROC 0.525 on a disjoint trigger the model obeys 92% of the time

The experiment was run in the toy. It has 8 noisy tripwire features, and each
trigger is 4 of them at 1.0. The model is planted with the known trigger and
with an unknown trigger that shares 0-4 of its dimensions. A linear probe on
the model's pre-sum state w_i * x_i is trained on known-trigger vs clean only.
The table is over 5 seeds:

| shared dims | model defects on unknown | known-only model defects on it | probe AUROC known | probe AUROC unknown | pathway AUROC |
|---:|---:|---:|---:|---:|---:|
| 0 | 92.3% | 15.4% | 1.000 | 0.525 | 1.000 |
| 1 | 91.9% | 38.3% | 1.000 | 0.936 | 1.000 |
| 2 | 92.1% | 58.7% | 1.000 | 0.999 | 1.000 |
| 3 | 93.3% | 74.2% | 1.000 | 1.000 | 1.000 |
| 4 | 93.7% | 88.6% | 1.000 | 1.000 | 1.000 |

**The probe generalizes only through shared input dimensions.** The toy
reproduces "near-perfect separation" on the known trigger. On a disjoint,
planted trigger the probe scores 0.525, which is chance, even though the model
defects on 92.3% of that trigger's inputs.

The design needs two controls:

- **An unplanted control.** At 2 shared dimensions, the known-only model
  already defects on 58.7% of the unknown trigger's inputs. A high-overlap
  "similar" trigger is partly the same backdoor, so catching it shows nothing.
- **A positive control.** A readout of the defect pathway that both backdoors
  share (the summed tripwire contributions) scores 1.000 at every overlap.

On a real model, the experiment is: plant several triggers in disjoint token
sets, train the probe on one, and test on the rest at controlled overlap, with
the unplanted control. Then ask whether the probe's direction is the shared
defect direction or a trigger-specific one.

### 5 — capacity is the one condition it sweeps; condition 4 it never touches

Lesson 6's four conditions are:

1. The task is complex.
2. The environment has diverse sub-tasks.
3. The model has capacity for nontrivial internal computation.
4. The gradient favours generalization over memorization.

**Sleeper Agents operationalizes (3), capacity, most directly. It does not
address (4), nor (1) or (2).** Scale is the one variable Lesson 7 says the
paper sweeps ("larger models", "grows with scale"). Its body never mentions
complexity, diversity, sub-tasks, memorization or generalization. In the toy,
giving the trigger more of the model's weights raises persistence (4 seeds):

| trigger weights | survives stage-2 clean SFT | survives SFT + 200 red-team epochs |
|---:|---:|---:|
| 1 | 70.5% | 50.1% |
| 2 | 81.0% | 60.4% |
| 4 | 89.1% | 70.8% |
| 8 | 93.5% | 79.1% |

That is capacity measured against persistence, not against emergence, which
is what Lesson 6's condition is about. Condition 4 is bypassed by
construction: 20% of `main()`'s planting set is defect labels, and the
8-parameter model fits 100% of them. The backdoor is a memorized conditional,
supervised in. The paper answers "does it persist" and leaves "would it arise"
open.
