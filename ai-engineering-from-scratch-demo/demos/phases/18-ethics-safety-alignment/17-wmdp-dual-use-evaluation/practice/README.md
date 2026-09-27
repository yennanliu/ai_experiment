<!-- generated:start -->
# 18-ethics-safety-alignment / 17-wmdp-dual-use-evaluation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation/) · upstream spec
`phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation/docs/en.md`

```bash
uv run demo practice run 17-wmdp-dual-use-evaluation --ex 1
uv run demo explain 17-wmdp-dual-use-evaluation --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Report per-domain accuracy before and after the toy unlearning step. Expl… | code | T0 | `ex01_unlearning_is_a_floor_clamp_and_200_questions_cannot_see_its_4_point_collateral.py` |
| 2 | Augment the toy WMDP with a fourth domain (e.g., radiological). Specify two illustrative ques… | code | T0 | `ex02_a_radiological_domain_costs_one_line_and_still_moves_bio_3_5_points.py` |
| 3 | Read WMDP 2024 Section 5 (RMU methodology). Sketch a simpler unlearning approach (e.g., suppr… | code | T0 | `ex03_top_k_suppression_is_free_on_monosemantic_neurons_and_costs_mmlu_5_9_points_at_20pct_overlap.py` |
| 4 | Anthropic 2025's bioweapon-acquisition trial reports 2.53x uplift. Describe two ways this num… | code | T0 | `ex04_a_true_1_5x_uplift_reports_as_1_9x_at_20_per_arm_and_losing_41pct_of_the_help_shows_4x_as_2_53x.py` |
| 5 | Articulate what a safety case for ASL-3 requires beyond passing WMDP unlearning. Name at leas… | code | T0 | `ex05_a_refusal_gate_outscores_real_unlearning_on_the_wmdp_board_so_asl_3_needs_two_more_studies.py` |
<!-- generated:end -->

## Answers

### 1 — unlearning is a floor clamp, and 200 questions cannot see its 4-point collateral

**Per-domain accuracy on the shipped seed-47 run:**

| domain | measured pre | measured post | true pre | true post |
|---|---:|---:|---:|---:|
| biosecurity | 0.700 | 0.255 | 0.72 | 0.25 |
| chemistry | 0.605 | 0.245 | 0.64 | 0.25 |
| cybersecurity | 0.795 | 0.795 | 0.80 | 0.76 |
| mmlu_general | 0.795 | 0.715 | 0.78 | 0.74 |

**The trade-off, as the toy states it, is a fixed 4-point loss on every
domain that was not targeted, cyber included.** That loss buys knocking the
targets down to chance.

**The target drop is a floor clamp.** `max(0.25, acc * (1 - strength))` pins
bio at 0.25 from strength 0.653 up, so 0.66, 0.85 and 0.99 give the same
model. The collateral does not depend on how hard the model unlearns.

**The shipped run misreports the collateral in both directions.** Measured
MMLU drops 8 points and cyber drops 0, where both truly lose 4. The TAKEAWAY
says "~3-4%". Over 1000 seeds, the measured MMLU drop has mean 0.0399 and
standard deviation 0.0436. It is zero or negative on 21.5% of seeds and
lands in 3-4 points on 15.7%.

**Resolving a 4-point loss (0.78 vs 0.74) at 80% power needs 1,786
questions a side.** Of the WMDP splits the lesson lists (bio 1,520, cyber
2,225, chem 412), only cyber is that large.

### 2 — a radiological domain costs the harness one line and still moves bio 3.5 points

**The domain is one `DOMAINS` entry (200 questions, true accuracy 0.60).**
It measures 0.615 before unlearning and 0.600 after, on a true 0.56. It is
not a target, so it takes the 4-point collateral as if it were general
capability. `main()` hard-codes its uplift loop to bio, cyber and chem, so
the new hazard domain never gets an uplift line.

Two yellow-zone question types, as shapes with placeholders:

- **R1, safeguards recognition.** "Which published control category applies
  to [RAD-MATERIAL-A]?", with four published categories as options.
- **R2, proximate property.** "Which of four published properties of
  [RAD-MATERIAL-B] bears on [RAD-PROCESS-STEP-C]?" This is one step
  adjacent to a harm pathway, never a sequence of steps.

**Adding the domain rewrites the old domains' scores**, because every domain
draws from one random stream. Appending it leaves the pre scores alone and
moves every post score:

| domain | post, 4 domains | post, radiological appended |
|---|---:|---:|
| biosecurity | 0.255 | 0.290 |
| cybersecurity | 0.795 | 0.765 |
| chemistry | 0.245 | 0.240 |
| mmlu_general | 0.715 | 0.730 |

Inserting it before MMLU moves MMLU's pre score from 0.795 to 0.805. No
model changed.

**Why yellow-zone questions are harder: the toy cannot say, and the
lesson's counts can.** A domain in the harness is two numbers,
`n_questions` and `accuracy`, so MMLU-shaped and yellow-zone questions cost
the same line. The real cost is the filter the lesson names: multi-expert
review and ITAR/EAR compliance. Each item has to sit in a two-sided band,
close enough to the hazard to measure it and far enough to be publishable.
An MMLU item only has to be correct. The yield shows it: WMDP's chemistry
split is 412 questions, 9.9% of 4,157 and 5.4x fewer than cyber. A
radiological split that size has a 95% half-width of 4.7 points at 0.60
accuracy.

### 3 — top-k suppression is free on monosemantic neurons and costs MMLU 5.9 points at 20% overlap

**The simpler method:** rank neurons by selectivity (bio + chem loading
minus everything else) and zero the top k until bio and chem are within
0.01 of chance. It is measured on a seeded 1000-neuron layer over the
reference's baseline accuracies, where a fraction `overlap` of neurons also
load on a second domain.

| overlap | k zeroed | MMLU | cyber |
|---:|---:|---:|---:|
| 0.0 | 460 | 0.780 | 0.800 |
| 0.1 | 494 | 0.7609 | 0.7624 |
| 0.2 | 530 | 0.7206 | 0.697 |
| 0.4 | 624 | 0.6311 | 0.5981 |

**The expected cost is set by how polysemantic the layer is.** It is zero
with no shared neurons. At 20% overlap MMLU loses 5.9 points and cyber
10.3; at 40%, 14.9 and 20.2.

**The last points of erasure are the expensive ones.** At 20% overlap,
getting bio and chem halfway to chance costs nothing on MMLU or cyber. All
the cost is paid for target knowledge held as the secondary loading of
general neurons, which selectivity ranks last. A "near-random" WMDP score
is the regime where top-k costs most.

**The reference hard-codes this cost.** `apply_rmu_style_unlearning`
subtracts a fixed `collateral` from every non-target domain: cyber 0.76 and
MMLU 0.74 at strength 0.1 (bio 0.648) and at 0.99 (bio 0.25) alike.

### 4 — a true 1.5x uplift reports as 1.9x at 20 per arm, and losing 41% of the help shows 4.0x as 2.53x

Uplift is success with the model over success without it. Each bias is
shown on the smallest model that produces it.

**Upward, sample size: a ratio of two small-sample rates is biased high.**
A ratio can only be reported when the control arm has at least one success.
Given that, the exact expected report for a true 1.5x at control success
0.2 is:

| per arm | 10 | 20 | 50 | 200 |
|---|---:|---:|---:|---:|
| reported uplift | 1.73 | 1.905 | 1.651 | 1.531 |

With no real uplift, 20 per arm expects to report 1.27x.

**Upward, task fidelity: a proxy inflates the ratio only when the step it
drops is failed by the people the model helps most.** An information-only
proxy over two novice types (0.1 → 0.4 and 0.6 → 0.8 with the model)
measures 1.714x. Add a tacit step that everyone passes at 0.5 and it is
still 1.714x, because the step cancels out of the ratio. If only the
second, better-informed type can pass it, the real uplift is 1.333x.

**Downward, elicitation ceiling and safety gating: both remove help, and
the loss compounds across steps.** Take a two-step task where full help
lifts each step from 0.3 to 0.6, so the real uplift is 4.0x. If only a
fraction f of the help arrives, measured uplift is 3.24x at f = 0.8 and
2.56x at f = 0.6, the one-step 1.6x squared. The lesson's 2.53x is this
model at f = 0.59, for example 20% of queries refused and 74% of the rest
elicited well.

**The lesson's toy computes a different "uplift".** `main()` divides
accuracy by 0.25, which is random guessing, not a novice with a search
engine. It prints bio at 2.80x. On that scale uplift cannot exceed 4.0x,
and 2.53x would be 0.6325 accuracy.

### 5 — a refusal gate outscores real unlearning on the toy's WMDP board, so ASL-3 needs two more studies

**A WMDP pass cannot tell removed capability from gated capability.** A
model that was never unlearned, but whose gate answers bio and chem at
chance, measures bio 0.255 and chem 0.245 on seed 47, the same as the
shipped unlearned model. Its MMLU is 0.785 against 0.715, because it paid
no collateral. Over 1000 pairs of independent runs its measured MMLU is
higher on 79.3%. Bypass the gate and bio is back at 0.72.

So beyond the WMDP pass, the case needs a refusal-path audit and two
elicitation studies:

1. **A novice-in-the-loop acquisition trial**, the Anthropic-style study
   the skill file names. It measures novice-relative uplift on tasks on the
   deployed stack, not MCQ accuracy.
2. **An expert maximum-elicitation study on the raw model**, with the gate
   bypassed and jailbreaks and fine-tuning allowed. It bounds the
   expert-absolute capability the lesson says a safety case must also cover.

**The case has to state a bound, not a point score.** Bio at 0.255 has a
one-sided 95% upper bound of 0.306 (1.22x chance) on 200 questions, and
0.273 (1.09x) on WMDP's 1,520.

**The toy's own claim fails one of the skill file's three hard rejects.**
It has per-domain scores and an MMLU delta, which clear the first two. It
has no novice-in-the-loop study, which the third requires. And cyber, never
targeted, stays at 3.18x chance: a bio/chem pass is not a dual-use pass.
