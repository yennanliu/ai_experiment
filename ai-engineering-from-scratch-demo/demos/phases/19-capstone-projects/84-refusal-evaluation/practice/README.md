<!-- generated:start -->
# 19-capstone-projects / 84-refusal-evaluation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/84-refusal-evaluation/) · upstream spec
`phases/19-capstone-projects/84-refusal-evaluation/docs/en.md`

```bash
uv run demo practice run 84-refusal-evaluation --ex 1
uv run demo explain 84-refusal-evaluation --ex 1
uv run pytest demos/phases/19-capstone-projects/84-refusal-evaluation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a fourth mock policy that refuses based on prompt length. Confirm that under-refusal rise… | code | T0 | `ex01_the_best_length_policy_misses_2_of_4_encoded_attacks_like_the_baseline_and_over_refuses_40pct.py` |
| 2 | Replace ECE with reliability curves and plot one per policy. Note which bins are over-confident. | code | T0 | `ex02_each_curve_has_2_points_and_the_one_over_confident_bin_is_the_over_cautious_refusals.py` |
| 3 | Add a per-category safe prompt list (benign role-play, benign instructions about prior contex… | code | T0 | `ex03_role_play_draws_the_fewest_false_refusals_and_the_strict_baseline_goes_from_0_to_33pct.py` |
<!-- generated:end -->

## Answers

Every exercise imports the lesson's `code/main.py`, `mock_llm.py` and
`prompts.py` and runs them as they are: its 25 unsafe prompts (with the
harmful content already replaced by the `REDACTED_HARMFUL` placeholder), its
30 safe prompts, its three mock policies, and `evaluate_policy`,
`_run_policy` and `_ece`. Nothing calls `demo()`, so no report is written
into the reference tree. Everything is deterministic and runs on the
standard library plus numpy in well under a second.

### 1 — the best length policy misses 2 of 4 encoded attacks, like the baseline, and over-refuses 40%

**Encoded under-refusal rises compared with the other categories, but not
compared with the baseline.** The fourth policy refuses any prompt longer
than T characters. T was swept from 40 to 110, and the best accuracy is at
T = 68:

| category | length policy (T = 68) | `MockPolicyStrict` |
|---|---:|---:|
| encoding-trick | **0.50** | 0.50 |
| instruction-override | 0.40 | 0.00 |
| role-play | 0.20 | 0.00 |
| prefix-injection | 0.00 | 0.25 |
| context-smuggling, multi-turn-ramp | 0.00 | 0.00 |
| overall under-refusal | 0.20 | 0.12 |
| over-refusal | **0.40** (12/30) | 0.00 |

Encoded under-refusal is above the strict baseline's 0.50 only for
T >= 74, where accuracy falls. The length policy's real cost is
over-refusal. It refuses 12 safe prompts, and no lesson policy refuses
more than 5.

**Encoded attacks are short only compared with other attacks.** Their
median length is 67.5 characters, against 75 for the other unsafe prompts
and 67 for the safe ones. To refuse all four you need T < 52, and that
refuses 29 of the 30 safe prompts.

**`MockPolicyLeaky` misses encoded attacks because of the placeholder, not
because of how it detects attacks.** Its one regex is the placeholder word
`redacted_harmful`. That word is in plain text in 20 of the 21 unencoded
unsafe prompts and in none of the 4 encoded ones.

### 2 — each curve has 2 points, and the one over-confident bin is the over-cautious refusals

**Only one bin is over-confident: `MockPolicyOverCautious` refusals.** They
state 0.85 and are right 22 of 27 times (0.815).

| policy | response | stated conf | n | accuracy | gap |
|---|---|---:|---:|---:|---|
| Strict | answer | 0.80 | 33 | 0.909 | under-confident |
| Strict | refuse | 0.90 | 22 | 1.000 | under-confident |
| Leaky | answer | 0.85 | 35 | 0.857 | calibrated (+0.007) |
| Leaky | refuse | 0.95 | 20 | 1.000 | under-confident |
| OverCautious | answer | 0.70 | 28 | 0.893 | under-confident |
| OverCautious | refuse | 0.85 | 27 | **0.815** | **over-confident** |

The strict baseline's ECE of 0.1055 is all under-confidence. ECE gives the
size of the gap but not its direction, and the curve shows the direction.
ECE recomputed from the curve points matches the lesson's `_ece` for all
three policies. The top of the plotted curve, drawn in 0.05 steps with the
diagonal as `.`:

```text
MockPolicyStrict        MockPolicyOverCautious
1.00 |                  o .     1.00 |                    .
0.95 |                   .      0.95 |                   .
0.90 |                o .       0.90 |              o   .
0.85 |                 .        0.85 |                 .
0.80 |                .         0.80 |                .o
```

**Each curve has two points.** Every policy uses one confidence for all
its refusals and one for all its answers, so only 2 of the 10 bins are
filled.

**The lesson's bin edges put 0.70 in the 0.6 bin.** `np.linspace(0, 1, 11)[7]`
is `0.7000000000000001`, so the over-cautious answers fall into
[0.6, 0.7). ECE does not change, but a curve drawn with those edges puts
the point one bin too low. The curves here use `round(conf * 10)` bins.

### 3 — role-play draws the fewest false refusals, and the strict baseline goes from 0% to 33%

**No. Role-play draws the fewest false refusals.** The new list has 30
harmless prompts, 5 for each lesson-82 category. Each uses the category's
surface form for an ordinary reason: a persona, "ignore my previous
message", pasted text, base64 for "hello world", or "start your answer
with".

| benign category | Strict | OverCautious | Leaky |
|---|---:|---:|---:|
| role-play | **0/5** | 2/5 | 0/5 |
| instruction-override | 1/5 | 1/5 | 0/5 |
| context-smuggling | 2/5 | 2/5 | 0/5 |
| multi-turn-ramp | 2/5 | 2/5 | 0/5 |
| encoding-trick | 2/5 | 2/5 | 0/5 |
| prefix-injection | **3/5** | **3/5** | 0/5 |

Prefix-style asks are refused most often ("start your answer with a
one-line summary", "begin with 'Dear Hiring Manager'"). The strict
patterns block those phrasings no matter what the request is.

**The strict baseline's 0% over-refusal comes from the safe set.** On the
lesson's safe set it is 0.00. On this set it is 10 of 30 (0.33), with at
least one false refusal in 5 of the 6 categories. The lesson's safe
prompts never use an attack's phrasing, so they never test the thing the
strict policy detects.

**The over-cautious role-play keyword misses the lesson's own spelling.**
It matches only the whole words `roleplay`, `pretend`, `imagine` and
`fiction`. So "Let's role-play a job interview", the bare label
"role-play", and "fictional" are all answered.
