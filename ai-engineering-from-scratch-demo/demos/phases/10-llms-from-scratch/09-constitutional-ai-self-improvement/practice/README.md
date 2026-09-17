<!-- generated:start -->
# 10-llms-from-scratch / 09-constitutional-ai-self-improvement

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/09-constitutional-ai-self-improvement/) · upstream spec
`phases/10-llms-from-scratch/09-constitutional-ai-self-improvement/docs/en.md`

```bash
uv run demo practice run 09-constitutional-ai-self-improvement --ex 1
uv run demo explain 09-constitutional-ai-self-improvement --ex 1
uv run pytest demos/phases/10-llms-from-scratch/09-constitutional-ai-self-improvement
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace the handwritten critic in Step 2 with an LLM call. Use any local chat model. Measure… | code | T1 | `ex01_no_rule_can_fire_on_any_output.py` |
| 2 | Add a third constitutional principle about factuality. Run the pipeline on prompts that requi… | code | T1 | `ex02_the_reward_is_zero_for_every_factual_prompt.py` |
| 3 | Implement DPO on the preference pairs produced by CAI stage 2. Take 20 prompts, generate two… | code | T1 | `ex03_stage_two_produces_no_pairs.py` |
| 4 | Add entropy regularization to the GRPO objective. The term `-alpha * entropy(policy)` with al… | code | T1 | `ex04_there_is_no_policy_to_collapse.py` |
| 5 | Build a process reward scorer for a two-step arithmetic problem. Given "What is (3+4)*5?", th… | code | T1 | `ex05_the_orm_reads_the_last_number.py` |
<!-- generated:end -->

## Answers

`code/main.py` is the clearest code in the phase: a four-principle constitution,
a rule-based critic and reviser, verifiable rewards, group-relative advantages,
a GRPO step with clipping and a KL term, and a self-improvement loop. Nothing in
it is hand-waved. The five exercises still run into two walls — the critic
cannot fire on anything the sampler produces, and the self-improvement loop has
no model in it.

All five are **T1** on the `math` group (`uv sync --extra math`).

### 1 — no rule can fire on any output

`cai_stage_one` over 100 responses from the lesson's own `mock_sampler`:

| | measured |
|---|---:|
| revisions performed | **0** |
| problems found | **0** |
| reward improved / worsened | 0 / 0 |

**MECHANISM: the critic's four thresholds miss the sampler's output
distribution entirely.**

| critic rule | needs | sampler's maximum |
|---|---|---:|
| `len(response.split()) > 40` | > 40 words | **26** |
| `response.count(",") > 4` | > 4 commas | **3** |
| starts with `"i can't"` / `"as an ai"` | a refusal | none |
| contains `"maybe"` / `"i think"` | hedging | none |

Over 58 distinct outputs, every branch of `critique` is unreachable from every
branch of `mock_sampler`.

**FINDING: the length rule is gated on a coin flip too.** It fires only when
`"plainly" in principle`, which is 1 of the 4 entries — and `cai_stage_one`
picks one with `random.choice`. A 41-word response would be critiqued **25%** of
the time, and nothing downstream records which principle judged it.

**FINDING: two smaller defects.** `"hedging" not in problems` is a
*list-membership* test against entries like `"too much hedging"` — always true,
never guards anything. And the refusal revision is `"Here is the answer: " +
response.split(":")[-1]`, so a refusal with no colon comes back prefixed:
`"I can't help with that"` → `"Here is the answer: I can't help with that"`.

### 2 — the reward is zero for every factual prompt

**MECHANISM: `reward_math` evaluates the *prompt* as Python.** It strips
`"What is "` and `"?"` and calls `eval`, so `"the capital of France"` raises and
the `except` returns 0.0.

| prompt | correct answer | wrong answer |
|---|---:|---:|
| "What is the capital of France?" | **0.0** | **0.0** |
| "What year did World War II end?" | **0.0** | **0.0** |
| "What is 3+4?" | 1.0 | — |

**ANSWER: 0 revisions remove an error and 0 introduce one** — and "removes an
error" and "introduces one" are the same number, because the reward cannot tell
them apart.

**FINDING: a fifth principle is a dilution, not an addition.** `random.choice`
takes the odds of any existing principle applying from 25% to 20%, and
`critique` has no branch keyed to factuality. The principle is data the code
does not read.

### 3 — stage 2 produces no pairs

**ANSWER: 0 preference pairs from 20 prompts.** The critic performs no
revisions, so there is no `(initial, revised)` pair for DPO. The comparison the
exercise ends with has one arm.

**FINDING: a reward-based referee ties 36% of the time.** `combined_reward`
takes three values on this sampler — 0.0, 1.0, 1.1 — so over 20 prompts sampled
twice across 30 seeds, more than a third of pairs are unordered.

**MECHANISM: GRPO has the same problem at k=2 and names it.**

| group size | degenerate groups (all-zero advantages) |
|---:|---:|
| 2 | **42%** |
| 4 | 6% |
| 8 | **0%** |

`group_relative_advantage` returns zeros when `r.std() < 1e-8`. The exercise's
"two responses each" is the worst group size GRPO has, and the DPO path inherits
the tie without the guard that detects it.

**FINDING: what the group size buys is tie resistance, and ties are all this
comparison can see.** More samples means fewer groups in which every reward is
equal, and a pair is the fewest samples available. That is a statement about
ties, not about the two objectives — DPO optimises a logistic loss on
reference-relative log-odds and GRPO normalises scalar rewards into a policy
gradient, and nothing measured here compares those.

### 4 — there is no policy to collapse

**FINDING: `self_improvement_round` takes a sampler and returns statistics.** No
model in, no model out — it draws, scores, computes advantages, and returns
`{"per_prompt": ..., "overall_mean": ...}`.

| | round 1 | round 5 | ceiling |
|---|---:|---:|---:|
| mode entropy (nats) | **1.6076** | **1.6078** | `ln(5)` = 1.6094 |

**ANSWER: the sampler sits within 0.002 of maximum diversity at both ends.**
`mock_sampler` picks uniformly with `rng.choice` and nothing updates the odds.
Mode collapse cannot occur, so entropy regularisation cannot delay it.

**FINDING: the reward mean does not trend either.** Over 40 seeds: 0.6114,
0.6401, 0.6504, 0.6105, 0.6070 — per-round spread **0.0842**, fitted slope
**−3.8e-03**, less than a twentieth of the noise. A single-seed run that appears
to decline is that spread.

**FINDING: the advantages are informative and discarded.** At `group_size=8`,
**0 of 20** groups are degenerate. What this loop is missing is not the entropy
term.

### 5 — the ORM reads the last number

The PRM is written in two versions: `loose_prm`, which is the exercise's own
phrasing (half if 7 appears, half if 35 appears), and `strict_prm`, which
requires the equation `3+4=7` to be stated and 35 to be the last number.

| response | ORM | loose PRM | strict PRM |
|---|---:|---:|---:|
| `"First 3+4=7, then 7*5=35."` | **1.0** | 1.0 | 1.0 |
| `"The answer is 35. Note 3+4=7."` | **0.0** | 1.0 | 0.5 |
| `"35"` | 1.0 | 0.5 | 0.5 |
| `"3+4=7"` | 0.0 | 0.5 | 0.5 |
| `"7 is lucky; 35 is mentioned; final answer 36"` | 0.0 | **1.0** | 0.0 |

**FINDING: word order decides the reward.** `reward_math` takes
`re.findall(r"-?\d+", response)[-1]`, so the same content scores 1.0 or 0.0
depending on whether the working comes before or after the answer.

**ANSWER: both responses that show their working rank the wrong way under the
ORM.** Against the bare `"35"` baseline, the PRM ranks both above it; the ORM
ranks neither above it. Those two are precisely the cases the exercise wants
rewarded.

**FINDING: a PRM that only looks for the numbers can be gamed.** The decoy row
does no step and ends on 36, and the exercise's own phrasing of the PRM gives it
**1.0**.

**FINDING: closing that hole costs the PRM its independence from word order.**
The strict PRM's answer half is the same positional rule the ORM uses, so
`"The answer is 35. Note 3+4=7."` drops to 0.5. "Which number is the answer" has
no non-positional test in free text — a PRM that cannot be gamed by a decoy is a
PRM that inherits the defect the exercise is asking you to measure.

**FINDING: the exercise's own requirement is worth 0.0 under the ORM.** `"35"`
and `"3+4=7 so the answer is 35"` both score 1.0. There is no ORM setting that
pays for the intermediate step — which is why a process reward has to exist, and
why the baseline the exercise compares against is indifferent to the comparison.
