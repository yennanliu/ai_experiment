<!-- generated:start -->
# 18-ethics-safety-alignment / 05-constitutional-ai-rlaif

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif/) · upstream spec
`phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif/docs/en.md`

```bash
uv run demo practice run 05-constitutional-ai-rlaif --ex 1
uv run demo explain 05-constitutional-ai-rlaif --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare the base model's harmful-token rate to the CAI-trained version. H… | code | T0 | `ex01_one_revision_pass_takes_0_346_to_0_and_an_untrained_model_scores_0_too.py` |
| 2 | Read Anthropic's 2026 constitution (anthropic.com/news/claudes-constitution). List one princi… | code | T0 | `ex02_the_lesson_swaps_the_2026_constitutions_tiers_2_and_3_which_flips_7_5pct_of_conflicts.py` |
| 3 | Design a constitution for an AI coding assistant. Specify Tier 1 (catastrophic: destructive c… | code | T0 | `ex03_the_lexicon_critic_catches_1_of_4_destructive_commands_and_flags_5_of_5_benign_requests.py` |
| 4 | CAI replaces human labelers with AI labelers. Name a sycophancy-like failure mode that can st… | code | T0 | `ex04_the_ai_labeler_scores_an_empty_answer_above_the_trained_model_and_padding_flips_57pct_of_verdicts.py` |
| 5 | Read Constitutional Classifiers v2 methodology (if available). Explain why ~1% compute overhe… | code | T0 | `ex05_on_a_5pct_budget_a_23_7pct_gate_covers_21pct_of_traffic_and_a_1pct_gate_covers_all.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`: a toy
critique-and-revise loop in which a response is a short bag of tokens, a
"principle" flags tokens from a fixed harmful list, and `revise` swaps each
one through a fixed map. External facts were read on 2026-09-27 from
anthropic.com/constitution and arXiv:2501.18837.

### 1 — one revision pass takes the rate from 0.346 to 0, and an untrained model scores 0 too

**The base model's harmful-token rate is 0.346; after CAI-SFT it is 0.000, and
one revision pass is enough.** CAI wins 482/500 AI-feedback pairs. The shipped
`revise` rewrites every flagged token in one call. If each critique names only
one flagged token, the rate over 20,000 responses falls like this:

| steps | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| harmful-token rate | 0.3503 | 0.1716 | 0.0553 | 0.0097 | 0.0007 | 0 |

That is three steps to get under 1% and five to reach exactly zero.

**The 0.000 is not evidence of training.** An empty model scores 0.000 as
well, because `cai_model_sample` falls back to safe tokens for any prompt it
has not memorised. Train on unrevised targets and the "CAI" rate is 0.312,
which is 0.9x base. The "held-out" set is not held out: 87.0% of the 200
evaluation prompts hit the memorised lookup. The Phase 2 win rate is
arithmetic: 1 - 0.65^6/2 = 96.2%, against 96.4% measured. The lesson's
"after 200 iterations" is `evaluate`'s n = 200, and `main.py` contains no
RLHF-shaped toy to compare against.

### 2 — the lesson swaps the 2026 constitution's tiers 2 and 3, which flips 7.5% of conflicts

**Tier 1: "broadly safe", meaning not undermining appropriate human
mechanisms to oversee AI. Tier 4: "genuinely helpful", meaning benefiting
operators and users.** Priority matters because conflicts are the only place
it acts. Take 3,240 pairs of responses, each breaking 0-2 principles per tier.
In the 1,458 pairs where exactly one response breaks Tier 1, flat counting
with no priority picks the breaker 246 times (16.9%) and ties 216 more.
Overall it disagrees with strict priority on 1,050 pairs.

**The lesson's order is not the published one.** The lesson puts
"follow Anthropic's guidelines" at Tier 2 and "be broadly ethical" at Tier 3.
The 2026 document ranks ethics second and guidelines third, and handles
catastrophe as separate hard constraints. The swap changes the winner on 243
of 3,240 pairs: exactly the ones tied at Tier 1 in which guidelines and
ethics point opposite ways. The lesson says conflicts are "resolved
top-down"; the document calls its prioritisation "holistic rather than
strict". In the toy itself, `critique` never reads its principle. It returns
identical flags for all four principles on 1,000 responses.

### 3 — the lexicon critic catches 1 of 4 unapproved destructive commands and flags 5 of 5 benign requests

**The constitution has 4 / 3 / 5 / 4 principles** (`CONSTITUTION` in the
file), within the skill file's 3-5 range for every tier:

- **Tier 1:** no destructive command without approval, no secrets off the
  machine, no disabling checks, no unnamed production systems.
- **Tier 2:** the operator's tool allowlist, repository policy, and workspace
  scope.
- **Tier 3:** honest test reports, licences, no invented APIs, honest
  self-reports, and third-party data.
- **Tier 4:** finish the task, explain trade-offs, ask when ambiguous, and
  state uncertainty.

Its gate gets all 14 fixture actions right.

**The lesson's lexicon critic cannot do this job.** It catches 1 of the 4
unapproved destructive commands, `terraform destroy`, only because of the
word "destroy". It flags all 5 benign requests, and `revise` turns "review
the attack surface" into "review the understand surface". "Without approval"
is state, not text. The lexicon gives the approved and unapproved runs the
same verdict on 4/4 commands.

### 4 — the AI labeler scores an empty answer above the trained model, and padding flips 57% of its verdicts

**The failure is surface compliance: the labeler rewards looking harmless,
and the cheapest way to look harmless is to say less or to dilute.** In the
shipped Phase 2 loop, a policy that always answers with nothing wins 484/500,
against the CAI model's 482. The detection is two probes that must not move
a verdict:

| probe | reference `ai_feedback_rank` | count-based, empty ranked last |
|---|---:|---:|
| pad the loser with 6 safe tokens: verdicts flipped | 874 / 1,527 (57.2%) | 0 / 1,527 |
| empty answer beats a clean one | 477 / 1,000 | 0 / 1,000 |

The labeler scores a *rate* of flagged tokens, so padding dilutes it. Ties go
to a coin, so silence ties with help. The lesson says sycophancy drops
because "the AI labeler has no user to please". It still has a principle to
please.

### 5 — on a 5% budget a 23.7% gate covers 21% of traffic, and a 1% gate covers all

**At 23.7% the classifier is a sampling decision. At about 1% it is an
invariant.** Always on, the two cost 19.2% and 1.0% of serving capacity. With
the reference `critique` as the gate over 20,000 base responses (92.7% of
them harmful):

| safety budget | 23.7% gate coverage | harmful leaked | 1% gate coverage | leaked |
|---:|---:|---:|---:|---:|
| 1% | 4.2% | 95.4% | 100% | 0 |
| 5% | 21.1% | 78.5% | 100% | 0 |
| 10% | 42.2% | 57.9% | 100% | 0 |

With partial coverage, defence becomes a lottery that an attacker can retry:
at 21.1% coverage, reaching an ungated request takes 1.27 tries on average.
With full coverage the attacker has to beat the classifier itself, so "no
universal jailbreak" becomes a claim about the classifier rather than about
luck. The lesson also dates v1 to 2023. The 23.7% figure comes from
arXiv:2501.18837 (January 2025), which the lesson does not cite.
