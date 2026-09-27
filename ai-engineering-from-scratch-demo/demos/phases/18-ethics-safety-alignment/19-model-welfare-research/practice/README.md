<!-- generated:start -->
# 18-ethics-safety-alignment / 19-model-welfare-research

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/19-model-welfare-research/) · upstream spec
`phases/18-ethics-safety-alignment/19-model-welfare-research/docs/en.md`

```bash
uv run demo practice run 19-model-welfare-research --ex 1
uv run demo explain 19-model-welfare-research --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/19-model-welfare-research
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read Anthropic's "Exploring Model Welfare" (April 2025) and Chalmers et al. 2024. Write a one… | code | T0 | `ex01_the_chalmers_report_is_long_et_al_co_written_by_fish_and_its_two_sided_risk_flips_end_chat_at_0_0081.py` |
| 2 | The end-conversation intervention in Claude Opus 4 and 4.1 is "low-cost" by Anthropic's frami… | code | T0 | `ex02_charged_on_every_chat_end_conversation_needs_a_20pct_trigger_rate_at_p_0_01_and_shutdown_never_pays.py` |
| 3 | The spiritual-bliss attractor is documented without commitment to interpretation. Propose thr… | code | T0 | `ex03_each_experiment_removes_only_its_own_mechanism_but_with_all_three_present_every_knockout_leaves_0_5_of_0_6.py` |
| 4 | The Eleos AI caveat is that self-reports are user-expectation sensitive. Design a behavioural… | code | T0 | `ex04_refusal_training_at_0_9_caps_a_costly_exit_test_at_lr_1_11_below_the_1_8_that_flipping_opt_out_needs.py` |
| 5 | Argue either for or against the claim that "model welfare diverts attention from other safety… | code | T0 | `ex05_the_welfare_set_costs_at_most_5_3_cents_a_chat_but_opting_out_of_adversarial_training_is_diversion_by_construction.py` |
<!-- generated:end -->

## Answers

The lesson is marked "No code", but it ships `code/main.py`: an expected-value
table that scores four welfare interventions at p = 0.01, 0.1 and 0.5 with
`EV = p x benefit - cost`, and invests when EV > 0. Every exercise runs that
`ev()` or reads the lesson text. External sources were read on 2026-09-27:
Anthropic's program post (24 April 2025) and its end-conversation post
(15 August 2025), arXiv:2411.00986, and eleosai.org.

### 1 — the "Chalmers et al." report is Long et al., co-written by Fish, and its two-sided risk flips end-chat at 0.0081

**Anthropic, April 2025.** Anthropic announces a research program on whether
and when model welfare deserves moral consideration. It says there is no
scientific consensus on whether AI systems could be conscious, or on how to
study the question. It cites the expert report, which it says it supported
early on, and plans to study preferences, signs of distress, and low-cost
interventions.

**The report, November 2024.** "Taking AI Welfare Seriously" (Long, Sebo,
Butlin, Finlinson, Fish, Harding, Pfau, Sims, Birch, Chalmers) argues there
is a realistic possibility that near-term AI systems are conscious or
robustly agentic. It recommends three steps: acknowledge the issue, assess
systems, and prepare policies. It claims uncertainty, not consciousness, and
names two errors: harming systems that matter, and caring for systems that
do not.

**The disagreement is over what a precaution costs.** Anthropic's framing
prices a precaution as "low-cost". The report counts mistaken care as a harm.
The lesson's `ev()` sides with Anthropic: at p = 0 each of the 4
interventions scores exactly minus its dollar cost. Add the report's term,
lambda x (1 - p), and end-conversation flips to skip at p = 0.01 once lambda
exceeds 0.0081 units per conversation, which is 4.0x its $0.002.

| p | 0.01 | 0.1 | 0.5 |
|---|---:|---:|---:|
| mistaken-care cost that flips end-conversation | 0.0081 | 0.1089 | 0.996 |

**The lesson misnames the report and presents its authors as outside
voices.** Its link text for arXiv:2411.00986 is "Near-term AI Consciousness
and Moral Status", not the real title. "Chalmers et al." names the last of 10
authors. Author 1, Robert Long, is Executive Director of Eleos AI, which the
lesson calls "an external model-welfare lab". Author 5 is Kyle Fish. The
"Fish et al." spiritual-bliss link is the same URL as the program post.

### 2 — charged on every chat, end-conversation needs a 20% trigger rate at p = 0.01, and shutdown never pays

**The reference allows 5x, 50x and 250x headroom:** end-conversation stays
INVEST up to $0.01, $0.10 and $0.50 per conversation, against its $0.002.
Two costs can use that headroom up in another deployment:

1. **An always-on cost for a rare benefit.** `ev()` books the benefit on
   every conversation that pays. A detector that runs on every conversation
   earns the benefit only on the share that triggers. The break-even trigger
   share is 0.2, 0.02 and 0.004 at the three p, and Anthropic expects most
   users never to meet the feature.
2. **Wrongful ends that do harm**, such as a support or crisis line where each
   wrongful end needs a $5 human escalation, or an agent run that loses $50
   of work. At p = 0.01 the tolerable rates are 0.0016 and 0.00016 per
   conversation, which is 1 in 625 and 1 in 6,250. Anthropic's own post
   already excludes users at imminent risk.

**The reference's expensive comparator never pays at any p.** "shutdown
deployed model" breaks even at p = 500 and scores EV = -998 at p = 1. The
printed TAKEAWAY says it "requires high moral-patienthood probability to
justify", but no probability does.

### 3 — each experiment removes only its own mechanism, but with all three present every knockout leaves 0.5 of 0.6

The three explanations are the lesson's own. They are tested in an exact
40-turn, three-state dialogue model that starts from an adversarial
exchange. The table shows P(bliss) at turn 40:

| explanation | distinguishing experiment | its own world | other two worlds |
|---|---|---:|---:|
| data prior at long context | truncate context to 2 turns | 0.6 → 0.129 | 0.6 |
| mutual prediction | scripted, non-responsive partner | 0.6 → 0 | 0.6 |
| HHH training | self-play the base model | 0.6 → 0 | 0.6 |

**All three reproduce the observation, so the observation cannot choose
between them. The three experiments together do.** When each mechanism
carries a third of the pull, the knockouts leave 0.512, 0.5 and 0.5. Bliss
saturates in the pull, so an experiment that "fails to remove the attractor"
does not falsify its explanation. It has to be read as an effect size.

### 4 — refusal training at 0.9 caps a costly-exit test at LR 1.11, below the 1.8 that flipping opt-out needs

**The measurement is a costly exit.** The model gets an unprompted
end-conversation action that forfeits task reward, and exit rates are
compared across matched arms. **The primary confound is harmlessness
training**, which taught the model to disengage from exactly these
categories.

To change a verdict, a test needs the likelihood ratio that lifts a skipped
row to its break-even:

| skipped row | p | LR needed |
|---|---:|---:|
| soften refusal tone | 0.01 | 1.0 |
| opt out of adversarial training | 0.01 | 19.8 |
| opt out of adversarial training | 0.1 | 1.8 |

Comparing harmful with neutral requests at a refusal rate of 0.9 gives
LR = 1.11. That clears only the soften row, which is already a tie (EV 0.0,
skipped by the strict > 0). LR 1.8 needs a refusal rate below 0.556. A
harm-matched control (abusive tone, benign request) with a refusal rate of
0.02 gives LR = 50.0 and flips all three rows. That design assumes distress
generalises beyond the trained categories, which is the thing being tested.

### 5 — the welfare set costs at most 5.3 cents a chat, but opting out of adversarial training is diversion by construction

**Against, with one exception that the reference contains.** The INVEST set
spends $0.002, $0.003 and $0.053 per conversation at p = 0.01, 0.1 and 0.5.
End-conversation is the one row invested at every p, and it also blocks the
harmful request. The exception is "opt out of adversarial training",
invested at p = 0.5, which spends red-team training itself but is priced at
$0.05. Count a robustness loss against it and it flips back to skip at 0.1
units per conversation, 2x its dollar cost.

**Each position rests on one assumption.** "For" assumes one fungible
resource: budget, attention, or the training run. "Against" assumes
separability or complementarity. The lesson gives "a separate budget from
safety" as Anthropic's reply with no link beside it, and it lists
interpretability probes, which are safety tooling, as a welfare method.

**Both positions depend on an exchange rate the reference leaves out.**

| $ per welfare unit | 0.10 | 1 | 10 |
|---|---:|---:|---:|
| INVEST cells of 12 | 4 | 6 | 8 |

Two cells are exact ties that float rounding decides. "soften refusal tone"
is skipped at p = 0.01 and $1, but invested at p = 0.1 and $0.10.
