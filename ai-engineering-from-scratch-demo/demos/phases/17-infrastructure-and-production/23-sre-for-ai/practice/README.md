<!-- generated:start -->
# 17-infrastructure-and-production / 23-sre-for-ai

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/23-sre-for-ai/) · upstream spec
`phases/17-infrastructure-and-production/23-sre-for-ai/docs/en.md`

```bash
uv run demo practice run 23-sre-for-ai --ex 1
uv run demo explain 23-sre-for-ai --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/23-sre-for-ai
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. What if the log and metric agents disagree? How does the supervisor resolve? | code | T0 | `ex01_the_supervisor_matches_strings_so_agreeing_agents_disagree_and_two_unclears_auto_approve.py` |
| 2 | Define three "safe" auto-remediation actions for your service. Justify each. | code | T0 | `ex02_restart_is_safe_only_below_n_minus_one_over_n_load_and_the_reference_auto_approves_a_limit_change.py` |
| 3 | Write a structured runbook template: sections, required fields, verification commands. | code | T0 | `ex03_the_shipped_verify_query_compares_mib_to_a_percent_and_fires_on_an_idle_gpu.py` |
| 4 | Predictive detection fires at 12 min lead. What's your policy — pager, pre-drain, or both? | code | T0 | `ex04_the_89_percent_is_recall_and_the_page_decision_turns_on_the_precision_nobody_quotes.py` |
| 5 | Argue whether a 3-person team should adopt AI SRE in 2026 or wait. Consider maturity, volume,… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a three-agent triage simulator with a
supervisor. Exercises 1-3 run it. It has no predictor, so exercise 4 is a
closed-form model built from the lesson's prose, with its assumptions stated.

### 1 — the supervisor matches strings, so agreeing agents disagree and two "unclear"s auto-approve

**It does not resolve a split. It picks the bigger number and hands it to a
human.** Groups are ranked by the sum of their confidences. Split log (0.78)
and metric (0.82), with the runbook agent siding with neither, and MetricAgent
wins alone. One supporter means `adversarial_agreement` is False, so the gate
is "human approval required". An exact 0.8 / 0.8 tie goes to whichever agent
was listed first. Evidence is never read.

**The shipped run is already a "disagreement" between agents that agree.**
All three hypotheses describe the same KV-cache OOM. The grouping key is
`root_cause.split(" on ")[0].split(" hit ")[0][:30]`, so they land in three
groups, and the supervisor reports RunbookAgent alone at 0.88. Give log and
metric the same wording and they win together, but the reported confidence
*falls* to 0.80, because agreement is averaged rather than boosted.

**Two agents agreeing on nothing auto-approve the action.** On an incident
without "checkout" the log agent says "unclear" at 0.35. A metric agent that
also says "unclear" at 0.54 outsums RB-017's 0.88, and the result is top root
cause "unclear", confidence 0.445, "safe action auto-approved". Two
hypotheses at 0.45 beat one at 0.88 the same way.

**Agreement is a 30-character prefix, and the action is a constant.** "GPU
memory utilization hit 98%" and "… hit 40% only" count as agreement. Every run
proposes "restart pod + lower --gpu-memory-utilization". The metric and runbook
agents ignore the incident text, so a DNS incident still gets RB-017 at 0.88.

### 2 — restart is safe only below (N-1)/N load, and the reference auto-approves a limit change

The service is the lesson's vLLM checkout-summary pool. Each action is a
guard, and the guard is its justification:

| action | guard | why it is safe |
|---|---|---|
| restart one pod | survivor load u·N/(N-1) ≤ 1; one pod at a time, not the same pod twice in 30 min | recreated from the same image; blast radius one pod |
| revert last deploy | landed ≤ 60 min before onset, no migration | image revert is undone by redeploying |
| scale pool | stays in [2, 8], at most ±2 per step | a wrong call costs at most 2 GPUs |

The restart bound is the one people skip. (N-1)/N is 0.5 at N=2, 0.75 at N=4
and 0.88 at N=8, and at N=1 a restart is an outage. At 80% load on 4 pods the
survivors run at 106.7%, so the restart turns one bad pod into a pool
overload.

**The reference's proposed action is two actions, and the second is on the
lesson's own "not safe" list.** Lowering `--gpu-memory-utilization` is
"modify resource limits". The gate allows the restart and denies the flag
change. RB-017's own evidence nonetheless labels the pair "safe action", and
exercise 1's supervisor auto-approves it whenever two agents share a key.
vLLM preallocates its KV cache at startup, so the change only lands with a
restart. For an 8B bf16 model on an 80 GB GPU, ignoring activations, 0.90 →
0.85 cuts the KV budget from 56 to 52 GB, 7.1%: a capacity cut shipped as a
fix.

### 3 — the shipped verify query compares MiB to a percent, and fires on an idle GPU

The template has five sections and 15 required fields:

- **meta**: id, service, owner, last_verified
- **symptom**: alert, log_pattern, query
- **hypothesis**: cause, evidence
- **verify**: command, expect, unit
- **act**: action, guard, rollback

A validator rejects a runbook with any field missing, or with a verify
command whose metric unit does not match its `expect`. RB-017 rewritten in
the template validates and survives a markdown round trip. The reference's
RB-017, which is the runbook agent's evidence, fills 3 of the 15 fields: id,
last-applied date and action. It has no symptom query, verify step or
rollback.

**The unit field is what catches the shipped query.** The metric agent's
evidence is "DCGM_FI_DEV_FB_USED >= 97% for 240s". dcgm-exporter's metric list
documents that field as "Framebuffer memory used (in MiB)" (checked
2026-09-26). Read literally, the condition compares MiB to 97, so a GPU
holding only 16 GB of weights (16384 MiB) satisfies it from the moment the
model loads. The command that means 97% is `FB_USED / (FB_USED + FB_FREE)
>= 0.97`, which reads 0.2 on that GPU.

**Structure is also what makes retrieval work.** `runbook_agent` returns
RB-017 for any input, the checkout incident and a DNS incident alike.
Matching the log agent's evidence against the template's `log_pattern`
returns RB-017 for checkout and nothing for DNS. An empty result is itself
the signal to page a human.

### 4 — the 89% is recall, and the page decision turns on the precision nobody quotes

**Both, but asymmetrically: pre-drain on every fire, within exercise 2's
guard, and page only once precision is known.** The model takes recall 0.89,
the 12-minute lead and the 30-minute investigation from the lesson. It
assumes 4 outages a month, a 5-minute acknowledgement, half the causes
pod-local, a 3-minute drain and one prediction per minute:

| policy | outage-min / month | predictive pages / month |
|---|---:|---:|
| none | 140.0 | 0 |
| pager | 97.3 | 3.56 + false alarms |
| pre-drain | 77.7 | 0 |
| both | 56.3 | 3.56 + false alarms |

**Precision decides the pager half, and the lesson never states it.** At a
per-minute false-alarm rate of 1e-3 the pager sends 43.2 false pages a month
against 3.56 true ones, precision 7.6%. At 1e-4 it sends 4.32, precision
45.2%, and at 1e-5 it sends 0.43, precision 89.2%. So the pager is added only
after the false-alarm rate has been measured at or below 1e-4.

**A 12-minute lead does not cover the lesson's own 30-minute investigation.**
A paged engineer still needs 5 + 30 minutes, so a caught outage runs 23
minutes instead of 35: the page buys exactly the lead time. With the "first
20 minutes" automated, it runs 3. Early paging pays off only alongside triage.

A web search on 2026-09-26 found no source for the lesson's "MIT 2025"
89%-at-10-15-minutes result, and the lesson links none. The closest match was
Meta's ECC-trend work, 89-96% at 48-72 hours, which is a different horizon.
The figure is used above only as a given recall.

### 5 — a 3-person team should adopt the runbooks and the guard now, and wait on the agents

*Draws on "The Problem".*

The case rests on three things.

**Volume.** The lesson's claim is that the first 20 of a 30-minute
investigation are automatable. The lesson's own skill file
(`outputs/skill-ai-sre-plan.md`) refuses a full rollout below 10 incidents a
month. At that floor, automation saves about 200 engineer-minutes a month,
roughly an hour and a half per person. That is real, but small next to the
cost of running a supervisor and three agents, and small next to the cost of
the misfires exercise 1 shows.

**Risk.** The pattern as shipped auto-approves when two agents share a
30-character prefix, including two "unclear"s. Its constant action includes a
resource-limit change that the lesson itself calls unsafe (exercise 2). On a
3-person rotation, a wrong autonomous action at 3 a.m. has no second
responder to catch it.

**Maturity.** Grounding is only as good as the data. The shipped
verification query compares MiB to a percent (exercise 3), and a team that
cannot yet write unit-checked verify commands would be feeding its agents
queries that are always true.

**What to adopt in 2026:**

- the structured-runbook template, which is weeks 1-4 of the skill file's own
  12-week plan;
- a guarded allowlist of three actions;
- read-only triage from whatever managed copilot already sits in the team's
  observability stack, where the marginal cost is lowest.

That covers the lesson's strongest argument for a small team, operational
memory. When one of three people leaves, a third of the tribal knowledge goes
with them, and structured runbooks keep it.

**What to wait on:** autonomous remediation, and a custom multi-agent
supervisor, until two things hold:

- incident volume passes the skill file's floor of 10 a month;
- the team has a measured false-agreement rate for the gate.
