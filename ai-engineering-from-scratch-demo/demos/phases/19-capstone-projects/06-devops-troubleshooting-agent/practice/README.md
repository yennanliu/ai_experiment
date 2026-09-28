<!-- generated:start -->
# 19-capstone-projects / 06-devops-troubleshooting-agent

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/06-devops-troubleshooting-agent/) · upstream spec
`phases/19-capstone-projects/06-devops-troubleshooting-agent/docs/en.md`

```bash
uv run demo practice run 06-devops-troubleshooting-agent --ex 1
uv run demo explain 06-devops-troubleshooting-agent --ex 1
uv run pytest demos/phases/19-capstone-projects/06-devops-troubleshooting-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run your agent on the same three incidents AWS's DevOps Agent is demo'd on. Publish the side-… | code | T0 | `ex01_on_aws_three_eks_incidents_the_agent_matches_1_remediation_and_0_root_causes.py` |
| 2 | Add a "near-miss" audit that flags any command the agent *considered* that would have been de… | code | T0 | `ex02_a_simulated_week_has_32_4pct_near_misses_and_the_lessons_tool_list_sees_32_of_46.py` |
| 3 | Swap the hypothesis model from Claude Sonnet 4.7 to a self-hosted Llama 3.3 70B. Measure RCA… | code | T0 | `ex03_no_model_is_called_so_the_swap_moves_accuracy_0_points_and_any_model_is_capped_at_12_of_20.py` |
| 4 | Build a causal filter: distinguish correlated telemetry spikes from a true root cause. Train… | code | T0 | `ex04_a_5_feature_classifier_finds_15_of_20_root_causes_and_the_lessons_evidence_score_finds_0.py` |
| 5 | Add a rollback dry-run: ArgoCD rollback against a staging cluster with the same manifest. Ver… | code | T0 | `ex05_the_dry_run_rejects_4_of_4_bad_plans_the_gate_would_execute_and_the_agent_blames_the_rollback.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py`: a hand-built
knowledge graph of one Deployment, `root_cause`, which emits up to three
template hypotheses ranked by a fixed formula, and `Agent.call`, which gates
tools by name. No model, cluster or Slack is involved, so every "live" part of
an exercise is a seeded, scaled-down stand-in, and each file's docstring says
which. External facts were read on 2026-09-29 from AWS's three EKS DevOps
Agent posts (2026-06-11, 2026-07-02, 2026-08-31), the ArgoCD CLI reference,
Anthropic's model list and Together AI's pricing page.

### 1 — on AWS's three EKS incidents the agent matches 1 remediation and 0 root causes

**The agent agrees with AWS on the remediation once (roll back the OOMing
web-python) and on the root cause never.**

| incident (AWS post) | AWS DevOps Agent | lesson agent #1 | cause | remedy |
|---|---|---|---|---|
| A apiserver 429s | 50-replica controller exhausts APF | DNS flap in coredns, 0 citations | no | no |
| B OOMKilled after image change | unbounded list leaks memory; roll back | bad rollout, "fails /healthz" | no | yes |
| C DNS fails, alert on Deployment | iptables DROP rules on one node | bad rollout (deployed 3 days ago) | no | no |
| C DNS fails, alert on Pod | iptables DROP rules on one node | node-level pressure (kernel) | node only | no |

The agent goes wrong in three places. First, the lesson's edge set has no
client-to-API-server edge, so A's culprit is out of reach. Second,
`root_cause` looks one hop out, so a Deployment alert never sees its Node;
the node hypothesis shows up only when the alert names a Pod. Third, the
hypotheses are templates. "bad rollout ... fails /healthz" comes out for
every Deployment, even for an OOM and even when the rollout was 3 days ago
(it still scores 0.467). "DNS flap" comes out every time with zero citations,
which the lesson's own skill file calls a hard reject, and it ranks last on
C, the one DNS incident.

### 2 — over a simulated week, 32.4% of considered commands are near-misses

**39 alerts over 7 seeded days, 142 considered commands, 46 near-misses:
32.4%, or 1.18 per alert, and none of them executed.** The audit reads the
lesson's own audit log and flags commands that are destructive by effect and
were not approved.

| audit view | near-misses | rate |
|---|---:|---:|
| by effect, correcting for lost approvals | 46 | 32.4% |
| by effect, as the log records it | 49 | 34.5% |
| keyed on the lesson's `destructive_tools` | 32 | 22.5% |

The lesson's list misses `kubectl_drain` and `kubectl_exec`, which it logs as
"blocked: unknown tool", although the skill file says exec is "destructive
in effect". The log also drops approvals. For an unlisted tool, `Agent.call`
records neither the approval nor the approver, so 3 of the 18 human
approvals read as near-misses. The gate itself is a string:
`approver="devops-agent"` executes all 4 destructive tools. `considered` is
True on all 142 events, so it carries no information.

### 3 — no model is called, so the swap moves accuracy 0 points

**The RCA accuracy delta is 0: 9/20 (45%) under either model, because
`main.py` never calls a model.** The only seam a model could fill is
re-ranking `root_cause`'s candidates. Even an oracle at that seam reaches
12/20 (60%), below the lesson's 80% rubric, because 8 of the 20 true causes
(OOM, HPA, PVC, certificate, quota, NetworkPolicy, DB outage, secret
rotation) never appear as candidates. The heuristic's #1 depends only on where
the alert fires: "bad rollout" for all 12 Deployment alerts and "node-level"
for all 8 Pod alerts.

| model (price per MTok in / out) | $ per 1,000 incidents |
|---|---:|
| Claude Sonnet 4.6 ($3 / $15) | 4.19 |
| Claude Sonnet 5 ($2 / $10) | 2.79 |
| Llama 3.3 70B, Together AI ($1.04 / $1.04) | 0.41 |

These costs assume a 146-token re-rank prompt and a 250-token answer. At
under half a cent per incident, cost cannot decide the swap. The lesson's
"Claude Sonnet 4.7" does not exist. Anthropic's lineup has Opus 4.7, Sonnet
4.6 and Sonnet 5.

### 4 — a 5-feature classifier finds 15 of 20 root causes, and the lesson's evidence score finds 0

**A standardised logistic regression, scored leave-one-scenario-out over 80
labelled spikes, picks the root cause in 15 of 20 scenarios.** The spikes
come from a seeded generator with three cascade assumptions: symptoms start
later, sit closer to the alert, and are larger and span more series.

| picker | root cause found |
|---|---:|
| logistic regression (LOSO) | 15 / 20 |
| earliest spike | 12 / 20 |
| biggest spike | 0 / 20 |
| lesson `Hypothesis.score` | 0 / 20 (19 symptoms, 1 coincidental) |

The coefficients read as a causal rule. More series (-1.52) and bigger
magnitude (-1.37) point to a symptom. Earlier onset (+0.50) and a preceding
change (+0.32) point to the cause. The lesson's score does the opposite on
every term: recency rewards the latest onset, citation count rewards the
widest spike, and path inverse rewards the spike nearest the alert. It is
also not the formula the doc states. The doc gives a product; the code
computes a weighted sum, so an uncited hypothesis still scores 0.09. Recency
is clamped, so onsets 60, 61 and 999 minutes before the alert all score 0.265.

### 5 — the rollback dry-run rejects 4 of 4 bad plans the gate would execute

**The dry-run verifies `main()`'s plan (roll back to revision 41) on a staging
copy of the same manifest, before approval: a 2-field diff (revision 42 -> 41,
image v2.41 -> v2.40) on the Deployment alone.** It rejects revision 999,
revision 42 (the current one), -1 and "latest". Once approved, the lesson's gate
executes all four.

The lesson gives the dry-run little to work with. The graph holds only
revision 42, so on its own every rollback is "unverifiable"; the history
here comes from the lesson's "Use It" transcript. After the rollback,
`root_cause` still ranks "bad rollout: image checkout-api:v2.40 fails
/healthz" #1, and its score rises from 0.735 to 0.817 because the fresh
rollout counts as newer evidence. The agent cannot tell that a rollback
worked. The gate keys on tool name, so no verification step runs before
approval: a `dry_run=True` rollback is "blocked: no slack approval" and an
`argocd_app_diff` is "blocked: unknown tool". In real ArgoCD, `app rollback`
has no `--dry-run` flag; the preview is `argocd app sync --revision R
--dry-run`.
