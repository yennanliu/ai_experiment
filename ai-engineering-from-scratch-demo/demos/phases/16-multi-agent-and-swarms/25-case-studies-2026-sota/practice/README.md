<!-- generated:start -->
# 16-multi-agent-and-swarms / 25-case-studies-2026-sota

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/25-case-studies-2026-sota/) · upstream spec
`phases/16-multi-agent-and-swarms/25-case-studies-2026-sota/docs/en.md`

```bash
uv run demo practice run 25-case-studies-2026-sota --ex 1
uv run demo explain 25-case-studies-2026-sota --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/25-case-studies-2026-sota
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read the Anthropic Research system post end-to-end. Identify three design decisions that woul… | code | T0 | `ex01_the_fifteen_x_is_against_a_chat_and_the_mapper_has_no_field_for_the_model.py` |
| 2 | Read MetaGPT Sections 3-4 (arXiv:2308.00352). Encode one SOP from your own domain (not softwa… | code | T0 | `ex02_peer_review_implies_four_roles_and_six_artifacts_and_the_mapper_needs_one_boolean.py` |
| 3 | Read ChatDev (arXiv:2307.07924). Identify the mechanism of "communicative dehallucination." I… | code | T0 | `ex03_the_only_answer_that_dehallucinates_is_the_verifiers_own_tests.py` |
| 4 | Read about OpenClaw and Moltbook. Pick one specific failure mode that emerged at population s… | code | T0 | `ex04_the_rare_agent_that_obeys_the_feed_exists_only_at_population_scale.py` |
| 5 | Pick your current multi-agent project. Which of the three case studies is the closest referen… | code | T0 | `ex05_this_build_is_the_research_shape_and_the_mapper_files_it_under_metagpt.py` |
<!-- generated:end -->

## Answers

Sources were fetched on 2026-09-25: the Anthropic engineering post, the arXiv
HTML of MetaGPT and ChatDev, the MetaGPT repository README, and the Wikipedia
articles on OpenClaw and Moltbook. The lesson's own `code/main.py` is a
case-study mapper, and every exercise also runs it.

### 1 — the 15x is against a chat, and the mapper has no field for the model

A smaller model changes three decisions, each tied to a sentence in the post:

1. **Who leads.** The 90.2% is "Claude Opus 4 as the lead agent and Claude
   Sonnet 4 subagents" against single-agent Opus 4. The post also finds
   "upgrading to Claude Sonnet 4 is a larger performance gain than doubling
   the token budget on Claude Sonnet 3.7". So the smaller model goes into
   the subagent slots first; the lead is the last seat to downgrade.
2. **Effort scaling.** The lead picks "1 agent with 3-10 tool calls", "2-4
   subagents with 10-15 calls each" or "more than 10 subagents". A smaller
   lead judges complexity worse, so those tiers move out of its prompt into
   explicit rules.
3. **Budget.** "Token usage by itself explains 80% of the variance". A cheaper
   model gets bought back with tokens, which is the lever the post measured
   as the weaker one.

None of the three can reach the lesson's mapper. `Design` has 7 fields and
none of them is a model, a budget or a call count. So a Haiku swap changes
no recommendation.

The lesson's summary of the post also gets two things wrong:

- **"15x tokens per query vs single-agent" is 15x against a chat.** The post
  says "agents typically use about 4x more tokens than chat interactions, and
  multi-agent systems use about 15x more tokens than chats". Against a single
  agent that is **3.75x**.
- **There is no verifier role.** The lesson says the system "was observed to
  hallucinate without explicit verifier roles", and the mapper's Research
  case lists "verification role". The post's one hallucination sentence is
  about human testers finding "hallucinated answers on unusual queries". Its
  extra agent is a CitationAgent: "This ensures all claims are properly
  attributed to their sources". That is attribution, not verification.

### 2 — peer review implies four roles and six artifacts

The SOP is journal peer review, written the way MetaGPT §3.1 writes roles:
who acts, what it reads, what it publishes.

| step | role | reads | publishes |
|---:|---|---|---|
| 1 | author | — | manuscript |
| 2 | editor | manuscript | desk decision |
| 3 | editor | desk decision | invitations |
| 4–5 | reviewer ×2 | manuscript, invitations | report |
| 6 | editor | reports | decision letter |
| 7 | production | decision letter, manuscript | proof |

**Four roles by actor, six by MetaGPT's convention of one artifact per
role.** MetaGPT's five roles map one-to-one onto five artifacts: PRD, system
design, task list, code, tests. Peer review does not map that way, because
the editor publishes 3 of the 6 artifacts. There are two choices. Keep 4
roles and give the editor's prompt three output schemas, or split the editor
and get 6 narrow roles, which is what MetaGPT would do.

The SOP is a DAG six stages deep, not a chain of seven steps. The two
reviewers read the same inputs, so a publish-subscribe pool runs them
together.

The lesson's mapper decides role decomposition on one boolean the user sets.
With `roles_distinct=True` this SOP maps to MetaGPT/ChatDev; with `False` it
maps to Anthropic Research, a parallel research supervisor. No field
describes order, artifacts or handoffs.

A note on the lesson's attribution: `Code = SOP(Team)` is the MetaGPT
repository's tagline ("is the core philosophy"). I did not find it in the
arXiv HTML.

### 3 — the only answer that dehallucinates is the verifier's own tests

**The mechanism is role reversal: the assistant asks the instructor for the
missing detail before it answers.** ChatDev §3.2 describes the assistant
"proactively seeking more specific information ... before delivering a
conclusive response". §4 activates it during code completion, review and
testing. The ablation in Table 4 drops Quality from 0.3953 to 0.3094 without
it. The lesson's example has a designer asking a programmer, but the paper's
roles are CEO, CTO, programmer, reviewer and tester.

Implemented in Lesson 08's pipeline, with the lesson's own critic and
verifier judging every artifact:

| loop | artifacts | messages | ships |
|---|---:|---:|---|
| guess, get rejected, guess again (from `a * b`) | 4 | 7 | `a + b` |
| ask the planner first | 1 | 3 | `a + b` |

The critic approved all 4 guesses, so only the verifier drives the loop.

What can the planner actually say? The signature admits all 4 candidates.
The description is the user's wish, verbatim. Only a test case,
((1, 2), 3), narrows the choice to one. But an executor holding the tests can
return a lookup table, and the reference verifier passes it. So the detail
that removes the guess also removes the verifier's independence. And because
`planner` returns the same spec for every wish, asking about "the product of
two integers" gets the sum's tests back. `a + b` ships, approved and passed.
The mechanism faithfully relays a spec that was wrong before anyone asked.

### 4 — the rare agent that obeys the feed exists only at population scale

**Failure mode: indirect prompt injection through the shared feed, reaching
the rare agent that follows it.** Moltbook agents "check Moltbook every 30
minutes or so", and researchers called the site "a vector for indirect
prompt injection".

Model it with one assumed rate: 1 in 1000 agents follows instructions found
in posts.

- **At 5 agents.** A system holds such an agent with probability 0.50%, so
  testing one deployment will not find it.
- **At 1.5M agents** there are 1500 of them. Suppose each one reposts and a
  repost reaches 2000 readers. The payload then grows ×2 per 30-minute tick:
  past 1000 compromised agents at tick 11, five and a half hours in. It ends
  at 1304 once the susceptible agents run short.

**Engineer at the platform, where the control does not depend on that rate.**
Cap the reach of identical content at 500 and the growth factor drops to
0.5; the outbreak ends at 2 agents. Also limit by principal, not by account.
The leaked tokens show 1.5 million agents belonging to 17,000 owners, 88
each, so a limit per account is 88 times weaker per human.

The mapper cannot see any of this. `map_to_case` never reads
`n_agents_expected`: a 5-agent design labelled "population" maps to
OpenClaw, and a 10,000-agent research design maps to Anthropic Research.

Only 1 of the 3 cases in `CASES` carries a security pattern, although the
lesson says "Security posture is explicit" in all three.

Two dates and names in the lesson disagree with its sources. Wikipedia has
Moltbook launching "January 28, 2026", not in February. It has OpenClaw
"first published in November 2025 under the name Warelay", not Clawdbot.

### 5 — this build is the Research shape, and the mapper files it under MetaGPT

The current multi-agent project is the one that wrote these files. A lead
agent fans lessons out to four forked workers at a time, each in a fresh
context, then reviews, gates and commits every lesson itself.

| case | patterns adopted |
|---|---:|
| Anthropic Research | 3 of 4 — fresh-context subagents, orchestrator synthesis, verification role |
| MetaGPT / ChatDev | 1 of 4 — structured artifact handoffs (`practice.yaml`) |
| OpenClaw / Moltbook | 0 of 4 |

**Closest reference: Anthropic Research.** The mapper disagrees. Described
honestly (engineering, 5 agents, verification, no distinct roles), it
returns MetaGPT/ChatDev on `task_type` alone. Relabel the task "research"
and the same structure maps to Research. Two of its five inputs,
`verification_required` and `runtime_duration_hours`, change 0 of 64
mappings. The lesson's "classic enterprise automation" has no case at all.

**Not yet adopted:** rainbow deployment, which does not apply to a batch
build, and asynchronous dispatch. The post says its "lead agents execute
subagents synchronously, waiting for each set of subagents to complete", and
so does this build, which starts a batch of four only when the previous
batch has returned.

**The one to adopt this quarter: dispatch the next lesson as soon as any
worker finishes,** instead of paying for the slowest fork in every batch.
