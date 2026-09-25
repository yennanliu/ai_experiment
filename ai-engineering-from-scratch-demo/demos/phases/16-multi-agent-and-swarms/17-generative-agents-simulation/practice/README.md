<!-- generated:start -->
# 16-multi-agent-and-swarms / 17-generative-agents-simulation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/17-generative-agents-simulation/) · upstream spec
`phases/16-multi-agent-and-swarms/17-generative-agents-simulation/docs/en.md`

```bash
uv run demo practice run 17-generative-agents-simulation --ex 1
uv run demo explain 17-generative-agents-simulation --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/17-generative-agents-simulation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm 3+ agents converge at the party. Increase agents to 10 — does the… | code | T0 | `ex01_the_invitations_are_scripted_by_index_so_agents_five_to_nine_never_hear.py` |
| 2 | Remove the reflection step. What does behavior look like? Map to the ablation finding in Park… | code | T0 | `ex02_without_reflection_nobody_comes_because_reflection_is_the_only_wire.py` |
| 3 | Introduce a competing seeded goal ("Klaus wants to give a research talk at 5pm"). Do agents s… | code | T0 | `ex03_the_talk_cannot_spread_and_when_it_can_the_first_invitation_wins.py` |
| 4 | Add spatial constraints: Hobbs Cafe holds at most 4 agents. Does the simulation handle overfl… | code | T0 | `ex04_the_fifth_guest_is_turned_away_and_never_tries_again.py` |
| 5 | Read Park et al. (arXiv:2304.03442) Section 6 (emergent behavior experiments). Identify one b… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the invitations are scripted by index, so agents five to nine never hear

**3 or more converge at n=5: 5 of 5. At n=10 it is 5 of 10, the same five.**

| `run_simulation(n)` | converged |
|---:|---|
| 3, 4 | `IndexError` |
| 5 | 5 / 5 |
| 10 | 5 / 10 |
| 25 | 5 / 25 |

Agents 5–9 receive **0** observations. Nothing spreads, because the invitations
are three `if tick == ...` branches in `run_simulation` that name agents 1, 2,
3 and 4 by index. `Agent` has four public methods (observe, reflect,
update_plan, act) and none takes another agent, so no agent can invite anyone.
"No orchestrator. One seed." is printed by the only code deciding who hears
what: the loop itself.

If the agents do the inviting instead, emergence does scale. Each agent that
holds the belief invites one uninvited agent per tick, through the same
`observe`. The reference `Agent` then converges **10 of 10** and **25 of 25**.

It still stops at a wall. `update_plan` writes `Plan(tick=5, ...)` whatever the
invitation says, so an invitation "at tick 3" still plans tick 5. `act` fires
only when `p.tick == tick`, so `run_simulation(5, ticks=5)` converges 0 of 5.
Under the relay the knowers double each tick, reaching 64 by the end of
tick 5. Anyone invited later can never attend, so n=100 converges 64.

Also: `retrieve_top_k`, one of the lesson's three components, is called by
neither the simulation nor `Agent`.

### 2 — without reflection nobody comes, because reflection is the only wire

**Only the host comes: 1 of 5.** Agents 1–4 still receive their invitations,
at importance 7 and 8, and nothing reads them. `update_plan` looks only at
`beliefs`, and only `reflect` writes beliefs. Agent 0 attends because
`run_simulation` seeds its belief directly.

Park et al.'s ablation (§6, controlled evaluation) has the opposite shape:

| condition | TrueSkill μ |
|---|---:|
| full architecture | 29.89 |
| no reflection | 26.88 |
| no reflection, no planning | 25.64 |
| human crowdworker | 22.95 |
| no observation, reflection or planning | 21.21 |

Their agents without reflection got *worse*, and were still rated above human
authors. These agents stop. The ablations were also nested (reflection, then
planning too, then observation too), not one component dropped at a time as
the lesson's "dropping any one produces a measurable regression" implies.
Observation was never removed on its own.

The reason is that reflection here is a relay, not a synthesis. Delete
`reflect`, put its test (the words "invited" and "party at" in an importance
6+ memory from the last 5 ticks) straight into `update_plan`, and every final
location and plan is identical. The belief it writes is one fixed string,
"there is a party I was invited to", with no host, place or time in it.
Park's motivating example needs the agent to generalise *from content*:
Klaus picks Maria over Wolfgang because both are dedicated to research. Here
there is no content to generalise from.

And the trigger is brittle even with reflection on. Of 6 phrasings of the same
invitation, **3** form a belief. "asked me to come to her party", "a
Valentine's party, at HobbsCafe" and "Invited me to a Party" fail, because
`reflect` matches two case-sensitive substrings.

### 3 — the talk cannot spread, and when it can, the first invitation wins

Klaus is agent-4. He is seeded with a Library plan at tick 5 and invites
agent-3 at tick 0 and agent-1 at tick 1.

**On the reference, the party dominates completely, for a mechanical reason.**
The party gets agents 0–3; the talk gets Klaus alone. His invitations say
"research talk at Library", and `reflect` only forms a belief from a memory
containing "party at". A substring in `reflect` decides which goal can exist.

Let reflection form one belief per invited place and the agents **split,
3–2**. Agent-3 heard about the talk first and goes to the Library. Agent-1
heard about the party first and goes to the cafe. What decides it is
invitation order: `act` moves an agent to the *first* plan whose tick matches,
and plans are appended in the order beliefs formed. Deliver each talk
invitation one tick after that guest's party invitation and the talk keeps
only Klaus.

Importance, the one signal the memory records, changes nothing. Talk
invitations at 10 against party invitations at 6 give the same 3–2 split,
because importance only gates reflection at 6 and retrieval, which would weigh
it, is never called. Klaus himself holds two tick-5 plans (he got the party
invitation at tick 2) and goes to his talk only because his seeded plan comes
first in the list.

### 4 — the fifth guest is turned away and never tries again

**Unmodified, it hits the bathroom pattern exactly.** Five agents enter a
four-agent cafe at tick 5, with no error and no observation. A location is a
bare string, and "capacity" appears nowhere in the module. Park et al. §7.2
describes the same failure: agents entered a dorm bathroom "that can only be
occupied by one person" while someone was inside, because the norm "did not
percolate to the agents". Their proposed fix is to put the norm in the
location's state.

The obvious capacity rule (admit 4, refuse the fifth, record "HobbsCafe was
full" in its memory) does not make overflow graceful:

- **Refusal is permanent.** Agent-4 is still at home at tick 11. `act` fires
  a plan only when `p.tick == tick`, and `update_plan` adds a cafe plan only
  if none exists, so a missed appointment can never be rescheduled.
- **Who is refused is loop order.** Visit the agents in reverse and agent-0,
  the host who seeded the party, is turned away from its own party.
- **Waiting outside never works.** Give the refused agent a new plan for the
  next tick after each refusal: it tries 7 times in 7 ticks and gets in 0
  times. `act` only moves an agent *to* a place, and a `Plan` is an instant,
  not an interval, so no guest ever leaves. Park's agents "wait outside the
  bathroom if it is occupied"; that behaviour needs departure times, which
  this plan model cannot express.

### 5 — the party diffused because agents talked; here the loop does the talking

*Draws on "The Valentine's Day emergence".*

A note on sections first: §6 is the controlled evaluation (the interview-based
ablation quoted in answer 2). The emergent-behaviour experiments are §7.1.

**Not reproducible: information diffusion through conversation.** In
Smallville, Isabella's party went from 1 agent who knew (4%) to 13 (52%) over
two game days, and Sam's candidacy from 1 to 8 (32%). The authors checked
every "yes" against the dialogue in that agent's memory stream to rule out
hallucination. Network density rose from 0.167 to 0.74. On the day, 5 of the
12 invited agents came; 3 cited conflicts, and 4 said they were interested but
did not plan to come.

The lesson's own account, step 5, is "Neighbors tell other neighbors". None
of that can happen in the miniature. Knowledge moves only when
`run_simulation` calls `observe` on a hard-coded index (answer 1), so there is
nothing to measure a diffusion *rate* against. Attendance is deterministic, 5
of 5, because a belief turns into a plan without weighing any other
commitment. And nobody can be interested but absent, since there is no second
plan to conflict with (answer 3).

**The component to enhance is the agent-to-agent channel.** In Park's
architecture this sits in planning and reacting (§4.3): an agent perceives
another agent nearby, retrieves relevant memories, and decides whether to
start a dialogue, whose lines are written into *both* memory streams. That
needs three things the miniature lacks: a `perceive` step driven by
co-location, retrieval actually feeding the decision (`retrieve_top_k` is
never called), and plans with durations, so that a conflict and a departure
can exist at all.
