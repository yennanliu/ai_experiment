<!-- generated:start -->
# 16-multi-agent-and-swarms / 18-theory-of-mind-coordination

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination/) · upstream spec
`phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination/docs/en.md`

```bash
uv run demo practice run 18-theory-of-mind-coordination --ex 1
uv run demo explain 18-theory-of-mind-coordination --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm first-order ToM reduces duplication rate by ~7x. Does the gap per… | code | T0 | `ex01_the_prime_is_a_seating_chart_and_the_belief_model_does_nothing.py` |
| 2 | Implement second-order ToM (agent A models what B thinks about C). Does it improve over first… | code | T0 | `ex02_second_order_helps_only_when_exactly_one_agent_is_left_with_one_box.py` |
| 3 | Inject a hallucination into the ToM state: randomly flip one belief per turn. How much does t… | code | T0 | `ex03_one_wrong_belief_costs_half_a_collision_and_only_before_turn_one.py` |
| 4 | Read Li et al. (arXiv:2310.10701). Reproduce the "long-horizon degradation" finding: as turns… | code | T0 | `ex04_the_horizon_never_arrives_because_every_episode_ends_by_turn_three.py` |
| 5 | Read Riedl 2025 (arXiv:2510.05174). Implement the higher-order synergy statistic on your simu… | code | T0 | `ex05_the_synergy_is_all_in_the_no_tom_runs_and_it_is_the_collision_rule.py` |
<!-- generated:end -->

## Answers

### 1 — the prime is a seating chart and the belief model does nothing

**It is not 7x.** The lesson's ~7x is its expected ~35% against ~5%. The
bench prints 0.965 duplications per trial against **0.00**, a ratio with zero
in the denominator. The gap persists at 5 agents and 5 boxes, 1.97 against
0.00. Both conditions complete 200 of 200 trials; the lesson expects ~60% and
~95%.

| condition | full completions | duplications / trial | turns |
|---|---:|---:|---:|
| zeroth-order, 3x3 | 200 | 0.965 | 1.855 |
| first-order ToM, 3x3 | 200 | 0.00 | 1.0 |
| zeroth-order, 5x5 | 200 | 1.97 | 2.285 |
| first-order ToM, 5x5 | 200 | 0.00 | 1.0 |

The ToM condition has two ingredients, and taking them apart shows where the
gap comes from.

**Without the turn-0 "preference prime", first-order ToM *is* zeroth-order,
trial for trial**: identical results on 400 of 400 trials. After a collision,
the boxes a loser saw others target last turn are exactly the boxes just
collected, which are no longer available anyway. So the avoidance rule never
removes an option, and the same seed draws the same boxes.

**The prime is a fixed assignment by index.** Each agent is told that agent
j "prefers" box `j % n_boxes`, so agent i's only unclaimed box is box i. On
every seed, the ToM agents' turn-0 choices are exactly [0, 1, …, n−1]. An
agent with no model of anyone that simply takes its own number matches the
ToM condition on every trial. The measured coordination effect is a seating
chart handed out before turn 0.

The agent Build It describes, a `ToMAgent` with own beliefs and
per-other-agent belief models, is not in the module. There is an `Agent`
with a boolean `tom` and a flat list of (name, box) observations.

### 2 — second order helps only when exactly one agent is left with one box

The second-order agent rebuilds each other agent's observation window, which
is public. It runs that agent's own first-order rule on it and treats the
result as claimed whenever the rule leaves that agent a single box.

**On the lesson's task it improves nothing: it equals first-order on 800 of
800 trials, primed or not.** Primed, every agent is already down to one box.
Unprimed, nobody ever is.

It does help when one agent knows more than the others. Here that means a
prime in which some agents never announce a box:

| silent agents | first-order | second-order | zeroth-order |
|---|---:|---:|---:|
| 1 of 3 | 1.04 | **0.00** | 0.965 |
| 1 of 4 | 1.615 | **0.00** | 1.40 |
| 1 of 5 | 2.525 | **0.00** | 1.97 |
| 2 of 5 | 2.65 | 2.65 | 1.97 |

The silent agent heard everyone and is left with one box. The others heard
it say nothing, so each has two boxes left. Reasoning about the silent
agent's beliefs is what tells them which of the two is taken. With two
silent agents nobody is down to one box, and second order is first order
again.

The table also shows that **a partial prime makes first-order ToM worse than
no ToM at all**. The announced agents all avoid the same announced boxes and
pile onto the unannounced one.

Li et al. define second order more narrowly: "what others believe about *my*
mental state". GPT-4 scored 64.3% on it against 60.0% first-order, the one
row of their Table 2 where second order is not lower.

### 3 — one wrong belief costs half a collision, and only before turn one

**One random belief flip per turn takes duplications from 0.00 to 0.505 at
3x3 and to 0.47 at 5x5.** That is half the zeroth-order gap at 3x3 and a
quarter of it at 5x5. The zeroth-order agents get the same flips as a
control and move only from 0.965 to 0.975. They never read their beliefs;
the flips just perturb their random stream. So the ToM loss is belief damage.

**Only the turn-0 flip does anything.** Every duplication falls on turn 0:
101 of 101 at 3x3 and 94 of 94 at 5x5. The prime is the only belief that
ever removes an option, and here "one hallucination per turn" amounts to one
per episode, because the episode lasts one turn.

**The worst single wrong belief makes a collision certain.** I enumerated
every one-belief rewrite of the prime:

| size | rewrites | harmless | mean duplications | worst |
|---|---:|---:|---:|---|
| 3x3 | 18 | 6 | 0.491 | a collision on 200 of 200 seeds |
| 5x5 | 100 | 20 | 0.49 | a collision on 200 of 200 seeds |

The harmless rewrites are exactly the no-ops, the ones that rewrite a box to
itself. Every real rewrite costs something. An agent that thinks agent 1
wants box 0 is left with box 1, and walks into agent 1.

### 4 — the horizon never arrives, because every episode ends by turn three

*Draws on "Li et al. (arXiv:2310.10701)".*

**Nothing changes.** Zeroth-order, first-order, and first-order with a
hallucinated belief per turn all give identical results on every trial at
10, 20 and 30 turns, at both sizes. No episode in any condition lasts longer
than 3 turns, so the 10-turn budget is never reached.

**The paper does not contain a 10-to-30 result either.** Li et al.'s
bomb-defusal game (3 agents, 5 rooms, 5 bombs) ends on success, on deadlock,
or at 30 rounds, and no experiment varies the horizon. Their long-horizon
failure is about *where* facts sit in the context. Agents emit invalid
actions even though room connectivity is in the initial prompt, because that
information is far away (§6.4.1). An explicit belief state cut invalid
actions by 50.7% and raised GPT-4's first-order ToM accuracy from 60.0% to
80.1%.

The same mechanism exists in this code, but the task ends before it
matters. The avoidance rule reads at most the last n + 2 observations, and
each turn adds n − 1 more, so the prime drops out of the window after 3
turns at 3x3 and 2 turns at 5x5. By then it has already been used, on turn
0.

With more agents than boxes, a longer horizon only adds empty turns. Four
agents on 3 boxes get 0 full completions and 1.85 duplications per trial at
both 10 and 30 turns, and every trial uses the whole budget.

### 5 — the synergy is all in the no-ToM runs, and it is the collision rule

*Draws on "Riedl 2025 (arXiv:2510.05174)".*

Riedl's statistic is a partial information decomposition of time-delayed
mutual information (TDMI). For each agent pair, I({X_i,t, X_j,t}; T_ij,t+1)
is split into unique, redundant and synergistic parts, using Williams–Beer
I_min redundancy on plug-in probabilities (§2). The paper adds a "practical
criterion", S_macro = I(V_t; V_t+1) − Σ I(X_k,t; V_t+1).

Here X is each agent's committed box, T is the pair's next choices, and V is
the number of agents still searching. A shuffled-target control, which is
not in the paper, sizes the estimator's bias.

| 3x3 condition | synergy (bits) | TDMI (bits) | S_macro |
|---|---:|---:|---:|
| zeroth-order | 0.480 | 0.965 | −0.505 |
| zeroth-order, target shuffled | 0.078 | — | — |
| first-order ToM (primed) | 0 | 0 | 0 |

**Is the effect present without the ToM condition? Yes, and only there.**
Under the prime every agent takes box i and is done on turn 0. Every variable
is constant, so there is no information to decompose. The unprimed ToM
condition is the zeroth-order condition trial for trial (exercise 1), so it
has the same synergy.

**The synergy comes from the collision rule, not from coordination.** All of
it sits in the turn-0 → turn-1 step: 0.885 bits for agents 0 and 1, and 0
after. Whether agent 1 is still searching on turn 1 depends on whether it
picked the *same* box as agent 0. That is XOR-shaped: a function of both
choices that neither choice predicts alone. The environment's
first-in-order-wins resolution produces it, and no agent models anything.

The practical criterion finds no emergence in any condition. At 5x5 the
estimate is mostly bias: 0.524 bits measured against 0.390 shuffled, with 200
trials of plug-in counts over six values per agent.
