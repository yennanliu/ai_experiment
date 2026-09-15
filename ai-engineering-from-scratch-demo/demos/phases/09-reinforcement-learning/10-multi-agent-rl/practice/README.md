<!-- generated:start -->
# 09-reinforcement-learning / 10-multi-agent-rl

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/10-multi-agent-rl/) · upstream spec
`phases/09-reinforcement-learning/10-multi-agent-rl/docs/en.md`

```bash
uv run demo practice run 10-multi-agent-rl --ex 1
uv run demo explain 10-multi-agent-rl --ex 1
uv run pytest demos/phases/09-reinforcement-learning/10-multi-agent-rl
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Train independent Q-learning on the 2-agent cooperative GridWorld. How many episodes un… | code | T0 | `ex01_the_joint_state_costs_four_times_the_episodes.py` |
| 2 | Medium. Add a "coordination" task: the goal is reached only when both agents step onto it on… | code | T0 | `ex02_it_converges_until_the_agents_are_actually_independent.py` |
| 3 | Hard. Implement a centralized critic for MAPPO-style training and compare convergence speed t… | code | T1 | `ex03_the_critic_is_not_the_bottleneck_the_actor_is.py` |
<!-- generated:end -->

## Answers

One line of the lesson decides all three exercises:

```python
Q1 = defaultdict(default_q)          # indexed by s = (pos1, pos2)
```

Both "independent" Q-tables are keyed by the **joint state**. The agents are
independent in their *actions* only — each one sees exactly where the other is
standing at every step, which is the information independent learners are usually
defined by not having. Exercise 1 measures what that costs, exercise 2 measures
what it hides, and exercise 3 measures where it actually has to arrive.

Exercises 1 and 2 are **T0** (under 2 s); exercise 3 is **T1** (about 29 s).

### 1 — the joint state costs four times the episodes, and buys nothing here

**ANSWER: a median of 735 episodes, and one seed of ten never gets there.** Final
mean return 0.30, above zero on 8 of 10 seeds.

**FINDING: the best possible return is 3.0, not 10.** `step` pays `+10` on the joint
arrival and `−1` on every other turn. Agent 1 starts 8 moves from the goal, agent 2
starts 4, so the fastest episode is 8 turns and scores `−7 + 10`. The achievable band
is `[−90, 3.0]` — nowhere near the +10 the reward advertises.

**FINDING: the joint state costs 4× the episodes and buys nothing.**

| each agent indexes on | states | median crossing | final mean | above zero |
|---|---:|---:|---:|---:|
| joint `(pos1, pos2)` (shipped) | 625 | 735 | 0.30 | 8/10 |
| **own position** | **25** | **188** | **1.27** | **10/10** |

**MECHANISM: the shipped task is separable.** `done` requires both agents to *be* at
the goal, not to arrive together, and `move` at the goal returns the goal — so
whichever arrives first stands still and waits. Each agent's optimal policy is "walk
to the goal", which does not depend on the other at all. The joint state is 25× more
to learn with nothing in it to learn.

### 2 — it converges, until the agents are actually independent

The shipped `step` already requires both agents at the goal together, so the
exercise's task is a strictly stronger rule: neither agent may already be standing
there. That one clause is the only change.

| observation | rule | crossed | median | final mean | above zero |
|---|---|---:|---:|---:|---:|
| joint state | shipped | 9/10 | 735 | 0.30 | 8/10 |
| joint state | **strict** | 10/10 | 738 | −0.13 | 5/10 |
| own position | shipped | 10/10 | **188** | **1.27** | 10/10 |
| own position | **strict** | **0/10** | — | **−4.69** | **0/10** |

**ANSWER: yes, it still converges — because the learners are not independent.**
Requiring simultaneous arrival changes almost nothing (738 against 735) when each
agent can see where the other is standing when it decides.

**FINDING: what breaks is stability, not convergence.** Final mean falls from 0.30 to
−0.13; seeds ending above zero fall from 8 to 5.

**FINDING: take the joint state away and the task becomes unsolvable.** With local
observation the shipped rule is *easier* — 10/10 at a median 188 — and the strict rule
is **0 of 10**, never once crossing zero in 1,500 episodes. That is the coordination
failure the exercise is looking for.

**MECHANISM: camping is what made the shipped rule separable.** Removing waiting means
arrival has to be *timed*, and an agent that cannot see its partner has no way to
represent timing.

**FINDING: so the exercise's premise is already true of the lesson.** `done = (new1 ==
GOAL) and (new2 == GOAL)` is a coordination condition as shipped. What made it
coordinate-able was never the rule — it was the joint state the learners were handed.

### 3 — the centralised critic is not the bottleneck; the actor's eyes are

MAPPO is centralised training with decentralised execution, so both arms keep actors
over each agent's own position and differ only in the critic's input.

| task | actor sees | critic sees | above zero | final mean |
|---|---|---|---:|---:|
| **coordination** | local | local | 5/10 | −39.1 |
| **coordination** | local | **central** | **3/10** | **−69.1** |
| separable | local | local | 8/10 | −17.7 |
| separable | local | **central** | **10/10** | **+2.97** |
| **coordination** | **joint** | local | **10/10** | **+2.36** |
| **coordination** | **joint** | central | **10/10** | **+2.75** |

**ANSWER: on the coordination task the centralised critic makes it worse** — 3/10
against 5/10, and neither arm solves it.

**FINDING: on the separable task the same critic is a clear win** — 10/10 against
8/10, +2.97 against −17.7. It works exactly where coordination is *not* required,
which is the opposite of the setting the exercise proposes it for.

**MECHANISM: the actor cannot represent what the critic has learned.** Giving the
*actor* the joint state solves the coordination task outright, and the critic's input
then barely matters (+2.36 local, +2.75 central). MAPPO trains centrally and executes
locally — and a local policy here cannot *express* the timing the task requires, no
matter how good the value estimate steering it is. It is not free either: the
joint-state actor crosses at a median 1,369 episodes against 263 for a local one, the
same 625-against-25 cost the first two exercises measured.
