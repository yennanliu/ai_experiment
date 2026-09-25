<!-- generated:start -->
# 16-multi-agent-and-swarms / 20-marl-maddpg-qmix-mappo

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/20-marl-maddpg-qmix-mappo/) · upstream spec
`phases/16-multi-agent-and-swarms/20-marl-maddpg-qmix-mappo/docs/en.md`

```bash
uv run demo practice run 20-marl-maddpg-qmix-mappo --ex 1
uv run demo explain 20-marl-maddpg-qmix-mappo --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/20-marl-maddpg-qmix-mappo
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Measure the steps-to-goal gap between independent and MAPPO-style agents.… | code | T0 | `ex01_three_ctde_rows_are_one_policy_and_it_splits_pellets_one_agent_should_take.py` |
| 2 | Implement a competitive variant: two agents, one pellet, only the first to reach gets reward.… | code | T0 | `ex02_the_race_is_decided_at_spawn_and_a_team_reward_cannot_see_who_won.py` |
| 3 | Read MADDPG (arXiv:1706.02275) Section 3. Implement the exact critic update rule symbolically… | explain | T0 | prose, below |
| 4 | Read MAPPO (arXiv:2103.01955). Why do the authors argue centralized value + PPO beats off-pol… | explain | T0 | prose, below |
| 5 | Apply CTDE as a design pattern to a hypothetical LLM-agent system (e.g., research agent + sum… | code | T0 | `ex05_the_demos_ctde_runs_its_critic_on_every_deploy_step.py` |
<!-- generated:end -->

## Answers

### 1 — three CTDE rows are one policy, and it splits pellets one agent should take

**The gap grows in steps and holds in share.** Over the demo's own 500 seeds:

| grid | independent | MAPPO-style | gap | share |
|---|---:|---:|---:|---:|
| 4x4 | 3.212 | 2.726 | 0.486 | 15.1% |
| 6x6 | 4.654 | 4.014 | 0.640 | 13.8% |

Longer walks give duplicated effort more steps to waste. The *fraction* of
steps wasted barely moves.

The lesson's expected output is off by nearly 2x. It promises "~6 steps"
independent and "~3.5" for CTDE, with 3 as the optimum. A breadth-first search
over joint positions puts the true optimum at an average of **2.698**. That is
below the stated optimum, and exactly 3 in only 204 of 500 episodes.

**MADDPG, QMIX and MAPPO are one policy.** `run_mappo_style` returns
`run_maddpg_style(env)`, and QMIX's one-pellet branch computes what
`_assigned_targets` already returns. The three rows agree on **500 of 500**
episodes at both sizes, and nothing in any of them learns.

**The "centralized critic" is not optimal.** It gives each agent a distinct
pellet and picks the split with the smaller *total* distance. But
steps-to-goal is the *longer* of the two walks. It is worse than the optimum
in 13 episodes at 4x4 and 31 at 6x6:

- 12 and 26 of them because a split with a shorter longest walk existed;
- the rest because the best plan has one agent take both pellets, which the
  critic never allows.

On seed 262 independent beats it, 2 steps to 3.

Finally, the takeaways describe mechanics the code does not have. "Only the
closer agent moves per step": `move_or_wait` is defined and never called.
`Env`'s docstring charges collisions an extra step, and no line implements
that.

### 2 — the race is decided at spawn, and a team reward cannot see who won

**MADDPG handles it cleanly because it is the only one of the three with a
critic per agent, each trained on that agent's own reward.** MADDPG §4.1:
"Since each Q_i is learned separately, agents can have arbitrary reward
structures, including conflicting rewards in a competitive setting."

QMIX mixes per-agent utilities into one Q_tot for a team reward. MAPPO fits
one V to a shared return. In this race the team return is exactly 1 in 500
of 500 episodes, whoever wins. Its variance is **0**, so a shared value has
nothing to learn about who won. Agent 0's own return varies 0/1 and carries
all of it.

The variant also shows the environment cannot express competition at all:

- **The race is decided before the first step.** A greedy walk takes exactly
  the Manhattan distance and `step_toward` never looks at the other agent, so
  nothing can block. The closer agent wins 500 of 500 times.
- **`Env` cannot say who won.** `collect_if_on_pellet` discards the pellet
  without recording the collector. 85 of 500 starts are equal-distance, and
  reading the loop order gives all 85 to agent 0: 256 wins to 244.
- **The shipped "centralized" runner is the independent one.** With one
  pellet, `_assigned_targets` returns `(p, p)`. `run_maddpg_style` and
  `run_independent` take identical steps on 500 of 500 episodes.

### 3 — the MADDPG critic update, which lives in §4.1, not §3

*Draws on "MADDPG (2017) — the CTDE pattern"; the paper is arXiv:1706.02275.*

Section 3 of the paper is Background (Q-learning, policy gradients, DPG and
DDPG). The critic update is equation (6), in §4.1 "Multi-Agent Actor Critic".
In pseudocode:

```text
for each agent i, with actor mu_i(o_i; theta_i) and critic Q_i(x, a_1..a_N; phi_i):
    sample a minibatch of (x, a_1..a_N, r_1..r_N, x') from the shared replay buffer D
    # x is the joint state -- in the simplest case all observations (o_1..o_N)
    # the target uses every agent's TARGET actor on its own next observation
    a'_j = mu'_j(o'_j)                     for every agent j
    y    = r_i + gamma * Q'_i(x', a'_1..a'_N)
    critic loss   L(phi_i)   = mean over batch of (Q_i(x, a_1..a_N) - y)^2
    actor gradient           = mean over batch of
                               grad_theta mu_i(o_i) * grad_{a_i} Q_i(x, a_1..a_i..a_N)
                               evaluated at a_i = mu_i(o_i), other actions from D
    soft-update Q'_i and mu'_i toward Q_i and mu_i
```

The idea that makes it multi-agent is inside the target. Because Q_i is
conditioned on *all* actions, the next-state value does not change when
another agent's policy changes. The paper states this as P(s' | s, a_1..a_N,
π) being equal for any π. So the non-stationarity that breaks independent
learners is absorbed by conditioning. The price is that equation (6) needs
every other agent's target policy. §4.2 relaxes this by fitting approximate
policies to observed actions (equations 7 and 8).

Nothing in `code/main.py` corresponds to any line of this. The module has
no Q, no γ, no replay buffer and no target network. Its "critic" is a
hand-written assignment by Manhattan distance, and ex05 finds it running at
deploy time.

### 4 — the paper does not credit the centralized value

*Draws on "MAPPO (2022) — the overlooked default"; the paper is "The
Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games",
arXiv:2103.01955.*

The question assumes the authors argue that a centralized value plus PPO
*beats* off-policy MARL. They argue something weaker, and they do not
attribute it to the centralized value. Their abstract says PPO "often
achieves competitive or superior results". And of the variant *without*
global information: "Despite not utilizing global information, IPPO also
achieves similar or superior performance to centralized off-policy methods."
Their three strongest claims:

1. **Final return.** PPO-based methods match or beat strong off-policy
   baselines on four testbeds: MPE, SMAC, Google Research Football and
   Hanabi. They do this "with minimal hyperparameter tuning and without any
   domain-specific algorithmic modifications or architectures".
2. **Sample efficiency is comparable**, contrary to the belief that
   on-policy methods waste samples. This is measured in environment steps.
   The paper gives no wall-clock comparison.
3. **Implementation details decide it.** The five factors are: value
   normalization; value input that includes both local, agent-specific
   features and global features; at most 10–15 epochs without splitting data
   into mini-batches; a clip ratio under 0.2; and large batches.

The lesson's own summary also miscounts. It says "five benchmarks" and lists
particle-world and MPE separately, but MPE *is* the particle world. The
abstract says "four popular multi-agent testbeds".

The centralized value appears in claim 3 as an *input-representation* choice.
It is not the reason PPO wins. The authors also limit the result's scope
themselves: "all benchmarks use discrete action spaces, are all cooperative,
and in the vast majority of cases, contain homogeneous agents". That is one
more reason ex02's competitive race belongs to MADDPG. In `code/main.py`,
the MAPPO row is a single line calling MADDPG's.

### 5 — the demo's CTDE runs its critic on every deploy step

**The joint information is the other agents' state and the joint outcome.**
For research agent → summarizer → coder, the design-time record is every
agent's full context and output on a task, plus whether the final result was
right. From that record you can see that the summarizer dropped the caveat
the coder needed, or that research and coder both fetched the same page. At
runtime the coder sees only the summary.

CTDE as a design pattern means using those joint traces to set each agent's
prompt, output schema and hand-off convention, then running each agent on
its own input alone. On the lesson's grid, an actor restricted that way
recovers **34%** of the coordination gap. The best role convention fixed at
design time scores 3.048 steps, against 3.212 independent and 2.726 with the
joint state.

Two measurements keep this honest:

- **The shipped "CTDE" is centralized execution.** `run_maddpg_style` says
  "Deploy-time only the actors run", yet it calls `_assigned_targets`, which
  reads both positions, on 1363 of its 1363 steps. The demo's LLM gloss, "the
  router decides which sub-agent advances", is the same pattern. A router
  consulted at runtime is the critic deployed.
- **No convention tried closes the remaining 66%.** Row-major, column-major
  and diagonal roles score 3.154, 3.048 and 3.088. What they lack is which
  pellet the *other* agent is nearer to. If the LLM system needs facts like
  that, the design has to send a message at runtime. CTDE only moves what can
  be decided before the task starts.
