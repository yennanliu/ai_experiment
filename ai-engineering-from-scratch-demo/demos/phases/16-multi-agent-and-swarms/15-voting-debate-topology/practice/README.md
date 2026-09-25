<!-- generated:start -->
# 16-multi-agent-and-swarms / 15-voting-debate-topology

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/15-voting-debate-topology/) · upstream spec
`phases/16-multi-agent-and-swarms/15-voting-debate-topology/docs/en.md`

```bash
uv run demo practice run 15-voting-debate-topology --ex 1
uv run demo explain 15-voting-debate-topology --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/15-voting-debate-topology
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Plot the coordination-tax curve for graph topology: accuracy vs N, tokens… | code | T0 | `ex01_the_debate_rounds_buy_tokens_and_never_move_the_answer.py` |
| 2 | Implement A-HMAD: three agents with deliberately different biases. How does the all-same-bias… | code | T0 | `ex02_heterogeneity_only_helps_by_splitting_the_wrong_vote.py` |
| 3 | Add a "judge" role to the graph topology that does not vote, only scores the final consensus.… | code | T0 | `ex03_a_judge_that_scores_consensus_is_a_conformity_meter.py` |
| 4 | Read the AgentVerse paper (ICLR 2024). Identify which emergent behavior your implementation e… | code | T0 | `ex04_conformity_is_a_literal_and_no_setting_of_it_changes_an_answer.py` |
| 5 | Read MultiAgentBench (arXiv:2503.01935) Section 4 (topology experiments). Reproduce the "grap… | code | T0 | `ex05_graph_wins_by_one_voter_and_loses_the_token_column.py` |
<!-- generated:end -->

## Answers

### 1 — the debate rounds buy tokens and never move the answer

**The curve never inflects.** Each agent is either right, with its base
accuracy, or gives its bias. So the 2^N outcomes can be enumerated and the
curve computed exactly instead of sampled. Homogeneous agents, 800 tokens
per agent:

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.72 | 0.72 | 0.8087 | 0.8087 | 0.8624 | 0.8624 | 0.8984 | 0.8984 | 0.9238 |
| tokens | 800 | 1600 | 2400 | 3200 | 4000 | 4800 | 5600 | 6400 | 7200 |

Accuracy is concave from the first agent — odd-N gains of +0.0887, +0.0537,
+0.0360, +0.0254 — and tokens are linear. Every agent buys less than the one
before, and there is no knee at 4. The coordination tax cannot appear:
`tokens_per_call` is a constant 400 however many peers an agent reads. The
reference bench's 0.82, 0.905, 0.92 land within sampling error of the exact
values.

**The debate rounds never change the answer.** A dissenter moves to the
current majority with probability 0.4, and moving a vote to the argmax
cannot move the argmax. Rounds 1, 2, 3 and 4 give the same final answer on
**600 of 600** seeded runs. Graph is one plurality vote of all N agents, and
its second round doubles the tokens for nothing.

**Even N buys zero accuracy.** A tie goes to whichever answer `majority` saw
first, which is agent 0's. So N=4 scores exactly what N=3 does, and the same
holds for 6 and 5 and for 8 and 7.

The latency column the lesson promises, `wallclock_simulated`, does not
exist. The table prints `steps`, and graph's is 4 at every N. "Tokens
inflate ~7x over star/N=3" is 5600/1200 = **4.7x**.

### 2 — heterogeneity only helps by splitting the wrong vote

This runs Lesson 14's own monoculture profile through Lesson 14's own
aggregators. The A-HMAD version replaces the three shared-model "42%"
answers with three *different* wrong answers:

| aggregator | monoculture | A-HMAD |
|---|---|---|
| plurality | 42% | **4.2%** |
| CP-WBFT | 42% | **None** |
| DecentLLMs | 42% | **4.2%** |

A-HMAD fixes plurality and DecentLLMs but breaks CP-WBFT. There "4.2%" holds
1.67 of 3.77 confidence, 44% against the 50% threshold, so heterogeneity
turns a confident wrong answer into a refusal.

In the stochastic harness, **heterogeneity helps only by splitting the wrong
vote.** At the same 0.72 accuracy, three same-bias agents score 0.8087 and
three different-bias agents score 0.8652. The whole gain is p(1−p)^2: the
outcome where agent 0 is right and the other two are wrong in different ways.
That 1-1-1 split goes to whoever spoke first. So agent order decides it:
accuracies [0.9, 0.6, 0.6] score 0.936 in that order and 0.816 reversed,
while the same-bias baseline scores 0.792 either way.

The demo's `heterogeneous=True` changes the accuracies as well as the biases.
Its takeaway, "heterogeneous ensembles outperform homogeneous at every
topology/N", fails in **5 of 12** cells of its own table: chain at N=5 and 7,
tree at N=3 and 5, and star at N=3.

### 3 — a judge that scores consensus is a conformity meter

Without ground truth, the only thing a judge can score is agreement: the
fraction of agents that side with the final answer.

**As an observer the judge changes nothing, by construction.** It runs after
the last round. Used as a gate, re-running the debate until agreement is at
least 0.8, it changes behavior only toward *more* conformity. Over 200
debates at N=5, switches rise from 94 to 139, while all 200 final answers and
the 0.905 accuracy stay exactly as they were.

The score measures how long the debate ran, not whether it was right:

| rounds | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| mean judge score | 0.756 | 0.850 | 0.916 | 0.945 |
| accuracy | 0.905 | 0.905 | 0.905 | 0.905 |

It also scores wrong consensus almost as high as right: 0.821 for the 19
wrong finals against 0.853 for the 181 right ones. No threshold on this score
separates them. A judge that could separate them would need the answer, and
then it is a verifier, not a judge.

### 4 — conformity is a literal, and no setting of it changes an answer

*Draws on AgentVerse (ICLR 2024).*

AgentVerse names three emergent behaviors: volunteer, conformity and
destructive. The harness exhibits **conformity**, and only conformity:

- **Volunteer** behavior (time, resource and assistance contributions)
  cannot occur. A `SimAgent` has one method, `answer`, and every agent
  answers the same question, so there is nothing to volunteer for.
- **Conformity** is hardcoded: `rng.random() < 0.4` inside `run_graph`.

There is no prompt to change, so the "prompt change" is a change to that
literal. At 0.0, 0.4 and 1.0 it gives the **same 200 final answers**, at
0.905 accuracy. The opposite, a contrarian rule where majority members defect
to the runner-up with probability 0.4, does move answers: accuracy drops to
0.555, with 85 answers flipped right-to-wrong and 15 wrong-to-right. So
conformity is invisible to the vote it is supposed to distort, and its
opposite is the behavior that does damage.

The paper's own conformity example runs the other way from the lesson's
gloss. In AgentVerse's Minecraft case, Charlie drifts into off-task crafting,
Alice and Bob criticise it, and Charlie "acknowledges his mistake and
re-focuses on the mutual tasks". That is conformity to a *correct* critic.
The lesson's "even when the critic is wrong" is the sycophancy reading. The
lesson also leaves out AgentVerse's destructive behavior: agents harming
other agents or destroying a village library to get materials.

### 5 — graph wins by one voter, and loses the token column

*Draws on MultiAgentBench (arXiv:2503.01935) §4.3.*

The paper's topology result comes from its research scenario, where agents
co-author a 5-question proposal that an LLM scores on a 5-point scale. The
harness has one task: a string that is either "RIGHT" or not. So what can be
reproduced is the *shape* of the result, computed exactly at N=5:

| topology | accuracy | tokens |
|---|---:|---:|
| graph | 0.8624 | 4000 |
| star | 0.8087 | 2000 |
| tree | 0.72 | 2000 |
| chain | 0.72 | 2000 |

**Graph wins accuracy by one voter and loses the token column.** Graph votes
all N agents, while star's hub votes only its N−1 workers. So graph at N=3,
5, 7 scores exactly what star scores at N=4, 6, 8. And graph spends 2x star's
tokens at every N. §4.3 reports graph best on "task performance, planning
efficiency, and token usage", with star and graph at "similar task scores".
The harness reproduces the accuracy ordering and inverts the token one.

**The tree is its left half.** A 1-1 tie between the two sub-consensuses goes
left, so the final answer equals the left half's majority on 600 of 600 runs.
Tree at N=3 and N=5 is one agent, and at N=7 it is a 3-voter panel, yet it
pays for every leaf. The paper also ranks tree last; here it is last by
construction.

**The chain is exactly one agent at every length.** Each link adopts a
differing proposal with probability equal to its own accuracy, whether or not
the proposal is right. That leaves P(right) = p fixed:
p(1 − (1−p)p) + (1−p)p^2 = p. The demo's chain scores of 0.75, 0.78 and 0.71
are sampling noise around 0.72.
