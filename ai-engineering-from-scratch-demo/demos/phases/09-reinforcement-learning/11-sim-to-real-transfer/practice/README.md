<!-- generated:start -->
# 09-reinforcement-learning / 11-sim-to-real-transfer

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/11-sim-to-real-transfer/) · upstream spec
`phases/09-reinforcement-learning/11-sim-to-real-transfer/docs/en.md`

```bash
uv run demo practice run 11-sim-to-real-transfer --ex 1
uv run demo explain 11-sim-to-real-transfer --ex 1
uv run pytest demos/phases/09-reinforcement-learning/11-sim-to-real-transfer
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Train a Q-learning agent on the fixed-slip GridWorld (slip=0.0). Evaluate on slip ∈ {0.… | code | T0 | `ex01_the_curve_is_the_environment_not_the_agent.py` |
| 2 | Medium. Train a DR Q-learning agent sampling `slip ~ Uniform[0, 0.3]`. Evaluate the same swee… | code | T0 | `ex02_it_buys_three_percent_of_return_and_sixty_of_the_gap.py` |
| 3 | Hard. Implement a curriculum: start with slip=0.0, widen the DR range every time the policy h… | code | T1 | `ex03_the_curriculum_wins_without_ever_reaching_the_target.py` |
<!-- generated:end -->

## Answers

All three exercises measure a policy against a moving environment, and in all three
the number that matters only appears once you write down what the environment alone
was going to do. The lesson's `evaluate` samples 200 episodes; the greedy policy and
the slip together define a Markov chain, so every value below is solved exactly
instead — at slip 0.1 the sampling noise is larger than the entire effect being
measured.

Exercises 1 and 2 are **T0** (under a second); exercise 3 is **T1** (about a minute,
because its stopping rule solves the chain every 25 episodes).

### 1 — most of the curve is the environment, not the agent

| slip | trained agent | best possible | optimality gap |
|---:|---:|---:|---:|
| 0.0 | −8.00 | −8.00 | **0.000** |
| 0.1 | −8.90 | −8.79 | 0.111 |
| 0.3 | −11.57 | −11.15 | 0.428 |
| 0.5 | −16.60 | −15.67 | 0.931 |

**ANSWER: −8.00, −8.90, −11.57, −16.60.** The agent loses **8.60** across the sweep.

**FINDING: 7.67 of that 8.60 is the environment getting harder.** Only **0.93** — 11%
of the decline the plot shows — is the agent being wrong. A curve that falls with
slip proves nothing until it is placed against the curve that *has* to fall.

**FINDING: the degradation is entirely out-of-distribution, by construction.** At the
slip it trained on, the agent is exactly optimal (gap 0.0e+00). Every point of the gap
appears only at slips it never saw — which is what makes this a sim-to-real
measurement rather than a training-quality one.

**FINDING: sampling 200 episodes is not needed, and not harmless.** The lesson's
`evaluate` agrees with the exact value to within **0.14** — but the whole optimality
gap at slip 0.1 is **0.111**. At the low end of the sweep, the measurement noise
exceeds the effect being measured.

### 2 — it buys 3% of the return and 60% of the gap

| slip | fixed(0.0) | DR[0, 0.3] | optimum | gap: fixed → DR | closed |
|---:|---:|---:|---:|---:|---:|
| 0.0 | −8.000 | −8.000 | −8.000 | 0.000 → 0.000 | — |
| 0.1 | −8.902 | −8.857 | −8.791 | 0.111 → 0.066 | 41% |
| 0.3 | −11.574 | −11.343 | −11.147 | 0.428 → 0.196 | 54% |
| 0.5 | −16.603 | −16.043 | −15.672 | 0.931 → 0.371 | **60%** |

**ANSWER: 0.56 of return at slip=0.5 — 3.4% of the return, or 60% of the distance to
optimal.** The two denominators the question admits disagree by a factor of **18**.

**FINDING: DR closes a similar share at every slip, inside its range and outside it.**
slip=0.3 is the edge of the training distribution and slip=0.5 is well beyond it, and
nothing distinguishes them. Whatever DR learned generalises at the same rate in and
out of distribution.

**FINDING: the two agents are indistinguishable where they were trained.** Both return
*exactly* −8.000 at slip=0 — both are shortest paths. DR's entire benefit is a
different **choice among equally optimal policies**, so no measurement taken in the
training environment can see it.

**MECHANISM: they disagree on 6 of 24 cells, and the disagreement is the route.**

```text
fixed-slip                DR[0, 0.3]
v v v v v                 v > v v v
v v v v v                 > v v v v
v v v v v                 v > > > v
v v v v v                 v v > v v
> > > > .                 > > > > .
```

The fixed-slip agent walks the boundary — straight down the left column, then right
along the bottom. The DR agent takes a staircase through the interior. Both are 8
steps when nothing slips.

### 3 — the curriculum wins without ever reaching the target range

A note on the stopping rule first, because it is easy to get backwards: **"90% of
optimal" cannot be `value >= 0.9 * optimal` when returns are negative** — that demands
a policy *better* than optimal and never fires. The criterion used here is
`value >= optimal / 0.9`.

| arm | env steps | episodes | steps/episode | slip range when it passed |
|---|---:|---:|---:|---:|
| fixed DR [0, 0.3] | 7,067 | 325 | 21.7 | — |
| **curriculum** | **5,652** | 300 | **18.8** | **[0, 0.05]** |

**ANSWER: 5,652 against 7,067 — the curriculum needs 20% fewer steps.** Both arms
reach the target on 8 of 8 seeds.

**FINDING: it wins on step count while barely winning on episodes.** 300 against 325.
The saving is in episode *length*: an episode at slip 0 terminates sooner than one
drawn from the full range. The curriculum practises in a **cheaper** environment, not
a faster one — which is a real saving, but not the one "curriculum learning" usually
claims.

**FINDING: it passes while still six times narrower than the target.** When the
curriculum clears the rule at slip=0.3, its own ceiling is **0.05**. It has never
trained above a sixth of the slip it is being scored on.

**FINDING: the widening rule almost never fires.** Reaching 0.3 in steps of 0.05 would
take six promotions; the run ends after **one**. What the exercise describes as a
curriculum is, on this task, a warm-up at slip 0 followed by an early exit — the
schedule never gets far enough to be a schedule.
