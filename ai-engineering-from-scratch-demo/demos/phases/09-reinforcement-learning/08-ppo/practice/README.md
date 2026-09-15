<!-- generated:start -->
# 09-reinforcement-learning / 08-ppo

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/08-ppo/) · upstream spec
`phases/09-reinforcement-learning/08-ppo/docs/en.md`

```bash
uv run demo practice run 08-ppo --ex 1
uv run demo explain 08-ppo --ex 1
uv run pytest demos/phases/09-reinforcement-learning/08-ppo
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run PPO on 4×4 GridWorld with `ε=0.2, K=4`. Compare sample efficiency to A2C (one epoch… | code | T1 | `ex01_matching_env_steps_does_not_match_updates.py` |
| 2 | Medium. Sweep `K ∈ {1, 4, 10, 30}`. Plot return vs env steps and track mean KL per update. At… | code | T1 | `ex02_it_never_explodes_because_that_is_what_the_clip_is_for.py` |
| 3 | Hard. Replace the clipped surrogate with an adaptive KL penalty (`β` doubled if `KL > 2·targe… | code | T1 | `ex03_the_penalty_turns_itself_off_by_nine_orders.py` |
<!-- generated:end -->

## Answers

Three exercises, and all three turn on the same thing: **the shipped configuration
sits in the region where PPO's own safety mechanism does nothing.** Exercise 2 asks
where KL explodes and the answer is nowhere — until you delete the clip, at which
point it explodes exactly where expected. Exercise 3 asks you to replace the clip
with an adaptive penalty, and the penalty halves itself into nonexistence within
40 updates because there is nothing for it to constrain.

No new algorithm is written anywhere in this pack. "A2C, one epoch per rollout" is
`ppo_update(epochs=1)`; exercise 3 replaces exactly one term and calls the rollout,
GAE, normalisation and critic from the lesson's own module.

All three are **T1**: 12 seeds × 60 updates, about 4 s, 37 s and 9 s.

### 1 — matching env steps does not match updates, and PPO gets both

Each arm run until it has consumed a 4,000-step budget:

| | evaluated return | updates completed | env steps |
|---|---:|---:|---:|
| PPO (K=4) | **−6.25** | **51.6** | 4,024 |
| A2C (K=1) | −7.81 | 34.6 | 4,037 |

**ANSWER: PPO by 1.56, at a matched budget.**

**FINDING: matching env steps hands PPO *more* updates, not fewer.** A better
policy reaches the terminal sooner, so its rollouts are shorter and more of them
fit in a fixed step count. The arm that is learning faster is also charged less per
update — the opposite of the handicap "matched env steps" sounds like it applies.

**FINDING: at matched *updates* the mismatch runs the other way, and PPO still
wins.** Given 60 updates each, A2C consumes **5,468** env steps to PPO's **4,444**
— 23% more — and still finishes worse (−6.57 against −6.18). Neither budget is
neutral; the exercise's phrasing settles part of the answer before the run starts.

**FINDING: `gae` bleeds across episode and environment boundaries.**
`collect_rollout` concatenates **8 independent environments** into one flat buffer,
and `gae` never resets its accumulator:

```python
gae_val = delta + gamma * lam * gae_val      # never reset, even at done
```

A truncated episode bootstraps from the next environment's first state, and the
accumulator carries across the seam. Against the same recursion reset at each
boundary, the worst per-step advantage differs by **15.81**, on a board where |V|
is about 6. Only **87 of 6,034** buffered steps carry a `done` flag, so most
boundaries are truncations the recursion cannot see. Both arms inherit it equally,
so the comparison above survives — but it is a comparison, not a measurement of
PPO.

### 2 — it never explodes, because that is what the clip is for

| K | evaluated return | mean KL | KL (last 10) | clip fraction |
|---:|---:|---:|---:|---:|
| 1 | −6.57 | 0.0015 | 0.0006 | 0.002 |
| 4 | −6.18 | 0.0069 | 0.0038 | 0.048 |
| 10 | −6.12 | 0.0103 | 0.0052 | 0.090 |
| 30 | −6.11 | 0.0138 | **0.0047** | 0.136 |

**ANSWER: at no `K` in the sweep.** A 30× increase in epochs buys a 9× increase in
KL, to a value still under a typical 0.02 target — and over the last ten updates
the KL *falls* from K=10 to K=30.

**FINDING: remove the clip and it explodes exactly where the exercise expects.**

| K=30 | mean KL | return |
|---|---:|---:|
| clipped (ε=0.2) | 0.0138 | −6.11 |
| **no clip** (ε=1e9) | **0.5046** | −6.87 |

**37×.** A safety mechanism that is working is indistinguishable from one that is
unnecessary until you remove it — which is the only way to answer this exercise
honestly.

**FINDING: at K=4 the clip is doing nothing, and costs a little.** Clipped −6.18
against unclipped −6.08, mean KL 0.0069 against 0.0076. The shipped configuration
sits in the region where its own mechanism is inert.

**FINDING: return is flat from K=4 onward.** K=1→4 is worth 0.38; K=4→30 is worth
**0.07** for 7× the gradient work per env step. The sweep's whole useful range is
between its first two points.

**FINDING: the clip fraction is the axis that actually moves.** 0.002 → 0.136, a
**61×** rise, while KL rises 9× and return moves 0.45. More epochs really do push
the policy away from the one that collected the data; the clip converts that
pressure into *discarded samples* rather than divergence, which is why the KL
column stays flat.

### 3 — the penalty turns itself off, by nine orders of magnitude

Only the surrogate is replaced: `ratio·A` with a hard clip becomes
`ratio·A − β·KL`, whose logit gradient is `(ratio·A + β)·∇log π`. β starts at 1.0,
target 0.01, with the doubling rule the exercise states.

| | return | mean KL | would-clip |
|---|---:|---:|---:|
| adaptive KL penalty | **−6.06** | 0.0051 | 3.5% |
| clipped PPO (ε=0.2) | −6.18 | 0.0069 | — |
| no constraint (ε=1e9) | −6.08 | 0.0076 | — |

**ANSWER: the penalty matches the clip and then removes itself.**

**FINDING: β decays to 3.5e-10.**

| update | 1 | 5 | 20 | 40 | 60 |
|---|---:|---:|---:|---:|---:|
| median β | 0.50 | 0.25 | 1.2e-02 | 4.8e-06 | **3.5e-10** |

The halving rule fires whenever `KL < target/2 = 0.005`, which is most updates. By
the end the penalty term is **2.9e9×** smaller than it began: the arm is running
unconstrained policy gradient with normalised advantages, and the mechanism the
exercise asked you to build is no longer present.

**FINDING: which is fine, because unconstrained also works at this K.** All three
— clipped, penalised, and no constraint whatsoever — land within **0.12** of one
another. "Compare final return, stability, and clip-free-ness" has nothing to
separate at K=4.

**FINDING: the comparison only becomes real where exercise 2 found the clip
working.** Unconstrained at K=30 the mean KL is 0.30, 44× the clipped K=4 figure.
That is the regime the adaptive rule exists for — and it can only act there if β
has not already been halved into irrelevance on the easy updates that came first.
