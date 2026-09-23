<!-- generated:start -->
# 16-multi-agent-and-swarms / 07-society-of-mind-debate

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/07-society-of-mind-debate/) · upstream spec
`phases/16-multi-agent-and-swarms/07-society-of-mind-debate/docs/en.md`

```bash
uv run demo practice run 07-society-of-mind-debate --ex 1
uv run demo explain 07-society-of-mind-debate --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/07-society-of-mind-debate
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`, then set the round count to 5 and watch diminishing returns. At which rou… | code | T0 | `ex01_the_debate_ends_further_from_the_truth_than_it_started.py` |
| 2 | Add a fourth agent with an adversarial role: always disagree with the current majority. Does… | code | T0 | `ex02_the_adversary_breaks_convergence_under_both_update_rules.py` |
| 3 | Plot (print) the agreement score per round (fraction of agents on the majority answer). When… | code | T0 | `ex03_agreement_is_a_measure_of_spread_and_says_nothing_about_truth.py` |
| 4 | Read Du et al. Section 4 ablations. Replicate the "agents-only" vs "rounds-only" vs "both" re… | code | T0 | `ex04_the_rounds_knob_is_inert_once_the_update_is_correct.py` |
| 5 | Read "Should we be going MAD?" (arXiv:2311.17371) and list two debate variants beyond round-r… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the debate ends further from the truth than it started

Round 1 — but only once the update is fixed, and the diminishing returns the
exercise asks you to watch are not diminishing. They are negative.

| round | error vs truth (shipped) | error vs truth (simultaneous) |
|---:|---:|---:|
| control (round 0) | 1.833 | 1.833 |
| 1 | **2.242** | **0.889** |
| 2 | 2.431 | 0.889 |
| 3 | 2.484 | 0.889 |
| 4 | 2.497 | 0.889 |
| 5 | 2.499 | 0.889 |
| 6 | 2.500 | 0.889 |

The cause is one line. `run_debate` collects `(a, others)` pairs where `others`
holds **live references**, then calls `a.revise(others)` in sequence — so the
second agent averages against the first agent's *already revised* answer. That
is a Gauss-Seidel sweep. Du et al.'s algorithm is simultaneous: every agent
reads the same snapshot.

Fixed, the debate reaches the confidence-weighted mean in **one** round and
stays there to thirteen decimal places, because that mean is a fixed point of
the update. So the honest answer to "at which round does additional
convergence stop" is **round 1**, and the reason is not a plateau — it is that
there is nothing left to do.

As shipped it is worse than not debating: three rounds leave the team **36%**
further from the truth than the module's own round-0 control. And that is not
an artefact of these particular confidences. Permuting the three confidence
values over the three agents gives final errors of 1.92, 2.48, 3.30, 4.27,
5.24 and 5.65 — worse than the 1.83 control in **6 of 6** assignments, because
the sweep drags consensus toward whichever agent the list puts first, and here
that is the one furthest below the answer.

All three printed takeaways are contradicted by the table printed above them.
"1 round of exchange cuts the error most" — round 1 raises it. "Rounds 2–3
compound" — they compound the error. "Beyond round 3 the gain per round
shrinks" — the per-round change is **+0.19, +0.05, +0.01**, a plateau at the
wrong number.

### 2 — the adversary breaks convergence under both update rules

It breaks it, and the breakage is unbounded. The adversary answers
`mean + 5` — always five above whatever the others just agreed on:

| | round 5 error | agreement, any round |
|---|---:|---:|
| shipped sweep | 9.01 | 0.00 |
| simultaneous | 5.81 | 0.00 |

Agreement never leaves 0.00 in ten rounds under either rule. Averaging has no
fixed point once one participant's answer is *defined* as an offset from the
average — each round the mean moves up, so the adversary moves up, so the mean
moves up.

The control matters here. Against the module's round-0 baseline of 1.83 the
adversary looks bad; against the correct three-agent settle of **0.889**, which
is what a working debate actually achieves, it costs **6.5×** the error. Run
the experiment on the broken sweep alone and you cannot separate the
adversary's damage from the sweep's.

Worth being precise about the framing: convergence and accuracy are different
failures, and this adversary causes both, so the distinction does not show up.
An adversary that pulled toward a *fixed* wrong value would converge
beautifully onto it — agreement 1.00, error large — which is the more dangerous
shape and the one exercise 3 is about.

Finally, "add a fourth agent" is a structural change, not a list entry.
`DebateAgent.revise` is a concrete method computing a weighted mean, with two
parameters and no abstract base. An adversarial role cannot *be* a
`DebateAgent`; it has to be a separate class that merely looks like one.

### 3 — agreement is a measure of spread and says nothing about truth

Agreement hits 1.00 at **round 3** as shipped, and **round 1** once the update
is fixed. No, it is not equivalent to correct:

| | round agreement hits 1.00 | error at that round |
|---|---:|---:|
| shipped | 3 | **2.484** and rising |
| simultaneous | 1 | **0.889**, forever |

Two runs, full agreement in both, two different wrong answers.

The function is also not measuring what the exercise calls it. The exercise
says "fraction of agents on the majority answer"; `agreement_score` computes
the fraction within `tol` of the **mean**. With three agents at 38, 42.5 and
51 there is no majority answer — every answer is unique — yet it returns a
number. It is a spread statistic wearing a voting name.

And spread only moves one way here. Every revision is a convex combination of
the current answers, so the score is monotone — `0.00, 0.00, 1.00, 1.00, ...`
— and 1.00 is absorbing. The metric cannot report a debate going wrong, which
is exactly what this one is doing the whole time it is reporting 1.00.

One more thing to know before trusting the number: `tol = 0.1` is **absolute**.
Scale the whole problem by 100 — the same answers and the same disagreements,
denominated in cents rather than dollars — and the round at which agreement
hits 1.00 moves from **3** to **6**. The same debate is judged converged or
not according to its units.

### 4 — the rounds knob is inert once the update is correct

The replication fails under both rules, for opposite reasons:

| arm | shipped | simultaneous |
|---|---:|---:|
| control (3 agents, 0 rounds) | 1.833 | 1.833 |
| agents-only (6 agents, 1 round) | 1.504 | 0.944 |
| rounds-only (3 agents, 5 rounds) | **2.499** | 0.889 |
| both (6 agents, 5 rounds) | 1.537 | **0.944** |

Under the shipped sweep the rounds knob has the **wrong sign** — rounds-only is
worse than not debating — and "both" is worse than agents alone, because the
rounds are actively undoing what the extra agents contribute.

Under a simultaneous update every arm beats the control, but `both` equals
`agents-only` to four decimals: a difference of **0.0000**. The weighted mean
is a fixed point, so rounds 2 through 5 change nothing at all. The two knobs
the lesson calls independent are not both live; an ablation needs two factors
that can each move the outcome, and here one of them cannot.

The agents axis is not measuring what it looks like either. The six-agent arm
scores **worse** than the three-agent one (0.944 against 0.889) because the
three extra agents start further from the answer. Whether "more agents" helps
is decided entirely by where the extra agents begin, which this module fixes by
hand — so the arm measures the author's choice of initial answers rather than a
property of debate.

Which is the deeper problem: the ablation cannot be run as written at all.
`fresh_team(seed)` calls `random.seed(seed)` and then constructs three agents
from literals. `random` is never sampled — seeds 1 and 2 produce **identical**
teams, so the module's two demo runs debate the same starting state, and there
is no mechanism to draw the extra agents an agents-only arm needs. `math` is
imported and used zero times.

### 5 — judge-led and chain-of-debate

*Draws on "Heterogeneous debate".*

Two variants beyond round-robin:

**Judge-led debate.** Agents argue; a separate judge — not a participant —
reads the transcript and returns the answer. The structural difference is that
consensus stops being the aggregation rule. In round-robin the answer *is*
whatever the agents converge on, which is why exercise 3's metric can report
1.00 on a wrong number: agreement and answer are the same object. A judge
breaks that identity, so a debate can end 2–1 and still be scored on the
argument rather than the count. It also bounds the run: the judge decides when
to stop, instead of a round cap chosen in advance.

**Chain-of-debate.** Agents are arranged in a sequence rather than a circle:
each sees only its predecessor's output and passes to its successor. Cheaper
than round-robin — context grows by one contribution per hop instead of N —
but it forfeits the property that makes debate work, since an error introduced
early is never seen by anyone who could contradict it, only by agents that
inherit it. It is the same hazard as lesson 06's hierarchy, where the question
is gone one hand-off down.

The MAD paper's central caution applies directly to this module. A weak
participant drags consensus toward its wrong answer, because averaging weights
it like everyone else. Here that is literal: `revise` is a confidence-weighted
mean, and confidence is assigned by the author, not earned. Exercise 1 measures
the consequence — permute the confidences and the debate is worse than the
control in all six assignments. Heterogeneous debate helps against *correlated*
errors, which is a different failure from a single weak voice, and neither
variant above fixes the weighting.
