<!-- generated:start -->
# 16-multi-agent-and-swarms / 24-evaluation-coordination-benchmarks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks/) · upstream spec
`phases/16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks/docs/en.md`

```bash
uv run demo practice run 24-evaluation-coordination-benchmarks --ex 1
uv run demo explain 24-evaluation-coordination-benchmarks --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Identify which of the three simulated systems has the best cost-per-miles… | code | T0 | `ex01_b_is_cheapest_per_milestone_on_every_seed_and_never_the_most_accurate.py` |
| 2 | Read MultiAgentBench (arXiv:2503.01935). For your own task domain, decide which of the four t… | explain | T0 | prose, below |
| 3 | Read the SWE-bench Pro paper. What specifically makes it contamination-resistant? Could the s… | explain | T0 | prose, below |
| 4 | Read COMMA's finding on multimodal coordination. Design a simple multimodal coordination task… | code | T0 | `ex04_the_signal_is_the_channel_ablation_not_a_random_constant.py` |
| 5 | Apply the benchmark-claims checklist to one recent multi-agent paper's headline result. What… | code | T0 | `ex05_marbles_three_percent_grades_f_and_the_rubric_grades_one_claim_two_ways.py` |
<!-- generated:end -->

## Answers

### 1 — B is cheapest per milestone on every seed and never the most accurate

**system-B has the best cost per milestone, and no, it is not the most
accurate system — system-A is.**

| system | acc (seen) | acc (held) | cost/task | cost/mil (printed) |
|---|---:|---:|---:|---:|
| A | 0.900 | 0.731 | $0.30 | $0.371 |
| B | 0.550 | 0.681 | $0.12 | **$0.164** |
| C | 0.550 | 0.550 | $0.25 | $0.385 |

The scorecard is one seed, so each ranking was re-run over 400 seeds. B is
cheapest per milestone on **400 of 400**, and A is most accurate on held tasks
on 398. Those two rankings are properties of the systems.

**Second place on cost is a coin flip.** In expectation A costs $0.3825 per
milestone-unit and C $0.3820. A's 20% higher price is bought back almost
exactly by its higher milestone rate, and A is cheaper than C on 46% of
seeds.

**The contamination flag fires on clean systems about one seed in eight.** It
is `seen − held > 0.1` on 40 seen tasks. B and C carry zero contamination and
trip it on 12% and 15% of seeds; contaminated A trips it on 94%. B's printed
−0.13 is the same noise pointing the other way.

**Two of the three takeaways are false at the printed seed.** They are string
literals. "system-B … lowest raw accuracy" — B's held accuracy is 0.681,
above C's 0.550. "system-C sits in the middle" — C is lowest on held accuracy
on 400 of 400 seeds, and at seed 17 it is the most expensive per milestone.

Two unit problems. `cost_per_milestone_held` divides by the milestone *rate*
(milestones ÷ 4), so the column is cost per four milestones. B's true cost
per milestone is **$0.041**. And `random_baseline(rng)` returns 0.15
whatever the rng, so "vs random" is accuracy minus a literal.

### 2 — MARBLE cannot recommend a topology for anything but research

*Draws on "MultiAgentBench (MARBLE) — ACL 2025".*

Take the task domain of this repository: generate a solution, run it, review
the findings. It is coding-shaped.

The paper does not rank topologies for it. §4.3 compares star, chain, tree
and graph in **the research scenario only**. There graph wins "with the best
task performance, planning efficiency, and token usage", tree does poorly,
and star and graph score similarly. The benchmark has six scenarios
(research, Minecraft, database, coding, werewolf, bargaining), but the
topology comparison covers one of them.

So the lesson's "chain best for stepwise-refinement coding" and "star best
for fast-factual consolidation" are not results from this paper. Neither is
"coordination tax past ~4 agents": the scaling ablation (§5, Figure 8) says
more agents lower the overall KPI and names no threshold.

What the paper does justify is narrow. Any-to-any critique pays off on
open-ended research. For a coding pipeline, graph is a guess, and the honest
recommendation is to run MARBLE's own evaluator on your task with all four
topologies. The paper's other headline is also worth noting: gpt-4o-mini
had the highest average task score, so model choice moved results as much as
topology did.

### 3 — the contamination defence is licence plus secrecy, and only secrecy transfers

*Draws on "SWE-bench Pro — the reality check".*

SWE-bench Pro has 1,865 problems from 41 repositories in three sets:

- **public:** 11 repositories;
- **held-out:** 12 repositories, not released;
- **commercial:** 18 proprietary startup codebases, results released but
  problems not.

§3.1 names two defences. The public and held-out sets come "exclusively"
from repositories under strong copyleft (GPL) licences. The commercial set
is code that was never public at all.

These are different kinds of protection. The GPL choice is a **deterrent**:
it relies on model trainers choosing to exclude copyleft code, which the
benchmark cannot check. The held-out and commercial sets are **access
control**: a problem that was never published cannot be in a training
corpus. Only the second transfers to other benchmarks. Keep a private split
built from data you control, publish only the scores, and treat a widening
public-versus-private gap as the contamination signal. Refreshing a private
split works for any benchmark. The licence trick works only where a
deterrent licence exists and trainers honour it.

Two of the lesson's numbers need their sources. The "~23%" is Claude Opus 4.1
on the public set under a $2, 50-turn cap (§8 and appendix Table 5). The main
table's best public-set score is 43.6% (Claude Sonnet 4.5), and the best
commercial-set score is 17.8% (Claude Opus 4.1). And every number in the
paper is a single agent in the SWE-Agent scaffold, since the paper tests no
multi-agent system. The "~30–40 point delta" from agent teams is not
something SWE-bench Pro measured.

### 4 — the signal is the channel ablation, not a random constant

The task is a panel of 4 coloured wires over 3 stages. The solver sees the
panel (the image). The expert holds the rule for which wire to cut (the
manual) and never sees the panel. A wrong cut ends the episode. Partial
success is stages cleared ÷ 3, COMMA's PSR.

**The useful signal is the channel-on minus channel-off gap, on the same
episodes, with enough episodes to resolve it.** With the channel cut the
solver can only guess. That is COMMA's own random baseline, "the solver
agent chooses actions uniformly at random", so the baseline becomes a
measured arm of the task. A pair whose instructions arrive with probability
0.8 scores PSR 0.725, against 0.109 channel-off.

The random baseline depends on the task: exactly (1/4)³ = 0.0156 for success
and 0.109 for partial success. The lesson's `random_baseline` returns 0.15
for every rng and every task, which is right for neither.

**The episode count decides which claims the benchmark can make.** For a
two-arm test at 5% significance and 80% power:

| gap to resolve | episodes per arm |
|---|---:|
| COMMA's GPT-4o over random, 41.74 vs 18.70 PSR | 62 |
| a 3-point gap at 50% | 4,356 |
| smallest gap the lesson's 160 held tasks resolve | 0.16 |

The lesson attributes the random-baseline result to GPT-4o, but COMMA says
otherwise. Its abstract names *chain-of-thought open models* (R1-Onevision,
LLaVA-CoT) as the ones struggling to beat random. GPT-4o scores 41.74 ± 1.4
PSR against the random baseline's 18.70 ± 1.1 (Table 1), more than double.

### 5 — MARBLE's 3% grades F, and the rubric grades one claim two ways

The checklist is the lesson's own grader, `outputs/skill-benchmark-reader.md`,
encoded as a function. The claim graded is MultiAgentBench's "cognitive
planning improves milestone achievement rates by 3%" (abstract):

| check | verdict | why |
|---|---|---|
| benchmark + split | pass | MultiAgentBench, research scenario |
| contamination | fail | the model behind Figure 6 is not identified, so its cutoff cannot be checked |
| baseline | fail | compared with vanilla, CoT and group discussion (§3.1.1): an ablation of the same system |
| statistics | fail | no run count, deviation, interval or test |
| diversity | fail | research scenario only (§4.3) |
| cost | fail | token cost reported for protocols (Figure 5), not planning strategies |

**Grade: F.** Five weaknesses is a D by count, and "no statistics" is a
disqualifier. The paper is ACL 2025, so the preprint downgrade does not
apply. It is a suggestive ablation, not evidence that cognitive planning
helps.

The lesson's own scorecard fails the same rubric. It names the held split,
checks contamination and shows cost. But it runs 40 seen tasks with no
interval, uses the literal 0.15 as its random baseline, and covers one task
family. That is D by count and F by the hard rejects.

**The rubric grades one claim two ways.** Letters are counted ("B: one
weakness"), while "claims without baseline comparison" is a hard reject. A
claim missing only its baseline is therefore B or F depending on which rule
is read first. The encoding applies hard rejects first, the only order under
which they mean anything.
