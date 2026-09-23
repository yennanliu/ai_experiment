<!-- generated:start -->
# 15-autonomous-systems / 05-ai-scientist-v2

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/05-ai-scientist-v2/) · upstream spec
`phases/15-autonomous-systems/05-ai-scientist-v2/docs/en.md`

```bash
uv run demo practice run 05-ai-scientist-v2 --ex 1
uv run demo explain 05-ai-scientist-v2 --ex 1
uv run pytest demos/phases/15-autonomous-systems/05-ai-scientist-v2
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` with default parameters. What fraction of loop runs produce a "clean" pape… | code | T0 | `ex01_the_polish_stage_does_not_polish_anything.py` |
| 2 | The defaults already use Beel et al.'s 42% / 25%. Re-run with `--experiment-failure 0.20 --no… | code | T0 | `ex02_better_retries_make_the_output_worse.py` |
| 3 | Read Sakana's AI Scientist v2 repo README on sandbox requirements. Name two additional restri… | explain | T0 | prose, below |
| 4 | Read Beel et al. Section 4 on presentation-quality gap. Design one additional evaluator that… | code | T0 | `ex04_the_detector_is_a_field_the_loop_computes_and_throws_away.py` |
| 5 | Propose a human-review protocol for research-agent outputs that scales better than "a PhD rea… | code | T0 | `ex05_ninety_percent_recall_costs_eighty_one_percent_of_a_full_read.py` |
<!-- generated:end -->

## Answers

### 1 — the polish stage does not polish anything

Twenty thousand trials at the lesson's own `DEFAULT_SEED`, because a
1000-trial share moves in the second digit:

| | share of runs | share of submissions |
|---|---:|---:|
| submitted | 34.39% | — |
| clean (novel + valid) | 18.52% | 53.87% |
| polished-but-flawed | 15.87% | 46.13% |
| …with an experiment flaw | 9.79% | 28.48% |
| …novelty flaw alone | 6.07% | 17.65% |

So: **18.5% of runs are clean**, and **9.8% carry an experiment flaw**. But
the second question names a mechanism, and the mechanism does not exist.

**`polish_masks_weakness` is inert.** `run_one` assigns
`polished_hides_weakness` and never reads it — it is one of exactly two names
in the function that appear once (the other is the `LoopConfig` type). Setting
the parameter to 0.0, 0.7 or 1.0 leaves submissions at 0.3439 and the flawed
share at 0.4613, identical to four digits, because the `random.random()` call
happens whatever the value is and the comparison is discarded. One of the
config's six fields changes nothing at all — and it is the one the lesson's
headline is about.

**The flaw is decided at the retry.** Every retry-recovered experiment carries
a residual flaw by construction, so the share of papers reaching polish with an
experiment flaw is `f·r / (1 − f(1−r))` = 0.2848, which the simulation
reproduces exactly. Whatever the figure critique does, it happens downstream of
a decision already made.

**And the biggest filter has no quality signal in it.** Abandonment splits
19.2% at the experiment, 11.9% at the writeup and 34.5% at the internal
reviewer. The coin-flip reviewer discards more than the other two stages
together, and its accept probability is independent of both flaw flags — the
loop's largest filter filters at random.

### 2 — better retries make the output worse

| run | submissions | flawed / submissions | flawed / runs |
|---|---:|---:|---:|
| `--experiment-failure 0.20 --novelty-mislabel 0.10` | 38.46% | 20.76% | 7.99% |
| defaults (0.42 / 0.25) | 34.39% | 46.13% | 15.87% |
| `--experiment-failure 0.60 --novelty-mislabel 0.40` | 31.24% | 66.72% | 20.84% |

**The polished-but-flawed share shifts 20.8% → 66.7% of submissions, a 3.2×**,
and 8.0% → 20.8% of runs, a 2.6×. The submission rate moves the *other* way:
the pessimistic configuration ships fewer papers and a much larger fraction of
bad ones, so the two effects compound rather than trade off.

The closed form `1 − (1−m)(1 − f·r/(1 − f(1−r)))` gives 0.2088, 0.4636 and
0.6712 against the three simulated values, worst gap 0.0040 — which is the
Monte Carlo error on 20000 trials, so the whole exercise can be answered
without running anything.

**The two knobs act on different denominators.** `novelty_mislabel` never gates
a stage — it flips a flag and the run continues either way. `experiment_failure`
both gates and flaws: it removes runs at the experiment stage *and* marks the
survivors. That is why the submission rate falls even though nothing in the
config mentions submissions.

**And `retry_recovery` is the lever nobody is asked to move.** A recovered
experiment becomes a *flawed submission*; an unrecovered one is abandoned. So
sweeping recovery at the Beel defaults:

| `retry_recovery` | 0.0 | 0.25 | 0.55 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|
| flawed / submissions | 0.248 | 0.359 | 0.461 | 0.517 | 0.563 |

Making the agent better at repairing its own failed experiments monotonically
makes the papers worse. It is the one parameter of the six the exercises never
touch, and it is the one whose sign is surprising.

### 3 — two restrictions beyond Docker

*Draws on "The sandbox-escape concern".*

**Default-deny egress with an allowlist, not an allow-by-default network.** The
README's named risks are "dangerous packages, uncontrolled web access, and
spawning of unintended processes", and Docker addresses none of the first two:
a container with a default bridge network reaches the whole internet and PyPI,
so an LLM-written `pip install` is an arbitrary-code-execution path that
survives the isolation entirely. The restriction is a proxy that permits only
the package index, only a pinned mirror of it, and nothing else — with the
download hashes recorded so the run can be reproduced and audited afterwards.

**A syscall filter and hard resource ceilings, for a run measured in days.**
Docker shares the host kernel, which is exactly the surface a multi-day
unattended process has time to find; the lesson names the stronger tier itself —
"seccomp / gVisor preferred". Paired with that: a wall-clock deadline, a
process-count ceiling and a spend cap enforced by the orchestrator rather than
by the agent, because a run that lasts days will hit *some* limit and the only
question is whether the limit is one you chose. Both restrictions share a
property worth stating: they are enforced outside the agent's namespace, which
is the same argument Lesson 4 makes about evaluators.

### 4 — the detector is a field the loop computes and throws away

**The evaluator: a provenance check.** Record how the experiment reached its
result, and refuse any paper whose experiment was retry-recovered without an
independent re-run. It runs beside the existing stages on something they
already produce, which is what "additional" has to mean.

In this loop a residual flaw is *defined* as a retry-recovered experiment, so
the flag is a complete detector for that class: of 6878 submissions, 3173 are
flawed, and the flag catches 1959 of them — **recall 0.617 overall, recall
1.000 and precision 1.000 on the experiment-flaw class** — with no model,
reviewer or figure involved.

**The loop already computes the signal.** `run_one` binds `recovered` and reads
it once, to decide whether to abandon; zero of `Outcome`'s six fields record
that a retry happened. The evaluator the exercise asks for needs one boolean
that exists as a local variable and is dropped on return.

**What provenance cannot reach.** The remaining 1214 flawed submissions —
38.3% — carry a novelty flaw alone. `has_novelty_flaw` is drawn before any
stage runs and never gates one, so no downstream artifact correlates with it:
same paper, same figures, same retries. No presentation-quality evaluator
reaches that class at any threshold. Saying so is the difference between an
evaluator and a claim, and it is the half of the problem that still needs a
literature search rather than a log.

**And the figures are the wrong place to look**, because the stage the exercise
points at writes nothing. The presentation-quality gap in this simulator is
entirely upstream of presentation.

### 5 — ninety percent recall costs eighty-one percent of a full read

**The protocol.** Read every submission whose experiment was retry-recovered.
Sample the rest at rate `s`. One free parameter, priced on the same simulator:

| `s` | reviewer effort | flaws caught |
|---|---:|---:|
| 0.00 | 28.5% | 61.7% |
| 0.25 | 46.4% | 71.3% |
| 0.50 | 64.2% | 80.9% |
| 1.00 | 100% | 100% |

The first two thirds of recall cost under a third of the effort.

**The last third costs nearly all of the budget.** Reaching 90% recall needs
`s = 0.739`, which is 81.3% of a full read: the protocol saves 18.7% and no
more. The sampled arm is uniform in the flaw rate, so the second arm buys
recall at exactly the base rate and there is no cleverer place to point it.
Any protocol claiming better than that on this distribution is either using a
signal this loop does not emit, or is not measuring recall.

**The bottleneck is a ratio, and only one side of it moves.** The loop submits
one paper per 2.91 runs, and runs are compute. Reviewers are not. So the real
parameter is not `s` — it is reviewer-hours per submission, and the honest form
of the protocol pins `s` to a fixed review budget and lets recall fall out,
rather than choosing a recall target and discovering the bill.

**And improving the agent makes the cheap arm expensive.** Raising
`retry_recovery` to 1.0 moves the flagged share from 28.5% to 41.7% of
submissions, so the zero-sampling arm costs 41.7% for 74.2% recall. A better
agent does not shrink the review queue here; it moves the queue into the arm
that has to be read in full — which is the same result as exercise 2, seen from
the reviewer's side of the desk.
